from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal

import pytest
import yaml

from llm_router.evaluation.pricing import UnknownCost
from llm_router.models.registry import ModelCandidate, ModelRegistry
from llm_router.models.utility_predictor import CandidatePrediction, PredictionContractError
from llm_router.routing.candidate_filter import CandidateRequest
from llm_router.routing.policy import (
    FallbackMode,
    PolicyConfig,
    ScoredCandidate,
    UtilityWeights,
    Violation,
    decide,
)
from llm_router.routing.router import Router, RouteStatus

TODAY = date(2026, 10, 10)
REQUEST = CandidateRequest(input_tokens=1_000, max_output_tokens=2_000)


class TablePredictor:
    """Test double: returns fixed predictions. Not a model; never used outside tests."""

    def __init__(self, table: Mapping[str, tuple[float | None, float | None, int | None]]):
        self.table = table
        self.calls = 0

    def predict(
        self, query_features: Mapping[str, object], candidates: Sequence[ModelCandidate]
    ) -> list[CandidatePrediction]:
        self.calls += 1
        return [CandidatePrediction(c.model_id, *self.table[c.model_id]) for c in candidates]


@pytest.fixture
def two_versions(make_candidate, registry_data):
    """One family, two independently selectable versions with different prices."""
    return ModelRegistry.from_mapping(
        registry_data(
            make_candidate(
                "testprov/alpha-1",
                version="1",
                pricing={"input": 1.0, "output": 5.0},
                provider_metadata={"api_model_id": "alpha-1-2026-01-01"},
            ),
            make_candidate(
                "testprov/alpha-2",
                version="2",
                pricing={"input": 3.0, "output": 15.0},
                provider_metadata={"api_model_id": "alpha-2-2026-06-01"},
            ),
        )
    )


def route(registry, table, policy=None, request=REQUEST):
    router = Router(registry, TablePredictor(table), policy or PolicyConfig(), clock=lambda: TODAY)
    return router.route({}, request)


def test_selects_cheapest_version_meeting_constraints(two_versions):
    policy = PolicyConfig(min_quality=0.8, max_latency_p95_ms=5_000)
    cheap_ok = route(
        two_versions,
        {"testprov/alpha-1": (0.85, 1_000, 500), "testprov/alpha-2": (0.95, 2_000, 500)},
        policy,
    )
    assert cheap_ok.status == RouteStatus.SELECTED
    assert cheap_ok.model_id == "testprov/alpha-1"

    cheap_too_weak = route(
        two_versions,
        {"testprov/alpha-1": (0.70, 1_000, 500), "testprov/alpha-2": (0.95, 2_000, 500)},
        policy,
    )
    assert cheap_too_weak.model_id == "testprov/alpha-2"
    assessment = cheap_too_weak.decision.assessments[-1]
    assert assessment.model_id == "testprov/alpha-1"
    assert assessment.violations == (Violation.QUALITY_BELOW_MIN,)


def test_result_carries_exact_provider_model_id(two_versions):
    result = route(
        two_versions, {"testprov/alpha-1": (0.9, 100, 10), "testprov/alpha-2": (0.9, 100, 10)}
    )
    assert result.model_id == "testprov/alpha-1"
    assert result.provider == "testprov"
    assert result.provider_model_id == "alpha-1-2026-01-01"
    assert "lowest predicted cost" in result.explain()


def test_cost_uses_each_versions_own_pricing(two_versions):
    result = route(
        two_versions, {"testprov/alpha-1": (0.9, 100, 1_000), "testprov/alpha-2": (0.9, 100, 1_000)}
    )
    assert result.costs["testprov/alpha-1"].total == Decimal("0.006")  # 1k*$1 + 1k*$5
    assert result.costs["testprov/alpha-2"].total == Decimal("0.018")  # 1k*$3 + 1k*$15


def test_new_registry_entry_needs_no_code_change(make_candidate, registry_data):
    """Any number of families/versions works; nothing in routing names a family."""
    records = [
        make_candidate(
            f"newprov/{family}-{v}", family=family, version=str(v), pricing={"output": p}
        )
        for family, v, p in [("zeta", 1, 9.0), ("zeta", 2, 8.0), ("eta", 7, 7.0), ("theta", 3, 1.0)]
    ]
    registry = ModelRegistry.from_mapping(registry_data(*records))
    table = {r["model_id"]: (0.9, 100.0, 1_000) for r in records}
    result = route(registry, table)
    assert result.model_id == "newprov/theta-3"
    assert len(result.decision.assessments) == 4


def test_no_eligible_candidate_skips_prediction(two_versions):
    predictor = TablePredictor({})
    router = Router(two_versions, predictor, PolicyConfig(), clock=lambda: TODAY)
    result = router.route({}, CandidateRequest(input_tokens=500_000, max_output_tokens=1_000))
    assert result.status == RouteStatus.NO_ELIGIBLE_CANDIDATE
    assert result.candidate is None and result.provider_model_id is None
    assert predictor.calls == 0
    assert len(result.filter_result.excluded) == 2


def test_no_feasible_candidate_is_explicit(two_versions):
    result = route(
        two_versions,
        {"testprov/alpha-1": (0.5, 100, 10), "testprov/alpha-2": (0.6, 100, 10)},
        PolicyConfig(min_quality=0.9),
    )
    assert result.status == RouteStatus.NO_FEASIBLE_CANDIDATE
    assert result.candidate is None
    assert not result.decision.constraints_satisfied


def test_fallback_maximize_utility_is_flagged(two_versions):
    policy = PolicyConfig(
        min_quality=0.9,
        fallback=FallbackMode.MAXIMIZE_UTILITY,
        utility_weights=UtilityWeights(quality=1.0, cost=1.0, latency_ms=0.0),
    )
    result = route(
        two_versions,
        {"testprov/alpha-1": (0.5, 100, 10), "testprov/alpha-2": (0.8, 100, 10)},
        policy,
    )
    assert result.status == RouteStatus.FALLBACK
    assert result.model_id == "testprov/alpha-2"
    assert result.decision.constraints_satisfied is False  # constraints were not relaxed


def test_unknown_predicted_output_makes_cost_unknown(two_versions):
    result = route(
        two_versions, {"testprov/alpha-1": (0.9, 100, None), "testprov/alpha-2": (0.9, 100, 10)}
    )
    assert result.model_id == "testprov/alpha-2"
    weak = next(a for a in result.decision.assessments if a.model_id == "testprov/alpha-1")
    assert weak.violations == (Violation.COST_UNKNOWN,)


def test_unknown_price_component_makes_cost_unknown(make_candidate, registry_data):
    # Pricing required by the filter only covers uncached input/output; a tier gap shows up here.
    registry = ModelRegistry.from_mapping(
        registry_data(
            make_candidate(
                pricing={
                    "tiers": [
                        {"name": "long", "applies_when_prompt_tokens_over": 1_000, "input": 1}
                    ]
                }
            )
        )
    )
    result = route(
        registry,
        {"testprov/alpha-1": (0.9, 100, 10)},
        request=CandidateRequest(input_tokens=1_000, max_output_tokens=10, require_pricing=False),
    )
    assert result.status == RouteStatus.SELECTED
    over = route(
        registry,
        {"testprov/alpha-1": (0.9, 100, 10)},
        request=CandidateRequest(input_tokens=1_001, max_output_tokens=10, require_pricing=False),
    )
    assert over.status == RouteStatus.NO_FEASIBLE_CANDIDATE
    assert isinstance(over.costs["testprov/alpha-1"], UnknownCost)


@pytest.mark.parametrize(
    "table",
    [
        {"testprov/alpha-1": (0.9, 100, 10)},  # missing a candidate
        {"testprov/alpha-1": (0.9, 1, 1), "testprov/alpha-2": (0.9, 1, 1), "x/y": (0.9, 1, 1)},
    ],
)
def test_predictor_contract_is_enforced(two_versions, table):
    class Partial(TablePredictor):
        def predict(self, query_features, candidates):
            return [CandidatePrediction(m, *v) for m, v in self.table.items()]

    router = Router(two_versions, Partial(table), PolicyConfig(), clock=lambda: TODAY)
    with pytest.raises(PredictionContractError):
        router.route({}, REQUEST)


def test_prediction_values_are_validated():
    with pytest.raises(ValueError, match="quality must be in"):
        CandidatePrediction("x/y", quality=1.5, latency_p95_ms=1.0, output_tokens=1)


def test_ties_break_deterministically_by_model_id():
    scored = [
        ScoredCandidate("p/b", 0.9, Decimal("1"), 100.0),
        ScoredCandidate("p/a", 0.9, Decimal("1"), 100.0),
    ]
    assert decide(scored, PolicyConfig()).selected_model_id == "p/a"
    assert decide(list(reversed(scored)), PolicyConfig()).selected_model_id == "p/a"


def test_policy_from_shipped_config(repo_root):
    config = yaml.safe_load((repo_root / "configs" / "routing_policy.yaml").read_text("utf-8"))
    policy = PolicyConfig.from_mapping(config["decision"])
    assert policy.fallback == FallbackMode.NO_FEASIBLE_CANDIDATE
    assert policy.min_quality is None and policy.utility_weights is None


def test_utility_fallback_requires_weights():
    with pytest.raises(ValueError, match="requires utility_weights"):
        PolicyConfig(fallback=FallbackMode.MAXIMIZE_UTILITY)
    with pytest.raises(ValueError, match="unsupported objective"):
        PolicyConfig.from_mapping({"objective": "max_quality"})
