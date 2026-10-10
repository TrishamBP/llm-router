from __future__ import annotations

from decimal import Decimal

import pytest

from llm_router.evaluation.pricing import (
    BilledCost,
    CostComponent,
    CostEstimate,
    TokenUsage,
    UnknownCost,
    estimate_candidate_cost,
    select_tier,
)
from llm_router.models.registry import ModelCandidate, ModelRegistry, RegistryValidationError

TIER = {
    "name": "long",
    "applies_when_prompt_tokens_over": 100_000,
    "input": 4.0,
    "output": 20.0,
    "cache_read": 0.4,
    "cache_write_5m": 5.0,
    "cache_write_1h": 8.0,
}


@pytest.fixture
def candidate(make_candidate):
    def build(**overrides) -> ModelCandidate:
        return ModelCandidate.model_validate(make_candidate(**overrides))

    return build


def test_input_and_output_cost(candidate):
    cost = estimate_candidate_cost(candidate(), TokenUsage(input_tokens=1_000, output_tokens=500))
    assert isinstance(cost, CostEstimate)
    # 1000/1e6 * $2 + 500/1e6 * $10
    assert cost.components == {
        CostComponent.INPUT: Decimal("0.002"),
        CostComponent.OUTPUT: Decimal("0.005"),
    }
    assert cost.total == Decimal("0.007")
    assert cost.tier is None and cost.currency == "USD" and cost.kind == "estimate"


def test_cache_reads_and_writes_are_priced_once(candidate):
    usage = TokenUsage(
        input_tokens=1_000,  # uncached only
        output_tokens=0,
        cache_read_tokens=10_000,
        cache_write_5m_tokens=2_000,
        cache_write_1h_tokens=1_000,
    )
    cost = estimate_candidate_cost(candidate(), usage)
    assert isinstance(cost, CostEstimate)
    assert cost.components[CostComponent.INPUT] == Decimal("0.002")  # 1k * $2
    assert cost.components[CostComponent.CACHE_READ] == Decimal("0.002")  # 10k * $0.2
    assert cost.components[CostComponent.CACHE_WRITE_5M] == Decimal("0.005")  # 2k * $2.5
    assert cost.components[CostComponent.CACHE_WRITE_1H] == Decimal("0.004")  # 1k * $4
    assert cost.total == Decimal("0.013")
    assert usage.prompt_tokens == 14_000


def test_unused_component_does_not_need_a_price(candidate):
    model = candidate(pricing={"cache_read": None, "cache_write_5m": None, "cache_write_1h": None})
    cost = estimate_candidate_cost(model, TokenUsage(input_tokens=10, output_tokens=10))
    assert isinstance(cost, CostEstimate)


def test_missing_price_gives_unknown_cost_not_zero(candidate):
    model = candidate(pricing={"output": None, "cache_read": None})
    usage = TokenUsage(input_tokens=10, output_tokens=10, cache_read_tokens=5)
    cost = estimate_candidate_cost(model, usage)
    assert isinstance(cost, UnknownCost)
    assert cost.missing == (CostComponent.OUTPUT, CostComponent.CACHE_READ)
    assert "output" in cost.reason and cost.kind == "unknown"


def test_completely_unpriced_candidate_is_unknown(make_candidate):
    record = make_candidate()
    record["pricing"] = {}
    cost = estimate_candidate_cost(
        ModelCandidate.model_validate(record), TokenUsage(input_tokens=1, output_tokens=1)
    )
    assert isinstance(cost, UnknownCost)


@pytest.mark.parametrize(
    ("prompt_tokens", "expected_tier"),
    [(99_999, None), (100_000, None), (100_001, "long")],
)
def test_tier_threshold_is_exclusive(candidate, prompt_tokens, expected_tier):
    model = candidate(pricing={"tiers": [TIER]})
    tier = select_tier(model.pricing, prompt_tokens)
    assert (tier.name if tier else None) == expected_tier


def test_long_context_tier_reprices_the_entire_request(candidate):
    model = candidate(pricing={"tiers": [TIER]})
    usage = TokenUsage(input_tokens=100_001, output_tokens=1_000)
    cost = estimate_candidate_cost(model, usage)
    assert isinstance(cost, CostEstimate)
    assert cost.tier == "long"
    assert cost.total == Decimal("100001") / 1_000_000 * 4 + Decimal("0.02")


def test_cache_tokens_count_towards_tier_threshold(candidate):
    model = candidate(pricing={"tiers": [TIER]})
    usage = TokenUsage(input_tokens=1_000, output_tokens=0, cache_read_tokens=99_500)
    cost = estimate_candidate_cost(model, usage)
    assert isinstance(cost, CostEstimate)
    assert cost.tier == "long"
    assert cost.components[CostComponent.CACHE_READ] == Decimal("99500") / 1_000_000 * Decimal(
        "0.4"
    )


def test_highest_applicable_tier_is_selected(candidate):
    tiers = [TIER, {**TIER, "name": "longer", "applies_when_prompt_tokens_over": 500_000}]
    model = candidate(pricing={"tiers": tiers})
    assert select_tier(model.pricing, 300_000).name == "long"
    assert select_tier(model.pricing, 600_000).name == "longer"


def test_unknown_tier_price_does_not_fall_back_to_base(candidate):
    model = candidate(pricing={"tiers": [{**TIER, "output": None}]})
    cost = estimate_candidate_cost(model, TokenUsage(input_tokens=200_000, output_tokens=1))
    assert isinstance(cost, UnknownCost)
    assert cost.tier == "long" and cost.missing == (CostComponent.OUTPUT,)


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_token_counts_must_be_non_negative_integers(bad):
    with pytest.raises(ValueError, match="non-negative integer"):
        TokenUsage(input_tokens=bad, output_tokens=0)


def test_negative_prices_are_rejected(make_candidate, registry_data):
    with pytest.raises(RegistryValidationError, match="greater than or equal to 0"):
        ModelRegistry.from_mapping(registry_data(make_candidate(pricing={"output": -10})))


def test_billed_cost_is_a_separate_type():
    billed = BilledCost(
        model_id="testprov/alpha-1", amount=Decimal("0.01"), currency="USD", source="usage-report"
    )
    assert billed.kind == "billed"
    assert not isinstance(billed, CostEstimate)
    with pytest.raises(ValueError, match="cannot be negative"):
        BilledCost(model_id="x/y", amount=Decimal("-1"), currency="USD", source="usage-report")


def test_shipped_haiku_5_5_long_context_pricing(repo_root):
    registry = ModelRegistry.from_yaml(repo_root / "configs" / "model_registry.yaml")
    haiku = registry.get("anthropic/claude-haiku-5-5")
    short = estimate_candidate_cost(haiku, TokenUsage(input_tokens=100_000, output_tokens=1_000))
    long = estimate_candidate_cost(haiku, TokenUsage(input_tokens=200_000, output_tokens=1_000))
    assert isinstance(short, CostEstimate) and isinstance(long, CostEstimate)
    assert short.total == Decimal("0.0105")  # 0.1 * $0.10 + 0.001 * $0.50
    assert long.total == Decimal("0.1025")  # 0.2 * $0.50 + 0.001 * $2.50
    assert long.tier == "prompt_over_100k"
