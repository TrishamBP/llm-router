from __future__ import annotations

from datetime import date

import pytest
import yaml

from llm_router.models.registry import Capability, LifecycleStatus, ModelRegistry
from llm_router.routing.candidate_filter import (
    CandidateRequest,
    ExclusionReason,
    filter_candidates,
)

TODAY = date(2026, 10, 10)


@pytest.fixture
def registry_of(registry_data):
    def build(*records) -> ModelRegistry:
        return ModelRegistry.from_mapping(registry_data(*records))

    return build


def request(**overrides) -> CandidateRequest:
    values = {"input_tokens": 1_000, "max_output_tokens": 1_000, **overrides}
    return CandidateRequest(**values)


def reasons(result, model_id):
    return next(e.reasons for e in result.excluded if e.model_id == model_id)


def test_eligible_candidate_passes(make_candidate, registry_of):
    result = filter_candidates(registry_of(make_candidate()), request(), today=TODAY)
    assert [c.model_id for c in result.eligible] == ["testprov/alpha-1"]
    assert result.excluded == ()


def test_filters_by_capabilities_without_inferring_support(make_candidate, registry_of):
    registry = registry_of(
        make_candidate("testprov/yes"),
        make_candidate("testprov/no", capabilities={"supports_tools": False}),
        make_candidate("testprov/unknown", capabilities={"supports_tools": None}),
    )
    result = filter_candidates(
        registry, request(required_capabilities=frozenset({Capability.TOOLS})), today=TODAY
    )
    assert [c.model_id for c in result.eligible] == ["testprov/yes"]
    assert reasons(result, "testprov/no") == (ExclusionReason.CAPABILITY_UNSUPPORTED,)
    assert reasons(result, "testprov/unknown") == (ExclusionReason.CAPABILITY_UNKNOWN,)


def test_filters_by_context_window_including_output(make_candidate, registry_of):
    registry = registry_of(
        make_candidate(capabilities={"context_window_tokens": 10_000, "max_output_tokens": 2_000})
    )
    fits = filter_candidates(registry, request(input_tokens=9_000), today=TODAY)
    too_big = filter_candidates(registry, request(input_tokens=9_001), today=TODAY)
    assert fits.has_eligible
    assert reasons(too_big, "testprov/alpha-1") == (ExclusionReason.CONTEXT_WINDOW_TOO_SMALL,)


def test_filters_by_output_limit(make_candidate, registry_of):
    registry = registry_of(make_candidate(capabilities={"max_output_tokens": 4_000}))
    result = filter_candidates(registry, request(max_output_tokens=4_001), today=TODAY)
    assert reasons(result, "testprov/alpha-1") == (ExclusionReason.OUTPUT_LIMIT_TOO_SMALL,)


def test_unknown_limits_exclude(make_candidate, registry_of):
    registry = registry_of(
        make_candidate(capabilities={"context_window_tokens": None, "max_output_tokens": None})
    )
    result = filter_candidates(registry, request(), today=TODAY)
    assert reasons(result, "testprov/alpha-1") == (
        ExclusionReason.CONTEXT_WINDOW_UNKNOWN,
        ExclusionReason.OUTPUT_LIMIT_UNKNOWN,
    )


def test_filters_by_lifecycle_and_enablement(make_candidate, registry_of):
    registry = registry_of(
        make_candidate("testprov/active"),
        make_candidate("testprov/legacy", lifecycle={"status": "legacy"}),
        make_candidate("testprov/deprecated", lifecycle={"status": "deprecated"}),
        make_candidate("testprov/disabled", lifecycle={"enabled": False}),
        make_candidate("testprov/retired", lifecycle={"status": "retired", "enabled": False}),
        make_candidate("testprov/unverified", lifecycle={"status": "unverified", "enabled": False}),
    )
    default = filter_candidates(registry, request(), today=TODAY)
    assert [c.model_id for c in default.eligible] == ["testprov/active", "testprov/legacy"]
    assert reasons(default, "testprov/deprecated") == (ExclusionReason.STATUS_NOT_ALLOWED,)
    assert reasons(default, "testprov/disabled") == (ExclusionReason.NOT_ENABLED,)
    assert reasons(default, "testprov/retired") == (
        ExclusionReason.NOT_ENABLED,
        ExclusionReason.STATUS_NOT_ALLOWED,
    )

    with_deprecated = filter_candidates(
        registry,
        request(
            allowed_statuses=frozenset(
                {LifecycleStatus.ACTIVE, LifecycleStatus.LEGACY, LifecycleStatus.DEPRECATED}
            )
        ),
        today=TODAY,
    )
    assert "testprov/deprecated" in [c.model_id for c in with_deprecated.eligible]


@pytest.mark.parametrize("status", ["retired", "unavailable", "unverified"])
def test_non_routable_statuses_cannot_be_allowed(status):
    with pytest.raises(ValueError, match="can never be routed"):
        request(allowed_statuses=frozenset({LifecycleStatus(status)}))


def test_filters_by_effective_window(make_candidate, registry_of):
    registry = registry_of(
        make_candidate("testprov/future", lifecycle={"effective_from": date(2026, 11, 1)}),
        make_candidate("testprov/past", lifecycle={"effective_until": date(2026, 10, 9)}),
        make_candidate("testprov/today", lifecycle={"effective_until": TODAY}),
    )
    result = filter_candidates(registry, request(), today=TODAY)
    assert [c.model_id for c in result.eligible] == ["testprov/today"]
    assert reasons(result, "testprov/future") == (ExclusionReason.NOT_YET_EFFECTIVE,)
    assert reasons(result, "testprov/past") == (ExclusionReason.NO_LONGER_EFFECTIVE,)


def test_filters_by_provider_and_region(make_candidate, registry_of):
    registry = registry_of(
        make_candidate("testprov/a", provider_metadata={"region": "eu-west-1"}),
        make_candidate("otherprov/b", provider_metadata={"region": "us-east-1"}),
        make_candidate("testprov/c"),
    )
    by_provider = filter_candidates(
        registry, request(allowed_providers=frozenset({"testprov"})), today=TODAY
    )
    assert reasons(by_provider, "otherprov/b") == (ExclusionReason.PROVIDER_NOT_ALLOWED,)

    by_region = filter_candidates(
        registry, request(allowed_regions=frozenset({"eu-west-1"})), today=TODAY
    )
    assert [c.model_id for c in by_region.eligible] == ["testprov/a"]
    assert reasons(by_region, "otherprov/b") == (ExclusionReason.REGION_NOT_ALLOWED,)
    assert reasons(by_region, "testprov/c") == (ExclusionReason.REGION_UNKNOWN,)


def test_allowlist_and_denylist(make_candidate, registry_of):
    registry = registry_of(make_candidate("testprov/a"), make_candidate("testprov/b"))
    result = filter_candidates(
        registry,
        request(
            allowlist=frozenset({"testprov/a", "testprov/b"}), denylist=frozenset({"testprov/b"})
        ),
        today=TODAY,
    )
    assert [c.model_id for c in result.eligible] == ["testprov/a"]
    assert reasons(result, "testprov/b") == (ExclusionReason.DENYLISTED,)
    only_b = filter_candidates(registry, request(allowlist=frozenset({"testprov/b"})), today=TODAY)
    assert reasons(only_b, "testprov/a") == (ExclusionReason.NOT_ALLOWLISTED,)


def test_missing_pricing_excludes_only_when_required(make_candidate, registry_of):
    registry = registry_of(make_candidate(pricing={"output": None}))
    required = filter_candidates(registry, request(), today=TODAY)
    assert reasons(required, "testprov/alpha-1") == (ExclusionReason.PRICING_UNKNOWN,)
    optional = filter_candidates(registry, request(require_pricing=False), today=TODAY)
    assert optional.has_eligible


def test_filters_by_supported_parameters(make_candidate, registry_of):
    registry = registry_of(
        make_candidate("testprov/a", capabilities={"parameters": {"seed": True}}),
        make_candidate("testprov/b", capabilities={"parameters": {"seed": False}}),
        make_candidate("testprov/c", capabilities={"parameters": {}}),
    )
    result = filter_candidates(
        registry, request(required_parameters=frozenset({"seed"})), today=TODAY
    )
    assert [c.model_id for c in result.eligible] == ["testprov/a"]
    assert reasons(result, "testprov/b") == (ExclusionReason.PARAMETER_UNSUPPORTED,)
    assert reasons(result, "testprov/c") == (ExclusionReason.PARAMETER_UNKNOWN,)


def test_no_eligible_candidates_explains_every_exclusion(make_candidate, registry_of):
    registry = registry_of(
        make_candidate("testprov/a", lifecycle={"enabled": False}, pricing={"input": None}),
        make_candidate("testprov/b", capabilities={"max_output_tokens": 10}),
    )
    result = filter_candidates(registry, request(), today=TODAY)
    assert not result.has_eligible and result.eligible == ()
    assert reasons(result, "testprov/a") == (
        ExclusionReason.NOT_ENABLED,
        ExclusionReason.PRICING_UNKNOWN,
    )
    explanation = result.explain()
    assert "0 eligible" in explanation
    assert "testprov/a" in explanation and "testprov/b" in explanation


def test_request_from_shipped_routing_policy(repo_root):
    config = yaml.safe_load((repo_root / "configs" / "routing_policy.yaml").read_text("utf-8"))
    req = CandidateRequest.from_config(
        config["candidate_filter"], input_tokens=1_000, max_output_tokens=1_000
    )
    assert req.allowed_statuses == {LifecycleStatus.ACTIVE, LifecycleStatus.LEGACY}
    assert req.require_pricing and req.allowlist is None and req.denylist == frozenset()


def test_shipped_registry_is_conservative_about_unknown_capabilities(repo_root):
    registry = ModelRegistry.from_yaml(repo_root / "configs" / "model_registry.yaml")
    plain = filter_candidates(registry, request(), today=TODAY)
    assert {c.provider for c in plain.eligible} == {"anthropic"}  # Bedrock is not enabled
    assert len(plain.eligible) == 8

    tools = filter_candidates(
        registry, request(required_capabilities=frozenset({Capability.TOOLS})), today=TODAY
    )
    # Tool support is only recorded where the provider documentation states it.
    assert [c.model_id for c in tools.eligible] == [
        "anthropic/claude-haiku-5-5",
        "anthropic/claude-opus-5-5",
        "anthropic/claude-sonnet-5-5",
    ]

    long_prompt = filter_candidates(registry, request(input_tokens=500_000), today=TODAY)
    assert "anthropic/claude-haiku-4-5" not in [c.model_id for c in long_prompt.eligible]
