from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from llm_router.models.registry import (
    LifecycleStatus,
    ModelRegistry,
    RegistryError,
    RegistryValidationError,
    UnknownModelError,
)


def test_loads_valid_registry(make_candidate, registry_data):
    registry = ModelRegistry.from_mapping(
        registry_data(make_candidate("testprov/alpha-1"), make_candidate("testprov/alpha-2"))
    )
    assert len(registry) == 2
    assert registry.get("testprov/alpha-2").provider_metadata.api_model_id == "alpha-2-api"
    assert "testprov/alpha-1" in registry


def test_candidates_are_deterministically_ordered(make_candidate, registry_data):
    ids = ["testprov/zeta", "testprov/alpha", "otherprov/mid"]
    forward = ModelRegistry.from_mapping(registry_data(*(make_candidate(i) for i in ids)))
    backward = ModelRegistry.from_mapping(
        registry_data(*(make_candidate(i) for i in reversed(ids)))
    )
    assert [c.model_id for c in forward] == sorted(ids)
    assert [c.model_id for c in forward] == [c.model_id for c in backward]


def test_select_by_provider_family_version_status(make_candidate, registry_data):
    registry = ModelRegistry.from_mapping(
        registry_data(
            make_candidate("testprov/alpha-1", version="1"),
            make_candidate("testprov/alpha-2", version="2"),
            make_candidate("testprov/beta-1", family="beta"),
            make_candidate("otherprov/alpha-1"),
        )
    )
    assert [c.model_id for c in registry.select(family="alpha", provider="testprov")] == [
        "testprov/alpha-1",
        "testprov/alpha-2",
    ]
    assert [c.model_id for c in registry.select(version="2")] == ["testprov/alpha-2"]
    assert len(registry.select(status=LifecycleStatus.ACTIVE)) == 4


def test_get_unknown_model_raises(make_candidate, registry_data):
    registry = ModelRegistry.from_mapping(registry_data(make_candidate()))
    with pytest.raises(UnknownModelError, match="testprov/missing"):
        registry.get("testprov/missing")


def test_minimal_record_has_unknown_metadata_and_is_disabled(registry_data):
    record = {
        "model_id": "testprov/minimal",
        "display_name": "Minimal",
        "provider": "testprov",
        "family": "alpha",
        "version": "0",
    }
    registry = ModelRegistry.from_mapping(registry_data(record))
    candidate = registry.get("testprov/minimal")
    assert candidate.lifecycle.status == LifecycleStatus.UNVERIFIED
    assert candidate.lifecycle.enabled is False
    assert candidate.capabilities.supports_tools is None
    assert candidate.capabilities.context_window_tokens is None
    assert candidate.pricing.input is None  # unknown, not zero
    assert registry.routable() == ()


@pytest.mark.parametrize("status", ["retired", "unavailable", "unverified"])
def test_non_routable_status_cannot_be_enabled(make_candidate, registry_data, status):
    record = make_candidate(lifecycle={"status": status, "enabled": True})
    with pytest.raises(RegistryValidationError, match="cannot be enabled"):
        ModelRegistry.from_mapping(registry_data(record))


@pytest.mark.parametrize("status", ["retired", "unavailable"])
def test_retired_and_unavailable_are_not_routable(make_candidate, registry_data, status):
    registry = ModelRegistry.from_mapping(
        registry_data(
            make_candidate("testprov/gone", lifecycle={"status": status, "enabled": False}),
            make_candidate("testprov/live"),
        )
    )
    assert [c.model_id for c in registry.routable()] == ["testprov/live"]


def test_rejects_duplicate_model_ids(make_candidate, registry_data):
    with pytest.raises(RegistryValidationError, match="duplicate model_id 'testprov/alpha-1'"):
        ModelRegistry.from_mapping(registry_data(make_candidate(), make_candidate()))


def test_rejects_duplicate_provider_model_ids(make_candidate, registry_data):
    a = make_candidate("testprov/a", provider_metadata={"api_model_id": "same"})
    b = make_candidate("testprov/b", provider_metadata={"api_model_id": "same"})
    with pytest.raises(RegistryValidationError, match="share provider"):
        ModelRegistry.from_mapping(registry_data(a, b))


def test_same_api_id_on_different_providers_is_allowed(make_candidate, registry_data):
    a = make_candidate("testprov/a", provider_metadata={"api_model_id": "same"})
    b = make_candidate("otherprov/a", provider_metadata={"api_model_id": "same"})
    assert len(ModelRegistry.from_mapping(registry_data(a, b))) == 2


@pytest.mark.parametrize("price", [-0.01, True, "free"])
def test_rejects_invalid_prices(make_candidate, registry_data, price):
    with pytest.raises(RegistryValidationError, match=r"models\[0\] \(testprov/alpha-1\)\.pricing"):
        ModelRegistry.from_mapping(registry_data(make_candidate(pricing={"input": price})))


def test_prices_are_exact_decimals(make_candidate, registry_data):
    registry = ModelRegistry.from_mapping(registry_data(make_candidate(pricing={"input": 0.1})))
    assert registry.get("testprov/alpha-1").pricing.input == Decimal("0.1")


def test_prices_require_provenance(make_candidate, registry_data):
    record = make_candidate(pricing={"sources": [], "verified_at": None})
    with pytest.raises(RegistryValidationError, match="prices require verified_at"):
        ModelRegistry.from_mapping(registry_data(record))


@pytest.mark.parametrize(
    ("tiers", "message"),
    [
        (
            [
                {"name": "b", "applies_when_prompt_tokens_over": 200_000, "input": 1},
                {"name": "a", "applies_when_prompt_tokens_over": 100_000, "input": 1},
            ],
            "increasing order",
        ),
        (
            [
                {"name": "a", "applies_when_prompt_tokens_over": 100_000},
                {"name": "a", "applies_when_prompt_tokens_over": 200_000},
            ],
            "names must be unique",
        ),
        ([{"name": "a", "applies_when_prompt_tokens_over": 0}], "greater than 0"),
        (
            [{"name": "a", "applies_when_prompt_tokens_over": 10, "scope": "marginal"}],
            "entire_request",
        ),
    ],
)
def test_rejects_malformed_pricing_tiers(make_candidate, registry_data, tiers, message):
    with pytest.raises(RegistryValidationError, match=message):
        ModelRegistry.from_mapping(registry_data(make_candidate(pricing={"tiers": tiers})))


def test_rejects_contradictory_lifecycle_dates(make_candidate, registry_data):
    record = make_candidate(
        lifecycle={"effective_from": date(2026, 12, 1), "effective_until": date(2026, 1, 1)}
    )
    with pytest.raises(RegistryValidationError, match="effective_from must not be after"):
        ModelRegistry.from_mapping(registry_data(record))


def test_verified_status_requires_provenance(make_candidate, registry_data):
    record = make_candidate(lifecycle={"verified_at": None})
    with pytest.raises(RegistryValidationError, match="requires verified_at"):
        ModelRegistry.from_mapping(registry_data(record))


def test_enabled_candidate_requires_provider_model_id(make_candidate, registry_data):
    record = make_candidate(provider_metadata={"api_model_id": None})
    with pytest.raises(RegistryValidationError, match="requires provider_metadata.api_model_id"):
        ModelRegistry.from_mapping(registry_data(record))


def test_model_id_must_carry_provider_prefix(make_candidate, registry_data):
    record = make_candidate(provider="otherprov")
    with pytest.raises(RegistryValidationError, match="provider prefix"):
        ModelRegistry.from_mapping(registry_data(record))


def test_rejects_unknown_fields_and_reports_all_issues(make_candidate, registry_data):
    bad = make_candidate(pricing={"input": -1}, colour="blue")
    with pytest.raises(RegistryValidationError) as excinfo:
        ModelRegistry.from_mapping(registry_data(bad))
    assert len(excinfo.value.issues) == 2
    assert any("colour" in issue for issue in excinfo.value.issues)


def test_rejects_output_limit_above_context_window(make_candidate, registry_data):
    record = make_candidate(capabilities={"context_window_tokens": 1000, "max_output_tokens": 2000})
    with pytest.raises(RegistryValidationError, match="cannot exceed context_window_tokens"):
        ModelRegistry.from_mapping(registry_data(record))


def test_benchmark_metrics_require_a_measured_run(make_candidate, registry_data):
    with pytest.raises(RegistryValidationError, match="not_evaluated benchmark cannot carry"):
        ModelRegistry.from_mapping(registry_data(make_candidate(benchmark={"quality_score": 0.9})))
    with pytest.raises(RegistryValidationError, match="requires measured_at"):
        ModelRegistry.from_mapping(
            registry_data(make_candidate(benchmark={"status": "evaluated", "quality_score": 0.9}))
        )


def test_rejects_unsupported_schema_version(make_candidate):
    with pytest.raises(RegistryValidationError, match="schema_version"):
        ModelRegistry.from_mapping({"schema_version": 2, "models": [make_candidate()]})


def test_rejects_malformed_yaml(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text("schema_version: 1\nmodels: [unclosed\n", encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="invalid YAML"):
        ModelRegistry.from_yaml(path)


def test_rejects_non_mapping_yaml(tmp_path):
    path = tmp_path / "registry.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(RegistryValidationError, match="top level must be a mapping"):
        ModelRegistry.from_yaml(path)


def test_missing_file_raises_registry_error(tmp_path):
    with pytest.raises(RegistryError, match="cannot read"):
        ModelRegistry.from_yaml(tmp_path / "absent.yaml")


# --- the shipped configs/model_registry.yaml -------------------------------------------------


@pytest.fixture
def shipped(repo_root):
    return ModelRegistry.from_yaml(repo_root / "configs" / "model_registry.yaml")


def test_shipped_registry_is_valid_and_versions_are_separate(shipped):
    opus = shipped.select(provider="anthropic", family="opus")
    assert [c.version for c in opus] == ["4.7", "4.8", "5", "5.5"]
    assert len({c.provider_metadata.api_model_id for c in opus}) == 4


def test_shipped_registry_never_enables_unverified_candidates(shipped):
    for candidate in shipped:
        if candidate.lifecycle.status == LifecycleStatus.UNVERIFIED:
            assert not candidate.lifecycle.enabled, candidate.model_id
        if candidate.is_routable:
            assert candidate.lifecycle.sources and candidate.provider_metadata.sources
            assert candidate.pricing.input is not None and candidate.pricing.output is not None


def test_shipped_registry_keeps_bedrock_disabled(shipped):
    bedrock = shipped.select(provider="amazon-bedrock")
    assert bedrock and all(not c.lifecycle.enabled for c in bedrock)
    assert all(c.pricing.input is None for c in bedrock)


def test_shipped_registry_keeps_opus_4_9_pending_only(shipped):
    assert not shipped.select(family="opus", version="4.9")
    assert [p.requested_label for p in shipped.pending] == ["Claude Opus 4.9"]
