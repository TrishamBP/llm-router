from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from llm_router.data.model_catalog import (
    AnthropicModelsAdapter,
    CatalogEntry,
    CatalogParseError,
    CatalogSource,
    SourceKind,
    merge_catalog,
)
from llm_router.models.registry import LifecycleStatus, ModelRegistry

API_SOURCE = CatalogSource(
    SourceKind.PROVIDER_API, "https://api.anthropic.com/v1/models", date(2026, 10, 10)
)
MANUAL_SOURCE = CatalogSource(SourceKind.MANUAL, "notes/manual-entry.md", date(2026, 10, 10))

# Shape of a Models API list response (fields per Anthropic's Models API documentation).
MODELS_API_FIXTURE = {
    "data": [
        {
            "type": "model",
            "id": "known-api",
            "display_name": "Known",
            "created_at": "2026-01-01T00:00:00Z",
            "max_input_tokens": 200_000,
            "max_tokens": 32_000,
            "line": "alpha",
            "capabilities": {"thinking": {"types": {"disabled": {"supported": True}}}},
        },
        {
            "type": "model",
            "id": "brand-new-model",
            "display_name": "Brand New",
            "created_at": "2026-10-01T00:00:00Z",
            "max_input_tokens": 1_000_000,
            "max_tokens": 128_000,
            "line": "omega",
        },
    ],
    "has_more": False,
}


@pytest.fixture
def registry(make_candidate, registry_data):
    return ModelRegistry.from_mapping(
        registry_data(
            make_candidate(
                "anthropic/known",
                provider_metadata={"api_model_id": "known-api"},
                capabilities={"context_window_tokens": None, "max_output_tokens": 64_000},
                benchmark={
                    "status": "evaluated",
                    "quality_score": 0.8,
                    "benchmark_dataset_version": "v1",
                    "evaluation_run_id": "run-1",
                    "measured_at": date(2026, 9, 1),
                },
            ),
            make_candidate(
                "anthropic/unlisted", provider_metadata={"api_model_id": "unlisted-api"}
            ),
            make_candidate("otherprov/elsewhere"),
        )
    )


def test_anthropic_adapter_parses_models_api_payload():
    entries = AnthropicModelsAdapter().parse(MODELS_API_FIXTURE, API_SOURCE)
    assert [e.api_model_id for e in entries] == ["known-api", "brand-new-model"]
    known = entries[0]
    assert known.provider == "anthropic"
    assert known.context_window_tokens == 200_000
    assert known.max_output_tokens == 32_000
    assert known.line == "alpha"
    assert known.raw["capabilities"]["thinking"]["types"]["disabled"]["supported"] is True


@pytest.mark.parametrize(
    "payload",
    [{}, {"data": [{"display_name": "no id"}]}, {"data": [{"id": "x", "max_tokens": -5}]}],
)
def test_anthropic_adapter_rejects_malformed_payloads(payload):
    with pytest.raises(CatalogParseError):
        AnthropicModelsAdapter().parse(payload, API_SOURCE)


def test_merge_adds_new_models_as_unverified_and_disabled(registry):
    entries = AnthropicModelsAdapter().parse(MODELS_API_FIXTURE, API_SOURCE)
    report = merge_catalog(registry, entries, provider="anthropic")
    assert report.added == ("anthropic/brand-new-model",)
    new = report.registry.get("anthropic/brand-new-model")
    assert new.lifecycle.status == LifecycleStatus.UNVERIFIED
    assert new.lifecycle.enabled is False and not new.is_routable
    assert new.family == "omega" and new.version == "unknown"
    assert new.pricing.input is None  # discovery never invents prices
    assert new.capabilities.context_window_tokens == 1_000_000


def test_merge_fills_only_unknown_fields_and_reports_conflicts(registry):
    entries = AnthropicModelsAdapter().parse(MODELS_API_FIXTURE, API_SOURCE)
    report = merge_catalog(registry, entries, provider="anthropic")
    known = report.registry.get("anthropic/known")
    assert known.capabilities.context_window_tokens == 200_000  # was unknown: filled
    assert known.capabilities.max_output_tokens == 64_000  # curated: kept
    assert [(c.field, c.registry_value, c.catalog_value) for c in report.conflicts] == [
        ("max_output_tokens", 64_000, 32_000)
    ]
    assert [c.field for c in report.filled] == ["context_window_tokens"]
    assert known.capabilities.sources[-1].note.startswith("provider_api")


def test_merge_never_touches_pricing_lifecycle_or_benchmarks(registry):
    before = registry.get("anthropic/known")
    entries = AnthropicModelsAdapter().parse(MODELS_API_FIXTURE, API_SOURCE)
    after = merge_catalog(registry, entries, provider="anthropic").registry.get("anthropic/known")
    assert after.pricing == before.pricing and after.pricing.input == Decimal("2.0")
    assert after.lifecycle == before.lifecycle
    assert after.benchmark == before.benchmark


def test_manual_entries_do_not_fill_fields(registry):
    entry = CatalogEntry(
        provider="anthropic",
        api_model_id="known-api",
        source=MANUAL_SOURCE,
        context_window_tokens=1,
    )
    report = merge_catalog(registry, [entry], provider="anthropic")
    assert report.filled == ()
    assert report.registry.get("anthropic/known").capabilities.context_window_tokens is None


def test_merge_reports_registry_models_missing_from_catalog(registry):
    entries = AnthropicModelsAdapter().parse(MODELS_API_FIXTURE, API_SOURCE)
    report = merge_catalog(registry, entries, provider="anthropic")
    assert report.missing_from_catalog == ("anthropic/unlisted",)
    # Absence is reported for review; the candidate itself is unchanged.
    assert report.registry.get("anthropic/unlisted") == registry.get("anthropic/unlisted")


def test_merge_does_not_mutate_the_input_registry(registry):
    entries = AnthropicModelsAdapter().parse(MODELS_API_FIXTURE, API_SOURCE)
    merge_catalog(registry, entries, provider="anthropic")
    assert "anthropic/brand-new-model" not in registry
    assert registry.get("anthropic/known").capabilities.context_window_tokens is None


def test_merge_rejects_entries_for_another_provider(registry):
    entry = CatalogEntry(provider="otherprov", api_model_id="x", source=API_SOURCE)
    with pytest.raises(ValueError, match="otherprov"):
        merge_catalog(registry, [entry], provider="anthropic")


def test_merge_revalidates_filled_values(make_candidate, registry_data):
    registry = ModelRegistry.from_mapping(
        registry_data(
            make_candidate(
                "anthropic/small",
                provider_metadata={"api_model_id": "small-api"},
                capabilities={"context_window_tokens": None, "max_output_tokens": 64_000},
            )
        )
    )
    entry = CatalogEntry(
        provider="anthropic",
        api_model_id="small-api",
        source=API_SOURCE,
        context_window_tokens=1_000,
    )
    with pytest.raises(ValueError, match="cannot exceed context_window_tokens"):
        merge_catalog(registry, [entry], provider="anthropic")
