from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = {"url": "https://example.test/provider-docs", "checked_at": date(2026, 10, 1)}

CandidateFactory = Callable[..., dict[str, Any]]


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def make_candidate() -> CandidateFactory:
    """Build a valid, enabled, fully described candidate record; overrides are deep-merged."""

    def factory(model_id: str = "testprov/alpha-1", **overrides: Any) -> dict[str, Any]:
        provider, name = model_id.split("/", 1)
        base: dict[str, Any] = {
            "model_id": model_id,
            "display_name": name,
            "provider": provider,
            "family": "alpha",
            "version": "1",
            "lifecycle": {
                "status": "active",
                "enabled": True,
                "verified_at": date(2026, 10, 1),
                "sources": [SOURCE],
            },
            "capabilities": {
                "context_window_tokens": 200_000,
                "max_output_tokens": 64_000,
                "supports_tools": True,
                "supports_vision": True,
                "supports_structured_output": True,
                "parameters": {"temperature": True},
                "verified_at": date(2026, 10, 1),
                "sources": [SOURCE],
            },
            "pricing": {
                "input": 2.0,
                "output": 10.0,
                "cache_read": 0.2,
                "cache_write_5m": 2.5,
                "cache_write_1h": 4.0,
                "verified_at": date(2026, 10, 1),
                "sources": [SOURCE],
            },
            "provider_metadata": {"api_model_id": f"{name}-api", "sources": [SOURCE]},
        }
        return _deep_merge(base, overrides)

    return factory


@pytest.fixture
def registry_data() -> Callable[..., dict[str, Any]]:
    def build(
        *models: dict[str, Any], pending: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"schema_version": 1, "models": list(models)}
        if pending is not None:
            data["pending_candidates"] = pending
        return data

    return build
