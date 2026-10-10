"""Normalise provider model catalogues into registry updates.

Provider adapters parse a provider's catalogue payload into provider-neutral `CatalogEntry`
records. `merge_catalog` compares those entries with the curated registry and returns a new
registry plus a report. Rules:

- No network access here: adapters parse payloads that the caller fetched (or fixtures).
- Newly discovered models are added as `unverified` and disabled; they are never activated.
- Pricing, lifecycle status, enablement, and benchmark data are never changed by a merge.
- Only authoritative sources (provider API or provider docs) may fill fields that are unknown
  (`None`) in the registry. Conflicting curated values are kept and reported for review.

Implemented adapters: Anthropic Models API (`GET /v1/models`) response parsing.
Not implemented: Amazon Bedrock and other providers; add a `CatalogAdapter` for each.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any, Protocol

from llm_router.models.registry import (
    Capabilities,
    ModelCandidate,
    ModelRegistry,
    ProviderMetadata,
    Source,
)


class SourceKind(StrEnum):
    PROVIDER_API = "provider_api"
    PROVIDER_DOCS = "provider_docs"
    MANUAL = "manual"

    @property
    def authoritative(self) -> bool:
        return self in (SourceKind.PROVIDER_API, SourceKind.PROVIDER_DOCS)


@dataclass(frozen=True, slots=True)
class CatalogSource:
    kind: SourceKind
    reference: str  # URL or endpoint the payload came from
    retrieved_at: date


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    provider: str  # registry provider name, e.g. "anthropic"
    api_model_id: str
    source: CatalogSource
    display_name: str | None = None
    line: str | None = None  # provider-reported model line/family, if any
    context_window_tokens: int | None = None
    max_output_tokens: int | None = None
    region: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)  # untouched provider record


class CatalogAdapter(Protocol):
    provider: str

    def parse(self, payload: Mapping[str, Any], source: CatalogSource) -> list[CatalogEntry]: ...


class CatalogParseError(ValueError):
    pass


class AnthropicModelsAdapter:
    """Parses an Anthropic Models API list response: `{"data": [{"id": ..., ...}], ...}`.

    Uses the documented fields `id`, `display_name`, `max_input_tokens` (context window),
    `max_tokens` (output cap), and `line`. The `capabilities` object is preserved in `raw` but
    not mapped to registry capabilities until its keys are verified for each capability.
    """

    provider = "anthropic"

    def parse(self, payload: Mapping[str, Any], source: CatalogSource) -> list[CatalogEntry]:
        data = payload.get("data")
        if not isinstance(data, list):
            raise CatalogParseError("expected a 'data' list in the Models API response")
        entries = []
        for index, item in enumerate(data):
            if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
                raise CatalogParseError(f"data[{index}]: missing string 'id'")
            entries.append(
                CatalogEntry(
                    provider=self.provider,
                    api_model_id=item["id"],
                    source=source,
                    display_name=_optional_str(item, "display_name", index),
                    line=_optional_str(item, "line", index),
                    context_window_tokens=_optional_positive_int(item, "max_input_tokens", index),
                    max_output_tokens=_optional_positive_int(item, "max_tokens", index),
                    raw=dict(item),
                )
            )
        return entries


@dataclass(frozen=True, slots=True)
class FieldChange:
    model_id: str
    field: str
    registry_value: object
    catalog_value: object


@dataclass(frozen=True, slots=True)
class MergeReport:
    registry: ModelRegistry
    added: tuple[str, ...]  # new unverified, disabled candidates
    filled: tuple[FieldChange, ...]  # unknown registry fields filled from authoritative data
    conflicts: tuple[FieldChange, ...]  # curated values that differ; registry value kept
    missing_from_catalog: tuple[str, ...]  # registry candidates of this provider not listed


_FILLABLE = ("context_window_tokens", "max_output_tokens")


def merge_catalog(
    registry: ModelRegistry, entries: Iterable[CatalogEntry], *, provider: str
) -> MergeReport:
    entries = list(entries)
    wrong = sorted({e.provider for e in entries if e.provider != provider})
    if wrong:
        raise ValueError(f"entries for providers {wrong} passed to merge for '{provider}'")

    updated: dict[str, ModelCandidate] = {c.model_id: c for c in registry}
    added: list[str] = []
    filled: list[FieldChange] = []
    conflicts: list[FieldChange] = []
    seen: set[str] = set()

    for entry in sorted(entries, key=lambda e: e.api_model_id):
        existing = registry.find_by_provider_id(provider, entry.api_model_id)
        if existing is None:
            candidate = _new_candidate(entry)
            if candidate.model_id in updated:
                raise ValueError(f"cannot add '{candidate.model_id}': model_id already in use")
            updated[candidate.model_id] = candidate
            added.append(candidate.model_id)
            continue

        seen.add(existing.model_id)
        caps = existing.capabilities
        fills: dict[str, int] = {}
        for name in _FILLABLE:
            current = getattr(caps, name)
            incoming = getattr(entry, name)
            if incoming is None:
                continue
            if current is None and entry.source.kind.authoritative:
                fills[name] = incoming
                filled.append(FieldChange(existing.model_id, name, current, incoming))
            elif current is not None and current != incoming:
                conflicts.append(FieldChange(existing.model_id, name, current, incoming))
        if fills:
            # Re-validate rather than model_copy(update=...), which would skip validation.
            caps_data = caps.model_dump()
            caps_data.update(fills)
            caps_data["sources"] = (*caps.sources, _as_source(entry, "filled by catalogue merge"))
            caps_data["verified_at"] = caps.verified_at or entry.source.retrieved_at
            candidate_data = existing.model_dump()
            candidate_data["capabilities"] = Capabilities.model_validate(caps_data)
            updated[existing.model_id] = ModelCandidate.model_validate(candidate_data)

    missing = tuple(
        c.model_id for c in registry.select(provider=provider) if c.model_id not in seen
    )
    new_registry = ModelRegistry(updated.values(), registry.pending, source=registry.source)
    return MergeReport(new_registry, tuple(added), tuple(filled), tuple(conflicts), missing)


def _new_candidate(entry: CatalogEntry) -> ModelCandidate:
    source = _as_source(entry, "discovered by catalogue merge; review before enabling")
    authoritative = entry.source.kind.authoritative
    return ModelCandidate(
        model_id=f"{entry.provider}/{entry.api_model_id}",
        display_name=entry.display_name or entry.api_model_id,
        provider=entry.provider,
        family=entry.line or "unclassified",
        version="unknown",
        # status stays UNVERIFIED and enabled stays False: discovery never activates a model.
        capabilities=Capabilities(
            context_window_tokens=entry.context_window_tokens if authoritative else None,
            max_output_tokens=entry.max_output_tokens if authoritative else None,
            verified_at=entry.source.retrieved_at if authoritative else None,
            sources=(source,) if authoritative else (),
        ),
        provider_metadata=ProviderMetadata(
            api_model_id=entry.api_model_id, region=entry.region, sources=(source,)
        ),
        notes=f"Added from {entry.source.kind} catalogue on {entry.source.retrieved_at}.",
    )


def _as_source(entry: CatalogEntry, note: str) -> Source:
    return Source(
        url=entry.source.reference,
        checked_at=entry.source.retrieved_at,
        note=f"{entry.source.kind}: {note}",
    )


def _optional_str(item: Mapping[str, Any], key: str, index: int) -> str | None:
    value = item.get(key)
    if value is not None and not isinstance(value, str):
        raise CatalogParseError(f"data[{index}].{key}: expected a string")
    return value


def _optional_positive_int(item: Mapping[str, Any], key: str, index: int) -> int | None:
    value = item.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CatalogParseError(f"data[{index}].{key}: expected a positive integer")
    return int(value)
