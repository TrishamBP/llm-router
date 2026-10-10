"""Version-aware model registry: schema, validation, loading, and lookup.

The registry describes which model versions *could* be routed to and what is known about
them. It never makes routing decisions: per-request eligibility lives in
`llm_router.routing.candidate_filter`, and selection lives in `llm_router.routing.policy`.

Unknown metadata is represented as `None` and is never treated as supported, free, or fast.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

SCHEMA_VERSION = 1

_MODEL_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9._:@-]*$"
_PROVIDER_PATTERN = r"^[a-z0-9][a-z0-9_-]*$"
PRICE_FIELDS = ("input", "output", "cache_read", "cache_write_5m", "cache_write_1h")


class LifecycleStatus(StrEnum):
    UNVERIFIED = "unverified"  # existence/identifier not confirmed from an authoritative source
    ACTIVE = "active"  # provider: fully supported and recommended
    LEGACY = "legacy"  # provider: still served, no further updates
    DEPRECATED = "deprecated"  # provider: still served, retirement scheduled
    RETIRED = "retired"  # provider: requests fail
    UNAVAILABLE = "unavailable"  # not reachable for this project (access, region, account)


# Statuses that may ever be routed to. Everything else is excluded unconditionally.
ROUTABLE_STATUSES = frozenset(
    {LifecycleStatus.ACTIVE, LifecycleStatus.LEGACY, LifecycleStatus.DEPRECATED}
)


class BenchmarkStatus(StrEnum):
    NOT_EVALUATED = "not_evaluated"
    EVALUATED = "evaluated"
    STALE = "stale"  # evaluated, but the model, prompt set, or evaluator has since changed


class Capability(StrEnum):
    TOOLS = "tools"
    VISION = "vision"
    STRUCTURED_OUTPUT = "structured_output"


class RegistryError(Exception):
    """Base class for registry configuration and lookup errors."""


class RegistryValidationError(RegistryError):
    """The registry file or its records are malformed. `issues` lists every problem found."""

    def __init__(self, issues: Iterable[str], source: str | None = None) -> None:
        self.issues = tuple(issues)
        self.source = source
        where = f" in {source}" if source else ""
        super().__init__(f"invalid model registry{where}:\n  - " + "\n  - ".join(self.issues))


class UnknownModelError(RegistryError, KeyError):
    """No candidate with the requested canonical model ID exists in the registry."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "unknown model"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class Source(_Record):
    """Provenance for a group of fields: where the values came from and when they were checked."""

    url: str = Field(min_length=1)
    checked_at: date
    note: str | None = None


class PriceSet(_Record):
    """Prices per million tokens. `None` means unknown, never zero."""

    input: Decimal | None = Field(default=None, ge=0)
    output: Decimal | None = Field(default=None, ge=0)
    cache_read: Decimal | None = Field(default=None, ge=0)
    cache_write_5m: Decimal | None = Field(default=None, ge=0)
    cache_write_1h: Decimal | None = Field(default=None, ge=0)

    @field_validator(*PRICE_FIELDS, mode="before")
    @classmethod
    def _exact_decimal(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("price must be a number, not a boolean")
        # YAML yields floats; go through repr so 0.1 becomes Decimal("0.1"), not its binary value.
        return Decimal(repr(value)) if isinstance(value, float) else value

    def has_any_price(self) -> bool:
        return any(getattr(self, name) is not None for name in PRICE_FIELDS)


class PricingTier(PriceSet):
    """Alternative prices applied to the *entire* request once its prompt exceeds a threshold.

    The prompt size counts all input tokens: uncached input, cache reads, and cache writes.
    A tier's unknown component is unknown for that request; it never falls back to base prices.
    """

    name: str = Field(min_length=1)
    applies_when_prompt_tokens_over: int = Field(gt=0)
    prompt_token_basis: Literal["all_input_tokens"] = "all_input_tokens"
    scope: Literal["entire_request"] = "entire_request"


class Pricing(PriceSet):
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    unit: Literal["per_million_tokens"] = "per_million_tokens"
    tiers: tuple[PricingTier, ...] = ()
    verified_at: date | None = None
    sources: tuple[Source, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Pricing:
        thresholds = [tier.applies_when_prompt_tokens_over for tier in self.tiers]
        if thresholds != sorted(set(thresholds)):
            raise ValueError("tier thresholds must be unique and listed in increasing order")
        names = [tier.name for tier in self.tiers]
        if len(names) != len(set(names)):
            raise ValueError("tier names must be unique")
        has_prices = self.has_any_price() or any(tier.has_any_price() for tier in self.tiers)
        if has_prices and (self.verified_at is None or not self.sources):
            raise ValueError("prices require verified_at and at least one source")
        return self


class Capabilities(_Record):
    """Explicit capability metadata. `None` means unknown and is treated as unsupported."""

    context_window_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    supports_tools: bool | None = None
    supports_vision: bool | None = None
    supports_structured_output: bool | None = None
    # Request parameter -> whether it can be set (to a non-default value). Missing means unknown.
    parameters: dict[str, bool | None] = Field(default_factory=dict)
    verified_at: date | None = None
    sources: tuple[Source, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Capabilities:
        if (
            self.context_window_tokens is not None
            and self.max_output_tokens is not None
            and self.max_output_tokens > self.context_window_tokens
        ):
            raise ValueError("max_output_tokens cannot exceed context_window_tokens")
        return self

    def supports(self, capability: Capability) -> bool | None:
        return {
            Capability.TOOLS: self.supports_tools,
            Capability.VISION: self.supports_vision,
            Capability.STRUCTURED_OUTPUT: self.supports_structured_output,
        }[capability]


class Lifecycle(_Record):
    status: LifecycleStatus = LifecycleStatus.UNVERIFIED
    enabled: bool = False  # project opt-in; necessary but not sufficient for routing
    verified_at: date | None = None
    effective_from: date | None = None  # first date the candidate may be routed to
    effective_until: date | None = None  # last date the candidate may be routed to
    retirement_not_before: date | None = None  # provider commitment; informational only
    sources: tuple[Source, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Lifecycle:
        if (
            self.effective_from is not None
            and self.effective_until is not None
            and self.effective_from > self.effective_until
        ):
            raise ValueError("effective_from must not be after effective_until")
        if self.enabled and self.status not in ROUTABLE_STATUSES:
            raise ValueError(f"a candidate with status '{self.status}' cannot be enabled")
        if self.status != LifecycleStatus.UNVERIFIED and (
            self.verified_at is None or not self.sources
        ):
            raise ValueError(f"status '{self.status}' requires verified_at and at least one source")
        return self


class ProviderMetadata(_Record):
    api_model_id: str | None = Field(default=None, min_length=1)  # exact ID sent to the provider
    api_model_alias: str | None = None  # informational; routing always uses api_model_id
    deployment_name: str | None = None
    region: str | None = None
    sources: tuple[Source, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> ProviderMetadata:
        if self.api_model_id is not None and not self.sources:
            raise ValueError("api_model_id requires at least one source")
        return self


class Benchmark(_Record):
    """Curated summary of a reviewed evaluation run. Raw results live in evaluation artifacts."""

    status: BenchmarkStatus = BenchmarkStatus.NOT_EVALUATED
    quality_score: float | None = Field(default=None, ge=0, le=1)
    latency_p50_ms: float | None = Field(default=None, ge=0)
    latency_p95_ms: float | None = Field(default=None, ge=0)
    benchmark_dataset_version: str | None = None
    evaluation_run_id: str | None = None
    measured_at: date | None = None

    @model_validator(mode="after")
    def _check(self) -> Benchmark:
        metrics = (self.quality_score, self.latency_p50_ms, self.latency_p95_ms)
        if self.status == BenchmarkStatus.NOT_EVALUATED:
            if any(value is not None for value in metrics):
                raise ValueError("a not_evaluated benchmark cannot carry metrics")
        elif (
            self.measured_at is None
            or self.benchmark_dataset_version is None
            or self.evaluation_run_id is None
            or all(value is None for value in metrics)
        ):
            raise ValueError(
                f"benchmark status '{self.status}' requires measured_at, "
                "benchmark_dataset_version, evaluation_run_id, and at least one metric"
            )
        if (
            self.latency_p50_ms is not None
            and self.latency_p95_ms is not None
            and self.latency_p50_ms > self.latency_p95_ms
        ):
            raise ValueError("latency_p50_ms cannot exceed latency_p95_ms")
        return self


class ModelCandidate(_Record):
    """One routable destination: a specific model version on a specific provider."""

    model_id: str = Field(pattern=_MODEL_ID_PATTERN)  # canonical internal ID: "<provider>/<name>"
    display_name: str = Field(min_length=1)
    provider: str = Field(pattern=_PROVIDER_PATTERN)
    family: str = Field(min_length=1)  # descriptive only, never a routing destination
    version: str = Field(min_length=1)  # descriptive only
    lifecycle: Lifecycle = Field(default_factory=Lifecycle)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    pricing: Pricing = Field(default_factory=Pricing)
    provider_metadata: ProviderMetadata = Field(default_factory=ProviderMetadata)
    benchmark: Benchmark = Field(default_factory=Benchmark)
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ModelCandidate:
        if not self.model_id.startswith(f"{self.provider}/"):
            raise ValueError(f"model_id must start with the provider prefix '{self.provider}/'")
        if self.lifecycle.enabled and self.provider_metadata.api_model_id is None:
            raise ValueError("an enabled candidate requires provider_metadata.api_model_id")
        return self

    @property
    def is_routable(self) -> bool:
        """Registry-level gate only; request-specific checks happen in the candidate filter."""
        return (
            self.lifecycle.enabled
            and self.lifecycle.status in ROUTABLE_STATUSES
            and self.provider_metadata.api_model_id is not None
        )


class PendingCandidate(_Record):
    """A requested model version that could not be verified. Never routable."""

    requested_label: str = Field(min_length=1)
    family: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: Literal["unverified"] = "unverified"
    reason: str = Field(min_length=1)
    checked_at: date
    sources: tuple[Source, ...] = ()


class RegistryDocument(_Record):
    schema_version: Literal[1]
    models: tuple[ModelCandidate, ...]
    pending_candidates: tuple[PendingCandidate, ...] = ()


class ModelRegistry:
    """Immutable, deterministically ordered collection of validated model candidates."""

    def __init__(
        self,
        candidates: Iterable[ModelCandidate],
        pending: Iterable[PendingCandidate] = (),
        *,
        source: str | None = None,
    ) -> None:
        ordered = tuple(sorted(candidates, key=lambda c: c.model_id))
        issues = _identity_issues(ordered)
        if issues:
            raise RegistryValidationError(issues, source)
        self._candidates = ordered
        self._by_id = {c.model_id: c for c in ordered}
        self._pending = tuple(pending)
        self.source = source

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: str | None = None) -> ModelRegistry:
        # Top-level keys starting with "_" hold YAML anchors only and are not part of the schema.
        data = {key: value for key, value in data.items() if not str(key).startswith("_")}
        try:
            document = RegistryDocument.model_validate(data)
        except ValidationError as exc:
            raise RegistryValidationError(_format_validation_error(exc, data), source) from exc
        return cls(document.models, document.pending_candidates, source=source)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelRegistry:
        path = Path(path)
        return cls.from_mapping(load_registry_document(path), source=str(path))

    @property
    def candidates(self) -> tuple[ModelCandidate, ...]:
        return self._candidates

    @property
    def pending(self) -> tuple[PendingCandidate, ...]:
        return self._pending

    def __len__(self) -> int:
        return len(self._candidates)

    def __iter__(self) -> Iterator[ModelCandidate]:
        return iter(self._candidates)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._by_id

    def get(self, model_id: str) -> ModelCandidate:
        try:
            return self._by_id[model_id]
        except KeyError:
            raise UnknownModelError(f"no candidate with model_id '{model_id}'") from None

    def find_by_provider_id(self, provider: str, api_model_id: str) -> ModelCandidate | None:
        for candidate in self._candidates:
            meta = candidate.provider_metadata
            if candidate.provider == provider and meta.api_model_id == api_model_id:
                return candidate
        return None

    def select(
        self,
        *,
        provider: str | None = None,
        family: str | None = None,
        version: str | None = None,
        status: LifecycleStatus | None = None,
    ) -> tuple[ModelCandidate, ...]:
        return tuple(
            c
            for c in self._candidates
            if (provider is None or c.provider == provider)
            and (family is None or c.family == family)
            and (version is None or c.version == version)
            and (status is None or c.lifecycle.status == status)
        )

    def routable(self) -> tuple[ModelCandidate, ...]:
        return tuple(c for c in self._candidates if c.is_routable)


def load_registry_document(path: Path) -> Mapping[str, Any]:
    """Read and parse the registry YAML file without validating its records."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RegistryError(f"cannot read model registry {path}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RegistryValidationError([f"invalid YAML: {exc}"], str(path)) from exc
    if not isinstance(data, Mapping):
        raise RegistryValidationError(["top level must be a mapping"], str(path))
    return data


def _identity_issues(candidates: tuple[ModelCandidate, ...]) -> list[str]:
    issues: list[str] = []
    seen_ids: set[str] = set()
    seen_provider_ids: dict[tuple[str, str], str] = {}
    for candidate in candidates:
        if candidate.model_id in seen_ids:
            issues.append(f"duplicate model_id '{candidate.model_id}'")
        seen_ids.add(candidate.model_id)
        api_id = candidate.provider_metadata.api_model_id
        if api_id is not None:
            key = (candidate.provider, api_id)
            if key in seen_provider_ids:
                issues.append(
                    f"'{candidate.model_id}' and '{seen_provider_ids[key]}' share provider "
                    f"'{candidate.provider}' and api_model_id '{api_id}'"
                )
            else:
                seen_provider_ids[key] = candidate.model_id
    return issues


def _format_validation_error(exc: ValidationError, data: Mapping[str, Any]) -> list[str]:
    models = data.get("models")
    issues = []
    for error in exc.errors():
        loc = list(error["loc"])
        label = ".".join(str(part) for part in loc) or "<root>"
        if len(loc) >= 2 and loc[0] == "models" and isinstance(loc[1], int):
            model_id = None
            if (
                isinstance(models, list)
                and loc[1] < len(models)
                and isinstance(models[loc[1]], dict)
            ):
                model_id = models[loc[1]].get("model_id")
            rest = ".".join(str(part) for part in loc[2:])
            label = f"models[{loc[1]}]" + (f" ({model_id})" if model_id else "")
            label += f".{rest}" if rest else ""
        issues.append(f"{label}: {error['msg']}")
    return issues
