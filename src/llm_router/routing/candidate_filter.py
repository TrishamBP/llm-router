"""Request-level candidate eligibility.

Given a request description and the registry, return the candidates that may serve the request
and, for every other candidate, every reason it was excluded. Deterministic and inference-free:
no neural, XGBoost, or provider calls. Capabilities come only from explicit registry metadata;
unknown values exclude the candidate when the request depends on them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from llm_router.evaluation.pricing import TokenUsage, UnknownCost, estimate_candidate_cost
from llm_router.models.registry import (
    ROUTABLE_STATUSES,
    Capability,
    LifecycleStatus,
    ModelCandidate,
)


class ExclusionReason(StrEnum):
    NOT_ENABLED = "not_enabled"
    STATUS_NOT_ALLOWED = "status_not_allowed"
    NO_PROVIDER_MODEL_ID = "no_provider_model_id"
    NOT_YET_EFFECTIVE = "not_yet_effective"
    NO_LONGER_EFFECTIVE = "no_longer_effective"
    PROVIDER_NOT_ALLOWED = "provider_not_allowed"
    REGION_UNKNOWN = "region_unknown"
    REGION_NOT_ALLOWED = "region_not_allowed"
    NOT_ALLOWLISTED = "not_allowlisted"
    DENYLISTED = "denylisted"
    CONTEXT_WINDOW_UNKNOWN = "context_window_unknown"
    CONTEXT_WINDOW_TOO_SMALL = "context_window_too_small"
    OUTPUT_LIMIT_UNKNOWN = "output_limit_unknown"
    OUTPUT_LIMIT_TOO_SMALL = "output_limit_too_small"
    CAPABILITY_UNKNOWN = "capability_unknown"
    CAPABILITY_UNSUPPORTED = "capability_unsupported"
    PARAMETER_UNKNOWN = "parameter_unknown"
    PARAMETER_UNSUPPORTED = "parameter_unsupported"
    PRICING_UNKNOWN = "pricing_unknown"


DEFAULT_ALLOWED_STATUSES = frozenset({LifecycleStatus.ACTIVE, LifecycleStatus.LEGACY})


@dataclass(frozen=True, slots=True)
class CandidateRequest:
    """What a request needs from a candidate. Token counts are the request's expected sizes."""

    input_tokens: int  # all prompt tokens, including any expected cache reads/writes
    max_output_tokens: int
    required_capabilities: frozenset[Capability] = frozenset()
    required_parameters: frozenset[str] = frozenset()
    allowed_providers: frozenset[str] | None = None  # None: any provider
    allowed_regions: frozenset[str] | None = None  # None: no regional restriction
    allowlist: frozenset[str] | None = None  # canonical model IDs; None: no allowlist
    denylist: frozenset[str] = frozenset()
    allowed_statuses: frozenset[LifecycleStatus] = DEFAULT_ALLOWED_STATUSES
    require_pricing: bool = True

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.max_output_tokens <= 0:
            raise ValueError("input_tokens must be >= 0 and max_output_tokens must be > 0")
        not_routable = self.allowed_statuses - ROUTABLE_STATUSES
        if not_routable:
            names = ", ".join(sorted(not_routable))
            raise ValueError(f"statuses can never be routed to: {names}")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        input_tokens: int,
        max_output_tokens: int,
        **overrides: Any,
    ) -> CandidateRequest:
        """Build a request from the `candidate_filter` section of routing_policy.yaml."""

        def optional_set(key: str) -> frozenset[str] | None:
            value = config.get(key)
            return None if value is None else frozenset(value)

        values: dict[str, Any] = {
            "allowed_providers": optional_set("allowed_providers"),
            "allowed_regions": optional_set("allowed_regions"),
            "allowlist": optional_set("allowlist"),
            "denylist": frozenset(config.get("denylist") or ()),
            "require_pricing": bool(config.get("require_pricing", True)),
        }
        if config.get("allowed_statuses") is not None:
            values["allowed_statuses"] = frozenset(
                LifecycleStatus(s) for s in config["allowed_statuses"]
            )
        values.update(overrides)
        return cls(input_tokens=input_tokens, max_output_tokens=max_output_tokens, **values)


@dataclass(frozen=True, slots=True)
class Exclusion:
    model_id: str
    reasons: tuple[ExclusionReason, ...]
    details: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FilterResult:
    eligible: tuple[ModelCandidate, ...]
    excluded: tuple[Exclusion, ...] = field(default=())

    @property
    def has_eligible(self) -> bool:
        return bool(self.eligible)

    def explain(self) -> str:
        lines = [f"{len(self.eligible)} eligible: {', '.join(c.model_id for c in self.eligible)}"]
        lines += [f"excluded {e.model_id}: {'; '.join(e.details)}" for e in self.excluded]
        return "\n".join(lines)


def filter_candidates(
    candidates: Iterable[ModelCandidate], request: CandidateRequest, *, today: date
) -> FilterResult:
    """Split candidates into eligible and excluded, preserving the input (registry) order."""
    eligible: list[ModelCandidate] = []
    excluded: list[Exclusion] = []
    for candidate in candidates:
        problems = _check(candidate, request, today)
        if problems:
            reasons = tuple(dict.fromkeys(reason for reason, _ in problems))
            details = tuple(detail for _, detail in problems)
            excluded.append(Exclusion(candidate.model_id, reasons, details))
        else:
            eligible.append(candidate)
    return FilterResult(tuple(eligible), tuple(excluded))


def _check(
    candidate: ModelCandidate, request: CandidateRequest, today: date
) -> list[tuple[ExclusionReason, str]]:
    problems: list[tuple[ExclusionReason, str]] = []

    def fail(reason: ExclusionReason, detail: str) -> None:
        problems.append((reason, detail))

    lifecycle = candidate.lifecycle
    caps = candidate.capabilities
    meta = candidate.provider_metadata

    if not lifecycle.enabled:
        fail(ExclusionReason.NOT_ENABLED, "not enabled in the registry")
    if lifecycle.status not in request.allowed_statuses:
        fail(ExclusionReason.STATUS_NOT_ALLOWED, f"lifecycle status '{lifecycle.status}'")
    if meta.api_model_id is None:
        fail(ExclusionReason.NO_PROVIDER_MODEL_ID, "no verified provider model ID")
    if lifecycle.effective_from is not None and today < lifecycle.effective_from:
        fail(ExclusionReason.NOT_YET_EFFECTIVE, f"effective from {lifecycle.effective_from}")
    if lifecycle.effective_until is not None and today > lifecycle.effective_until:
        fail(ExclusionReason.NO_LONGER_EFFECTIVE, f"effective until {lifecycle.effective_until}")

    if (
        request.allowed_providers is not None
        and candidate.provider not in request.allowed_providers
    ):
        fail(ExclusionReason.PROVIDER_NOT_ALLOWED, f"provider '{candidate.provider}' not allowed")
    if request.allowed_regions is not None:
        if meta.region is None:
            fail(ExclusionReason.REGION_UNKNOWN, "region restriction set but region unknown")
        elif meta.region not in request.allowed_regions:
            fail(ExclusionReason.REGION_NOT_ALLOWED, f"region '{meta.region}' not allowed")
    if request.allowlist is not None and candidate.model_id not in request.allowlist:
        fail(ExclusionReason.NOT_ALLOWLISTED, "not in allowlist")
    if candidate.model_id in request.denylist:
        fail(ExclusionReason.DENYLISTED, "denylisted")

    # The context window is assumed to hold the prompt plus the requested output.
    needed_context = request.input_tokens + request.max_output_tokens
    if caps.context_window_tokens is None:
        fail(ExclusionReason.CONTEXT_WINDOW_UNKNOWN, "context window unknown")
    elif caps.context_window_tokens < needed_context:
        fail(
            ExclusionReason.CONTEXT_WINDOW_TOO_SMALL,
            f"context window {caps.context_window_tokens} < {needed_context} needed",
        )
    if caps.max_output_tokens is None:
        fail(ExclusionReason.OUTPUT_LIMIT_UNKNOWN, "max output tokens unknown")
    elif caps.max_output_tokens < request.max_output_tokens:
        fail(
            ExclusionReason.OUTPUT_LIMIT_TOO_SMALL,
            f"max output {caps.max_output_tokens} < {request.max_output_tokens} requested",
        )

    for capability in sorted(request.required_capabilities):
        supported = caps.supports(capability)
        if supported is None:
            fail(ExclusionReason.CAPABILITY_UNKNOWN, f"support for '{capability}' unknown")
        elif not supported:
            fail(ExclusionReason.CAPABILITY_UNSUPPORTED, f"'{capability}' not supported")
    for parameter in sorted(request.required_parameters):
        supported = caps.parameters.get(parameter)
        if supported is None:
            fail(ExclusionReason.PARAMETER_UNKNOWN, f"support for parameter '{parameter}' unknown")
        elif not supported:
            fail(ExclusionReason.PARAMETER_UNSUPPORTED, f"parameter '{parameter}' not supported")

    if request.require_pricing:
        # Price the worst case for this request: full prompt uncached, full output budget.
        usage = TokenUsage(
            input_tokens=request.input_tokens, output_tokens=request.max_output_tokens
        )
        cost = estimate_candidate_cost(candidate, usage)
        if isinstance(cost, UnknownCost):
            fail(ExclusionReason.PRICING_UNKNOWN, cost.reason)

    return problems
