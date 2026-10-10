"""Decision policy: choose one candidate from scored, eligible candidates.

Default objective (configured in configs/routing_policy.yaml):

    m* = argmin_m  C(q, m)   subject to  Q(q, m) >= Q_min  and  L95(q, m) <= L_max

Constraints are never relaxed silently. If no candidate is feasible, the configured fallback
applies and the decision is marked `constraints_satisfied=False`:
- `no_feasible_candidate`: return no selection, with per-candidate violations.
- `maximize_utility`: maximise  w_q*Q - w_c*C - w_l*L95  over candidates with complete predictions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any


class FallbackMode(StrEnum):
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"
    MAXIMIZE_UTILITY = "maximize_utility"


class DecisionStatus(StrEnum):
    SELECTED = "selected"  # all constraints satisfied
    FALLBACK = "fallback"  # selected by the fallback policy; constraints NOT satisfied
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"


class Violation(StrEnum):
    COST_UNKNOWN = "cost_unknown"
    QUALITY_UNKNOWN = "quality_unknown"
    QUALITY_BELOW_MIN = "quality_below_min"
    LATENCY_UNKNOWN = "latency_unknown"
    LATENCY_ABOVE_MAX = "latency_above_max"


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    quality: float
    cost: float  # per currency unit
    latency_ms: float

    def __post_init__(self) -> None:
        values = (self.quality, self.cost, self.latency_ms)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("utility weights must be finite and >= 0")


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    min_quality: float | None = None
    max_latency_p95_ms: float | None = None
    fallback: FallbackMode = FallbackMode.NO_FEASIBLE_CANDIDATE
    utility_weights: UtilityWeights | None = None

    def __post_init__(self) -> None:
        if self.fallback == FallbackMode.MAXIMIZE_UTILITY and self.utility_weights is None:
            raise ValueError("fallback 'maximize_utility' requires utility_weights")

    @classmethod
    def from_mapping(cls, decision: Mapping[str, Any]) -> PolicyConfig:
        """Build from the `decision` section of routing_policy.yaml."""
        objective = decision.get("objective", "min_predicted_cost")
        if objective != "min_predicted_cost":
            raise ValueError(f"unsupported objective '{objective}'")
        constraints = decision.get("constraints") or {}
        weights = decision.get("utility_weights") or {}
        utility_weights = None
        if any(weights.get(k) is not None for k in ("quality", "cost", "latency_ms")):
            utility_weights = UtilityWeights(
                quality=float(weights["quality"]),
                cost=float(weights["cost"]),
                latency_ms=float(weights["latency_ms"]),
            )
        return cls(
            min_quality=constraints.get("min_quality"),
            max_latency_p95_ms=constraints.get("max_latency_p95_ms"),
            fallback=FallbackMode(decision.get("fallback", FallbackMode.NO_FEASIBLE_CANDIDATE)),
            utility_weights=utility_weights,
        )


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    model_id: str
    quality: float | None
    cost: Decimal | None
    latency_p95_ms: float | None


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    model_id: str
    violations: tuple[Violation, ...]
    utility: float | None = None

    @property
    def feasible(self) -> bool:
        return not self.violations


@dataclass(frozen=True, slots=True)
class Decision:
    status: DecisionStatus
    selected_model_id: str | None
    constraints_satisfied: bool
    assessments: tuple[CandidateAssessment, ...]  # ranked: feasible first, then by cost
    explanation: str


def decide(scored: Sequence[ScoredCandidate], config: PolicyConfig) -> Decision:
    assessments = {
        s.model_id: CandidateAssessment(s.model_id, _violations(s, config)) for s in scored
    }
    ranked = sorted(scored, key=lambda s: (not assessments[s.model_id].feasible, *_cost_key(s)))
    feasible = [s for s in ranked if assessments[s.model_id].feasible]

    if feasible:
        best = feasible[0]
        return Decision(
            status=DecisionStatus.SELECTED,
            selected_model_id=best.model_id,
            constraints_satisfied=True,
            assessments=tuple(assessments[s.model_id] for s in ranked),
            explanation=(
                f"lowest predicted cost among {len(feasible)} feasible candidate(s): "
                f"{best.model_id} (cost={best.cost}, quality={best.quality}, "
                f"latency_p95_ms={best.latency_p95_ms})"
            ),
        )

    if config.fallback == FallbackMode.MAXIMIZE_UTILITY and config.utility_weights is not None:
        weights = config.utility_weights
        utilities: dict[str, float] = {}
        for s in scored:
            if s.quality is not None and s.cost is not None and s.latency_p95_ms is not None:
                utilities[s.model_id] = (
                    weights.quality * s.quality
                    - weights.cost * float(s.cost)
                    - weights.latency_ms * s.latency_p95_ms
                )
        ranked_ids = sorted(utilities, key=lambda m: (-utilities[m], m))
        final = tuple(
            CandidateAssessment(a.model_id, a.violations, utilities.get(a.model_id))
            for a in sorted(
                assessments.values(),
                key=lambda a: (
                    a.model_id not in utilities,
                    -utilities.get(a.model_id, 0.0),
                    a.model_id,
                ),
            )
        )
        if ranked_ids:
            return Decision(
                status=DecisionStatus.FALLBACK,
                selected_model_id=ranked_ids[0],
                constraints_satisfied=False,
                assessments=final,
                explanation=(
                    "no candidate satisfied all constraints; fallback 'maximize_utility' "
                    f"selected {ranked_ids[0]} (utility={utilities[ranked_ids[0]]:.6g})"
                ),
            )
        return Decision(
            status=DecisionStatus.NO_FEASIBLE_CANDIDATE,
            selected_model_id=None,
            constraints_satisfied=False,
            assessments=final,
            explanation="no feasible candidate and none with complete predictions for fallback",
        )

    return Decision(
        status=DecisionStatus.NO_FEASIBLE_CANDIDATE,
        selected_model_id=None,
        constraints_satisfied=False,
        assessments=tuple(assessments[s.model_id] for s in ranked),
        explanation=f"no candidate satisfied all constraints ({len(scored)} assessed)",
    )


def _violations(s: ScoredCandidate, config: PolicyConfig) -> tuple[Violation, ...]:
    violations: list[Violation] = []
    if s.cost is None:
        violations.append(Violation.COST_UNKNOWN)
    if config.min_quality is not None:
        if s.quality is None:
            violations.append(Violation.QUALITY_UNKNOWN)
        elif s.quality < config.min_quality:
            violations.append(Violation.QUALITY_BELOW_MIN)
    if config.max_latency_p95_ms is not None:
        if s.latency_p95_ms is None:
            violations.append(Violation.LATENCY_UNKNOWN)
        elif s.latency_p95_ms > config.max_latency_p95_ms:
            violations.append(Violation.LATENCY_ABOVE_MAX)
    return tuple(violations)


def _cost_key(s: ScoredCandidate) -> tuple[Decimal, float, float, str]:
    """Order by cost, then higher quality, then lower latency, then model ID (deterministic)."""
    inf = Decimal("Infinity")
    return (
        s.cost if s.cost is not None else inf,
        -(s.quality if s.quality is not None else -math.inf),
        s.latency_p95_ms if s.latency_p95_ms is not None else math.inf,
        s.model_id,
    )
