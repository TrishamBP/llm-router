"""Version-aware routing orchestration.

    request -> candidate filter (registry) -> predictor (hybrid model) -> pricing -> policy

The registry supplies candidates; the predictor supplies quality/latency/output-token estimates;
pricing turns predicted tokens into model-specific cost; the policy selects. No step contains
family- or version-specific conditionals, so adding a registry entry needs no code change.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from llm_router.evaluation.pricing import (
    CostEstimate,
    CostResult,
    TokenUsage,
    estimate_candidate_cost,
)
from llm_router.models.registry import ModelCandidate, ModelRegistry
from llm_router.models.utility_predictor import UtilityPredictor, validate_predictions
from llm_router.routing.candidate_filter import CandidateRequest, FilterResult, filter_candidates
from llm_router.routing.policy import (
    Decision,
    DecisionStatus,
    PolicyConfig,
    ScoredCandidate,
    decide,
)


class RouteStatus(StrEnum):
    SELECTED = "selected"
    FALLBACK = "fallback"  # chosen by the fallback policy; constraints not satisfied
    NO_FEASIBLE_CANDIDATE = "no_feasible_candidate"
    NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"


@dataclass(frozen=True, slots=True)
class RoutingResult:
    status: RouteStatus
    candidate: ModelCandidate | None
    filter_result: FilterResult
    decision: Decision | None
    costs: Mapping[str, CostResult]

    @property
    def model_id(self) -> str | None:
        return self.candidate.model_id if self.candidate else None

    @property
    def provider(self) -> str | None:
        return self.candidate.provider if self.candidate else None

    @property
    def provider_model_id(self) -> str | None:
        """The exact identifier to send to the provider."""
        return self.candidate.provider_metadata.api_model_id if self.candidate else None

    def explain(self) -> str:
        parts = [f"status={self.status}", self.filter_result.explain()]
        if self.decision is not None:
            parts.append(self.decision.explanation)
        return "\n".join(parts)


class Router:
    def __init__(
        self,
        registry: ModelRegistry,
        predictor: UtilityPredictor,
        policy: PolicyConfig,
        *,
        clock: Callable[[], date] = date.today,
    ) -> None:
        self._registry = registry
        self._predictor = predictor
        self._policy = policy
        self._clock = clock

    def route(
        self, query_features: Mapping[str, object], request: CandidateRequest
    ) -> RoutingResult:
        filtered = filter_candidates(self._registry, request, today=self._clock())
        if not filtered.has_eligible:
            return RoutingResult(RouteStatus.NO_ELIGIBLE_CANDIDATE, None, filtered, None, {})

        predictions = validate_predictions(
            filtered.eligible, self._predictor.predict(query_features, filtered.eligible)
        )
        costs: dict[str, CostResult] = {}
        scored: list[ScoredCandidate] = []
        for candidate in filtered.eligible:
            prediction = predictions[candidate.model_id]
            cost_value = None
            if prediction.output_tokens is not None:
                # Assumes no cache hits: the request does not yet describe expected cache usage.
                usage = TokenUsage(
                    input_tokens=request.input_tokens, output_tokens=prediction.output_tokens
                )
                cost = estimate_candidate_cost(candidate, usage)
                costs[candidate.model_id] = cost
                if isinstance(cost, CostEstimate):
                    cost_value = cost.total
            scored.append(
                ScoredCandidate(
                    model_id=candidate.model_id,
                    quality=prediction.quality,
                    cost=cost_value,
                    latency_p95_ms=prediction.latency_p95_ms,
                )
            )

        decision = decide(scored, self._policy)
        selected = (
            self._registry.get(decision.selected_model_id)
            if decision.selected_model_id is not None
            else None
        )
        status = {
            DecisionStatus.SELECTED: RouteStatus.SELECTED,
            DecisionStatus.FALLBACK: RouteStatus.FALLBACK,
            DecisionStatus.NO_FEASIBLE_CANDIDATE: RouteStatus.NO_FEASIBLE_CANDIDATE,
        }[decision.status]
        return RoutingResult(status, selected, filtered, decision, costs)
