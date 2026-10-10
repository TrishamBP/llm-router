"""Compare model versions on a consistent workload of measured outcomes.

Every comparison requires one dataset version, one evaluation version, and the same query set
for every model; otherwise `ComparisonError` is raised. All conclusions come from measured
`OutcomeRecord`s: nothing here assumes a newer version is better, and missing measurements stay
`None` rather than being imputed.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from llm_router.schemas import OutcomeRecord, OutcomeStatus


class ComparisonError(ValueError):
    """Outcomes do not form a consistent workload."""


@dataclass(frozen=True, slots=True)
class ModelSummary:
    model_id: str
    n: int
    success_rate: float
    error_rate: float  # failures and refusals
    timeout_rate: float
    mean_quality: float | None  # over outcomes with a quality score
    quality_n: int
    latency_p50_ms: float | None  # over outcomes with latency_total_ms
    latency_p95_ms: float | None
    mean_estimated_cost: Decimal | None  # None unless every outcome has an estimate
    mean_actual_cost: Decimal | None  # None unless every outcome has a billed cost
    estimate_error: Decimal | None  # mean(actual - estimated) when both are complete


@dataclass(frozen=True, slots=True)
class ComparisonReport:
    dataset_version: str
    evaluation_version: str
    overall: Mapping[str, ModelSummary]
    by_task_category: Mapping[str, Mapping[str, ModelSummary]]


@dataclass(frozen=True, slots=True)
class Savings:
    model_id: str
    baseline_model_id: str
    mean_cost: Decimal
    baseline_mean_cost: Decimal
    relative_savings: Decimal  # 1 - cost / baseline_cost


@dataclass(frozen=True, slots=True)
class RegretReport:
    mean_regret: float
    per_query: Mapping[str, float]


def percentile(values: Sequence[float], q: float) -> float:
    """Percentile with linear interpolation between closest ranks (NumPy's default method)."""
    if not values:
        raise ValueError("percentile of an empty sequence")
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(model_id: str, outcomes: Sequence[OutcomeRecord]) -> ModelSummary:
    if not outcomes:
        raise ValueError(f"no outcomes for '{model_id}'")
    n = len(outcomes)
    statuses = [o.status for o in outcomes]
    qualities = [o.quality_score for o in outcomes if o.quality_score is not None]
    latencies = [o.latency_total_ms for o in outcomes if o.latency_total_ms is not None]
    estimated = [o.estimated_cost for o in outcomes]
    actual = [o.actual_cost for o in outcomes]
    complete_estimates = all(c is not None for c in estimated)
    complete_actuals = all(c is not None for c in actual)

    def mean_decimal(values: Iterable[Decimal | None]) -> Decimal:
        return sum((v for v in values if v is not None), Decimal(0)) / n

    estimate_error = None
    if complete_estimates and complete_actuals:
        estimate_error = mean_decimal(
            a - e for a, e in zip(actual, estimated, strict=True) if a is not None and e is not None
        )
    return ModelSummary(
        model_id=model_id,
        n=n,
        success_rate=statuses.count(OutcomeStatus.SUCCESS) / n,
        error_rate=(statuses.count(OutcomeStatus.FAILURE) + statuses.count(OutcomeStatus.REFUSAL))
        / n,
        timeout_rate=statuses.count(OutcomeStatus.TIMEOUT) / n,
        mean_quality=sum(qualities) / len(qualities) if qualities else None,
        quality_n=len(qualities),
        latency_p50_ms=percentile(latencies, 50) if latencies else None,
        latency_p95_ms=percentile(latencies, 95) if latencies else None,
        mean_estimated_cost=mean_decimal(estimated) if complete_estimates else None,
        mean_actual_cost=mean_decimal(actual) if complete_actuals else None,
        estimate_error=estimate_error,
    )


def compare_models(outcomes: Sequence[OutcomeRecord]) -> ComparisonReport:
    dataset_version, evaluation_version = _check_workload(outcomes)
    by_model: dict[str, list[OutcomeRecord]] = defaultdict(list)
    by_category: dict[str, dict[str, list[OutcomeRecord]]] = defaultdict(lambda: defaultdict(list))
    for outcome in outcomes:
        by_model[outcome.model_id].append(outcome)
        category = outcome.task_category or "uncategorized"
        by_category[category][outcome.model_id].append(outcome)
    return ComparisonReport(
        dataset_version=dataset_version,
        evaluation_version=evaluation_version,
        overall={m: summarize(m, rows) for m, rows in sorted(by_model.items())},
        by_task_category={
            category: {m: summarize(m, rows) for m, rows in sorted(models.items())}
            for category, models in sorted(by_category.items())
        },
    )


def pareto_front(summaries: Mapping[str, ModelSummary]) -> tuple[str, ...]:
    """Models not dominated on (higher mean quality, lower mean estimated cost)."""
    points = {
        m: (s.mean_quality, s.mean_estimated_cost)
        for m, s in summaries.items()
        if s.mean_quality is not None and s.mean_estimated_cost is not None
    }
    front = []
    for m, (quality, cost) in points.items():
        dominated = any(
            q2 >= quality and c2 <= cost and (q2 > quality or c2 < cost)
            for other, (q2, c2) in points.items()
            if other != m
        )
        if not dominated:
            front.append(m)
    return tuple(sorted(front))


def cheapest_meeting_quality(
    summaries: Mapping[str, ModelSummary], min_quality: float
) -> str | None:
    qualifying = [
        s
        for s in summaries.values()
        if s.mean_quality is not None
        and s.mean_quality >= min_quality
        and s.mean_estimated_cost is not None
    ]
    if not qualifying:
        return None
    return min(qualifying, key=lambda s: (s.mean_estimated_cost, s.model_id)).model_id


def quality_constrained_savings(
    summaries: Mapping[str, ModelSummary], baseline_model_id: str, min_quality: float
) -> Savings | None:
    """Savings of the cheapest model meeting `min_quality`, relative to a fixed baseline model."""
    baseline = summaries[baseline_model_id]
    chosen = cheapest_meeting_quality(summaries, min_quality)
    if chosen is None or baseline.mean_estimated_cost is None:
        return None
    cost = summaries[chosen].mean_estimated_cost
    assert cost is not None  # guaranteed by cheapest_meeting_quality
    if baseline.mean_estimated_cost == 0:
        return None
    return Savings(
        model_id=chosen,
        baseline_model_id=baseline_model_id,
        mean_cost=cost,
        baseline_mean_cost=baseline.mean_estimated_cost,
        relative_savings=1 - cost / baseline.mean_estimated_cost,
    )


def best_quality_under_latency(
    summaries: Mapping[str, ModelSummary], max_latency_p95_ms: float
) -> str | None:
    qualifying = [
        s
        for s in summaries.values()
        if s.mean_quality is not None
        and s.latency_p95_ms is not None
        and s.latency_p95_ms <= max_latency_p95_ms
    ]
    if not qualifying:
        return None
    return min(qualifying, key=lambda s: (-(s.mean_quality or 0.0), s.model_id)).model_id


def routing_regret(
    routed: Mapping[str, str],
    outcomes: Sequence[OutcomeRecord],
    *,
    eligible: Mapping[str, Collection[str]] | None = None,
    failure_score: float = 0.0,
) -> RegretReport:
    """Per-query regret: best measured score among eligible candidates minus the routed score.

    Unsuccessful outcomes score `failure_score`. Successful outcomes without a quality score
    cannot be compared and raise `ComparisonError`.
    """
    _check_workload(outcomes)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for o in outcomes:
        if o.status != OutcomeStatus.SUCCESS:
            scores[o.query_id][o.model_id] = failure_score
        elif o.quality_score is None:
            raise ComparisonError(f"{o.query_id}/{o.model_id}: successful outcome without quality")
        else:
            scores[o.query_id][o.model_id] = o.quality_score

    per_query: dict[str, float] = {}
    for query_id, model_id in sorted(routed.items()):
        measured = scores.get(query_id, {})
        allowed = set(eligible[query_id]) if eligible is not None else set(measured)
        candidates = {m: s for m, s in measured.items() if m in allowed}
        if model_id not in measured:
            raise ComparisonError(f"no measured outcome for routed model {model_id} on {query_id}")
        if model_id not in allowed:
            raise ComparisonError(f"routed model {model_id} was not eligible for {query_id}")
        per_query[query_id] = max(candidates.values()) - measured[model_id]
    if not per_query:
        raise ComparisonError("no routed queries")
    return RegretReport(mean_regret=sum(per_query.values()) / len(per_query), per_query=per_query)


def _check_workload(outcomes: Sequence[OutcomeRecord]) -> tuple[str, str]:
    if not outcomes:
        raise ComparisonError("no outcomes")
    datasets = {o.query_dataset_version for o in outcomes}
    evaluations = {o.evaluation_version for o in outcomes}
    if len(datasets) != 1 or len(evaluations) != 1:
        raise ComparisonError(
            f"outcomes mix dataset versions {sorted(datasets)} or "
            f"evaluation versions {sorted(evaluations)}"
        )
    queries_by_model: dict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    for o in outcomes:
        key = (o.model_id, o.query_id)
        if key in seen:
            raise ComparisonError(f"duplicate outcome for {o.model_id} on {o.query_id}")
        seen.add(key)
        queries_by_model[o.model_id].add(o.query_id)
    query_sets = {frozenset(q) for q in queries_by_model.values()}
    if len(query_sets) != 1:
        raise ComparisonError("models were evaluated on different query sets")
    return datasets.pop(), evaluations.pop()
