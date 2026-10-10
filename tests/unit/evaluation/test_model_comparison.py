from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest

from llm_router.evaluation.model_comparison import (
    ComparisonError,
    best_quality_under_latency,
    compare_models,
    pareto_front,
    percentile,
    quality_constrained_savings,
    routing_regret,
)
from llm_router.schemas import OutcomeRecord, OutcomeStatus

MEASURED_AT = datetime(2026, 10, 1, tzinfo=UTC)


def outcome(
    model_id: str, query_id: str, quality: float | None = 0.8, **overrides
) -> OutcomeRecord:
    values = {
        "query_id": query_id,
        "query_dataset_version": "queries-v1",
        "task_category": "code",
        "model_id": model_id,
        "provider": model_id.split("/")[0],
        "provider_model_id": model_id.split("/")[1] + "-api",
        "family": "alpha",
        "version": "1",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "quality_score": quality,
        "quality_method": "exact_match" if quality is not None else None,
        "latency_total_ms": 1_000.0,
        "estimated_cost": Decimal("0.01"),
        "status": OutcomeStatus.SUCCESS,
        "measured_at": MEASURED_AT,
        "evaluation_version": "eval-v1",
    }
    values.update(overrides)
    return OutcomeRecord(**values)


def workload() -> list[OutcomeRecord]:
    rows = []
    for q, (qa, qb) in enumerate([(0.9, 0.6), (0.7, 0.8), (0.5, 0.4), (1.0, 0.9)]):
        rows.append(
            outcome(
                "p/a", f"q{q}", qa, estimated_cost=Decimal("0.04"), latency_total_ms=2_000.0 + q
            )
        )
        rows.append(
            outcome("p/b", f"q{q}", qb, estimated_cost=Decimal("0.01"), latency_total_ms=500.0 + q)
        )
    return rows


@pytest.mark.parametrize("q", [0, 25, 50, 90, 95, 100])
def test_percentile_matches_numpy_linear(q):
    values = [12.0, 3.0, 7.5, 100.0, 42.0, 8.0, 9.0]
    assert percentile(values, q) == pytest.approx(float(np.percentile(values, q)))


def test_summaries_report_quality_latency_and_cost():
    report = compare_models(workload())
    a = report.overall["p/a"]
    assert a.n == 4 and a.success_rate == 1.0
    assert a.mean_quality == pytest.approx(0.775)
    assert a.latency_p50_ms == pytest.approx(2_001.5)
    assert a.latency_p95_ms == pytest.approx(percentile([2000, 2001, 2002, 2003], 95))
    assert a.mean_estimated_cost == Decimal("0.04")
    assert a.mean_actual_cost is None  # no billed costs recorded
    assert report.dataset_version == "queries-v1" and report.evaluation_version == "eval-v1"
    assert set(report.by_task_category) == {"code"}


def test_error_and_timeout_rates_and_failed_quality():
    rows = [
        outcome("p/a", "q1"),
        outcome("p/a", "q2", None, status=OutcomeStatus.TIMEOUT, error_type="timeout"),
        outcome("p/a", "q3", None, status=OutcomeStatus.FAILURE, error_type="http_500"),
        outcome("p/a", "q4", None, status=OutcomeStatus.REFUSAL, error_type="refusal"),
    ]
    s = compare_models(rows).overall["p/a"]
    assert (s.success_rate, s.timeout_rate, s.error_rate) == (0.25, 0.25, 0.5)
    assert s.quality_n == 1  # failures are not scored as zero-quality successes


def test_estimated_and_actual_cost_are_kept_separate():
    rows = [
        outcome("p/a", "q1", estimated_cost=Decimal("0.010"), actual_cost=Decimal("0.012")),
        outcome("p/a", "q2", estimated_cost=Decimal("0.020"), actual_cost=Decimal("0.020")),
    ]
    s = compare_models(rows).overall["p/a"]
    assert s.mean_estimated_cost == Decimal("0.015")
    assert s.mean_actual_cost == Decimal("0.016")
    assert s.estimate_error == Decimal("0.001")
    partial = compare_models([*rows[:1], outcome("p/a", "q2", actual_cost=None)])
    assert partial.overall["p/a"].mean_actual_cost is None


@pytest.mark.parametrize(
    "rows",
    [
        [outcome("p/a", "q1"), outcome("p/b", "q1", query_dataset_version="queries-v2")],
        [outcome("p/a", "q1"), outcome("p/b", "q1", evaluation_version="eval-v2")],
        [outcome("p/a", "q1"), outcome("p/a", "q2"), outcome("p/b", "q1")],
        [outcome("p/a", "q1"), outcome("p/a", "q1")],
        [],
    ],
    ids=["dataset", "evaluation", "query-set", "duplicate", "empty"],
)
def test_inconsistent_workloads_are_rejected(rows):
    with pytest.raises(ComparisonError):
        compare_models(rows)


def test_cost_quality_trade_offs():
    summaries = compare_models(workload()).overall
    assert pareto_front(summaries) == ("p/a", "p/b")  # a: higher quality, b: cheaper
    savings = quality_constrained_savings(summaries, baseline_model_id="p/a", min_quality=0.65)
    assert savings is not None and savings.model_id == "p/b"
    assert savings.relative_savings == Decimal("0.75")
    assert quality_constrained_savings(summaries, "p/a", min_quality=0.99) is None


def test_latency_constrained_quality():
    summaries = compare_models(workload()).overall
    assert best_quality_under_latency(summaries, max_latency_p95_ms=10_000) == "p/a"
    assert best_quality_under_latency(summaries, max_latency_p95_ms=1_000) == "p/b"
    assert best_quality_under_latency(summaries, max_latency_p95_ms=10) is None


def test_routing_regret_against_best_measured_candidate():
    routed = {"q0": "p/a", "q1": "p/a", "q2": "p/b", "q3": "p/a"}
    report = routing_regret(routed, workload())
    assert report.per_query == pytest.approx({"q0": 0.0, "q1": 0.1, "q2": 0.1, "q3": 0.0})
    assert report.mean_regret == pytest.approx(0.05)


def test_routing_regret_respects_eligibility_and_failures():
    rows = [
        outcome("p/a", "q1", 0.9),
        outcome("p/b", "q1", None, status=OutcomeStatus.TIMEOUT, error_type="timeout"),
    ]
    assert routing_regret({"q1": "p/b"}, rows).per_query["q1"] == pytest.approx(0.9)
    only_b = routing_regret({"q1": "p/b"}, rows, eligible={"q1": ["p/b"]})
    assert only_b.per_query["q1"] == 0.0
    with pytest.raises(ComparisonError, match="not eligible"):
        routing_regret({"q1": "p/a"}, rows, eligible={"q1": ["p/b"]})


def test_outcome_records_reject_inconsistent_values():
    with pytest.raises(ValueError, match="only successful outcomes"):
        outcome("p/a", "q1", 0.5, status=OutcomeStatus.FAILURE)
    with pytest.raises(ValueError, match="requires quality_method"):
        outcome("p/a", "q1", 0.5, quality_method=None)
    with pytest.raises(ValueError, match="latency_ttft_ms cannot exceed"):
        outcome("p/a", "q1", latency_ttft_ms=2_000.0, latency_total_ms=1_000.0)
