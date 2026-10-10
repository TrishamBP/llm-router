"""Prediction contract between the hybrid router and the decision policy.

The hybrid model (neural router + XGBoost router + fusion) will implement `UtilityPredictor`.
No implementation exists yet; this module defines only the input/output contract and its
validation, so routing code can be written and tested against it.

Contract:
- Input: query features (from `llm_router.features`) and the *eligible* candidates for the
  request (already filtered; the predictor must not re-filter or add candidates).
- Output: exactly one `CandidatePrediction` per input candidate, matched by canonical model ID.
- A value the model cannot predict is `None`. It must never be replaced by a default.
- Cost is not predicted directly: the predictor estimates completion tokens, and cost is then
  computed from registry pricing (`llm_router.evaluation.pricing`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from llm_router.models.registry import ModelCandidate


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    model_id: str
    quality: float | None  # predicted quality on the benchmark scale, [0, 1]
    latency_p95_ms: float | None  # predicted P95 end-to-end latency
    output_tokens: int | None  # predicted completion tokens

    def __post_init__(self) -> None:
        if self.quality is not None and not (0.0 <= self.quality <= 1.0):
            raise ValueError(f"{self.model_id}: quality must be in [0, 1], got {self.quality}")
        if self.latency_p95_ms is not None and (
            not math.isfinite(self.latency_p95_ms) or self.latency_p95_ms < 0
        ):
            raise ValueError(f"{self.model_id}: latency_p95_ms must be finite and >= 0")
        if self.output_tokens is not None and self.output_tokens < 0:
            raise ValueError(f"{self.model_id}: output_tokens must be >= 0")


class UtilityPredictor(Protocol):
    def predict(
        self, query_features: Mapping[str, object], candidates: Sequence[ModelCandidate]
    ) -> Sequence[CandidatePrediction]: ...


class PredictionContractError(ValueError):
    """The predictor's output does not match the candidates it was given."""


def validate_predictions(
    candidates: Sequence[ModelCandidate], predictions: Sequence[CandidatePrediction]
) -> dict[str, CandidatePrediction]:
    expected = {c.model_id for c in candidates}
    by_id: dict[str, CandidatePrediction] = {}
    for prediction in predictions:
        if prediction.model_id in by_id:
            raise PredictionContractError(f"duplicate prediction for '{prediction.model_id}'")
        by_id[prediction.model_id] = prediction
    missing = sorted(expected - by_id.keys())
    unexpected = sorted(by_id.keys() - expected)
    if missing or unexpected:
        raise PredictionContractError(
            f"predictions must cover exactly the eligible candidates; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return by_id
