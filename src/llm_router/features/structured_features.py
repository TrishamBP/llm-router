"""Structured features.

Candidate-side features describe a model version from registry metadata so the hybrid router
can generalise across versions. Unknown metadata stays `None`; it is never imputed as 0 or False
(imputation, if any, is a modelling decision made explicitly in the training pipeline).
"""

from __future__ import annotations

from llm_router.models.registry import ModelCandidate

FeatureValue = str | int | float | bool | None


def candidate_features(candidate: ModelCandidate) -> dict[str, FeatureValue]:
    caps = candidate.capabilities
    pricing = candidate.pricing
    bench = candidate.benchmark

    def price(value: object) -> float | None:
        return None if value is None else float(str(value))

    return {
        "model_id": candidate.model_id,
        "provider": candidate.provider,
        "family": candidate.family,
        "version": candidate.version,
        "lifecycle_status": str(candidate.lifecycle.status),
        "context_window_tokens": caps.context_window_tokens,
        "max_output_tokens": caps.max_output_tokens,
        "supports_tools": caps.supports_tools,
        "supports_vision": caps.supports_vision,
        "supports_structured_output": caps.supports_structured_output,
        "price_input_per_mtok": price(pricing.input),
        "price_output_per_mtok": price(pricing.output),
        "price_cache_read_per_mtok": price(pricing.cache_read),
        "has_long_context_tier": bool(pricing.tiers),
        "benchmark_quality": bench.quality_score,
        "benchmark_latency_p95_ms": bench.latency_p95_ms,
    }
