"""Typed records shared across modules."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class OutcomeStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"  # provider or client error after retries
    TIMEOUT = "timeout"
    REFUSAL = "refusal"  # the model declined (e.g. stop_reason "refusal")


class OutcomeRecord(BaseModel):
    """One measured result of running one candidate on one query.

    Provider-neutral: adapters translate provider responses into this record. Unmeasured values
    are `None`, never zero. Timing boundaries (client-side wall clock, including network):
    - `latency_ttft_ms`: request sent -> first output token received (streaming only).
    - `latency_total_ms`: request sent -> final token received.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    query_id: str = Field(min_length=1)
    query_dataset_version: str = Field(min_length=1)
    task_category: str | None = None

    model_id: str = Field(min_length=1)  # canonical registry ID
    provider: str = Field(min_length=1)
    provider_model_id: str = Field(min_length=1)  # exact ID sent to the provider
    family: str = Field(min_length=1)  # as recorded in the registry at evaluation time
    version: str = Field(min_length=1)

    prompt_tokens: int | None = Field(default=None, ge=0)  # uncached input tokens
    completion_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)

    quality_score: float | None = Field(default=None, ge=0, le=1)
    quality_method: str | None = None  # e.g. "exact_match", "llm_judge:<judge-id>", "human"

    latency_ttft_ms: float | None = Field(default=None, ge=0)
    latency_total_ms: float | None = Field(default=None, ge=0)

    estimated_cost: Decimal | None = Field(default=None, ge=0)
    actual_cost: Decimal | None = Field(default=None, ge=0)  # provider-reported, if available
    cost_currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")

    status: OutcomeStatus
    error_type: str | None = None
    retries: int = Field(default=0, ge=0)

    measured_at: datetime
    evaluation_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> OutcomeRecord:
        if self.quality_score is not None and self.quality_method is None:
            raise ValueError("quality_score requires quality_method")
        if self.status != OutcomeStatus.SUCCESS and self.quality_score is not None:
            raise ValueError("only successful outcomes carry a quality_score")
        if self.status == OutcomeStatus.SUCCESS and self.error_type is not None:
            raise ValueError("successful outcomes cannot have an error_type")
        if (
            self.latency_ttft_ms is not None
            and self.latency_total_ms is not None
            and self.latency_ttft_ms > self.latency_total_ms
        ):
            raise ValueError("latency_ttft_ms cannot exceed latency_total_ms")
        return self
