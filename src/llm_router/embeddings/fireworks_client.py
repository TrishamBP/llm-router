"""Fireworks.ai embeddings via the OpenAI-compatible endpoint.

Usage (connectivity test, one synthetic query, one API request):
    uv run python -m llm_router.embeddings.fireworks_client --test

Credentials come from the environment or a local, git-ignored `.env`:
    FIREWORKS_API_KEY, FIREWORKS_EMBEDDING_MODEL, FIREWORKS_BASE_URL (optional)
The API key is never printed, logged, written to artifacts, or included in exceptions.

Retry policy: 429, timeouts, connection errors, and 5xx are retried with bounded exponential
backoff and full jitter (honouring Retry-After). Authentication, permission, unknown-model,
and request-validation errors are permanent and are never retried.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from llm_router.config import EmbeddingConfig, load_config
from llm_router.logging import configure_logging, get_logger

log = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"


class EmbeddingError(Exception):
    """Base class. Messages never contain credentials."""


class MissingCredentialsError(EmbeddingError):
    pass


class AuthenticationFailedError(EmbeddingError):
    pass


class ModelNotAvailableError(EmbeddingError):
    """Unknown model ID, or a model that does not serve embeddings."""


class RequestRejectedError(EmbeddingError):
    """The provider rejected the request itself (e.g. input too long). Not retried."""

    def __init__(self, message: str, status: int | None) -> None:
        super().__init__(message)
        self.status = status


class RetriesExhaustedError(EmbeddingError):
    pass


class InvalidEmbeddingResponseError(EmbeddingError):
    pass


@dataclass(frozen=True)
class FireworksSettings:
    api_key: str = field(repr=False)
    model: str
    base_url: str = DEFAULT_BASE_URL

    @classmethod
    def from_env(cls, *, dotenv_path: str | Path | None = ".env") -> FireworksSettings:
        if dotenv_path is not None:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path, override=False)
        api_key = os.environ.get("FIREWORKS_API_KEY", "").strip()
        model = os.environ.get("FIREWORKS_EMBEDDING_MODEL", "").strip()
        missing = [
            name
            for name, value in (
                ("FIREWORKS_API_KEY", api_key),
                ("FIREWORKS_EMBEDDING_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise MissingCredentialsError(
                f"missing environment variable(s): {', '.join(missing)}. "
                "Set them in a local .env file (see .env.example); never commit the key."
            )
        base_url = os.environ.get("FIREWORKS_BASE_URL", "").strip() or DEFAULT_BASE_URL
        return cls(api_key=api_key, model=model, base_url=base_url)


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: np.ndarray  # (n, d) float32, rows in input order
    prompt_tokens: int | None
    total_tokens: int | None
    reported_model: str | None
    attempts: int
    elapsed_s: float


class EmbeddingClient(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch: ...


def parse_embedding_response(
    response: Any, expected: int
) -> tuple[np.ndarray, int | None, int | None]:
    """Validate an embeddings response and return (vectors, prompt_tokens, total_tokens)."""
    data = getattr(response, "data", None)
    if not isinstance(data, list) or len(data) != expected:
        got = len(data) if isinstance(data, list) else type(data).__name__
        raise InvalidEmbeddingResponseError(f"expected {expected} embeddings, got {got}")
    items = sorted(data, key=lambda d: getattr(d, "index", 0))
    if [getattr(d, "index", i) for i, d in enumerate(items)] != list(range(expected)):
        raise InvalidEmbeddingResponseError("embedding indices are not 0..n-1")
    vectors = []
    for item in items:
        values = getattr(item, "embedding", None)
        if not isinstance(values, list) or not values:
            raise InvalidEmbeddingResponseError(
                "embedding is not a non-empty list of floats (is this an embeddings model?)"
            )
        vectors.append(values)
    dims = {len(v) for v in vectors}
    if len(dims) != 1:
        raise InvalidEmbeddingResponseError(f"inconsistent embedding dimensions {sorted(dims)}")
    matrix = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise InvalidEmbeddingResponseError("embedding contains non-finite values")
    usage = getattr(response, "usage", None)
    prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
    total = getattr(usage, "total_tokens", None) if usage is not None else None
    return matrix, prompt, total


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


class FireworksEmbeddingClient:
    """Thin, retrying wrapper around `OpenAI(...).embeddings.create`."""

    def __init__(
        self,
        settings: FireworksSettings,
        config: EmbeddingConfig,
        *,
        openai_client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = settings.model
        self._secret = settings.api_key
        self._config = config
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._clock = clock
        self._last_request: float | None = None
        self.requests = 0
        if openai_client is None:
            from openai import OpenAI

            # SDK-level retries off: retries are handled (and bounded) here.
            openai_client = OpenAI(
                api_key=settings.api_key,
                base_url=settings.base_url,
                timeout=config.timeout_s,
                max_retries=0,
            )
        self._client = openai_client

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts:
            raise ValueError("embed() needs at least one text")
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        if self._config.dimensions is not None:
            kwargs["dimensions"] = self._config.dimensions
        started = self._clock()
        for attempt in range(1, self._config.max_attempts + 1):
            self._respect_rate_limit()
            try:
                self.requests += 1
                response = self._client.embeddings.create(**kwargs)
            except Exception as exc:  # classified below; unknown errors re-raised
                delay = self._classify(exc, attempt)
                log.warning(
                    "embedding request failed; retrying", attempt=attempt, delay_s=round(delay, 2)
                )
                self._sleep(delay)
                continue
            vectors, prompt, total = parse_embedding_response(response, len(texts))
            return EmbeddingBatch(
                vectors=vectors,
                prompt_tokens=prompt,
                total_tokens=total,
                reported_model=getattr(response, "model", None),
                attempts=attempt,
                elapsed_s=self._clock() - started,
            )
        raise RetriesExhaustedError(
            f"embedding request failed after {self._config.max_attempts} attempts "
            "(rate limits, timeouts, or server errors)"
        )

    def _respect_rate_limit(self) -> None:
        interval = 60.0 / self._config.requests_per_minute
        now = self._clock()
        if self._last_request is not None and now - self._last_request < interval:
            self._sleep(interval - (now - self._last_request))
        self._last_request = self._clock()

    def _classify(self, exc: Exception, attempt: int) -> float:
        """Raise for permanent errors; return the backoff delay for transient ones."""
        import openai

        status = getattr(exc, "status_code", None)
        message = _redact(str(getattr(exc, "message", "") or exc), self._secret)[:300]
        if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
            raise AuthenticationFailedError(
                f"Fireworks rejected the credentials (HTTP {status}). Check FIREWORKS_API_KEY."
            ) from None
        if isinstance(exc, openai.NotFoundError):
            raise ModelNotAvailableError(
                f"model '{self.model}' not found or not available for embeddings "
                f"(HTTP 404): {message}"
            ) from None
        if isinstance(exc, openai.BadRequestError | openai.UnprocessableEntityError):
            raise RequestRejectedError(
                f"request rejected (HTTP {status}): {message}", status
            ) from None
        transient = isinstance(
            exc, openai.RateLimitError | openai.APITimeoutError | openai.APIConnectionError
        ) or (isinstance(exc, openai.APIStatusError) and (status or 0) >= 500)
        if not transient:
            raise EmbeddingError(
                f"unexpected embedding error ({type(exc).__name__}): {message}"
            ) from None
        if attempt >= self._config.max_attempts:
            raise RetriesExhaustedError(
                f"embedding request failed after {attempt} attempts; last error "
                f"{type(exc).__name__} (HTTP {status})"
            ) from None
        cap = min(self._config.max_backoff_s, self._config.initial_backoff_s * 2 ** (attempt - 1))
        delay = self._rng.uniform(0, cap)  # full jitter
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            delay = min(max(delay, retry_after), self._config.max_backoff_s)
        return delay


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None
    return seconds if seconds is not None and math.isfinite(seconds) and seconds >= 0 else None


def connectivity_test(client: EmbeddingClient, query: str, out_dir: Path) -> dict[str, Any]:
    """Embed one synthetic query, validate it, and save the vector and metadata."""
    batch = client.embed([query])
    vector = batch.vectors[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vector.npy", vector)
    metadata = {
        "model": client.model,
        "reported_model": batch.reported_model,
        "dimension": int(vector.shape[0]),
        "norm": float(np.linalg.norm(vector)),
        "prompt_tokens": batch.prompt_tokens,
        "total_tokens": batch.total_tokens,
        "latency_s": round(batch.elapsed_s, 3),
        "attempts": batch.attempts,
        "tested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "query_chars": len(query),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fireworks embeddings connectivity test")
    parser.add_argument("--test", action="store_true", help="embed one synthetic query")
    parser.add_argument("--config", default="configs/complexity.yaml")
    args = parser.parse_args(argv)
    if not args.test:
        parser.error("nothing to do; pass --test")
    configure_logging()
    cfg = load_config(args.config)
    try:
        settings = FireworksSettings.from_env()
        client = FireworksEmbeddingClient(settings, cfg.embeddings)
        metadata = connectivity_test(
            client, cfg.embeddings.connectivity_query, cfg.paths.embeddings_dir / "connectivity"
        )
    except EmbeddingError as exc:
        raise SystemExit(f"connectivity test failed: {exc}") from None
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
