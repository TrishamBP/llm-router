from __future__ import annotations

import json
import random
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest

from llm_router.config import EmbeddingConfig
from llm_router.embeddings.fireworks_client import (
    AuthenticationFailedError,
    FireworksEmbeddingClient,
    FireworksSettings,
    InvalidEmbeddingResponseError,
    MissingCredentialsError,
    ModelNotAvailableError,
    RequestRejectedError,
    RetriesExhaustedError,
    connectivity_test,
    parse_embedding_response,
)

SECRET = "fw_test_SECRET_value_123"  # fake value; never a real key
SETTINGS = FireworksSettings(api_key=SECRET, model="accounts/test/models/embed-test")
REQUEST = httpx.Request("POST", "https://api.fireworks.ai/inference/v1/embeddings")


def response(vectors, prompt_tokens=7, model="accounts/test/models/embed-test"):
    data = [SimpleNamespace(index=i, embedding=list(v)) for i, v in enumerate(vectors)]
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)
    return SimpleNamespace(data=data, usage=usage, model=model)


def status_error(cls, status, headers=None, message=f"bad things {SECRET}"):
    return cls(
        message, response=httpx.Response(status, request=REQUEST, headers=headers or {}), body=None
    )


class FakeOpenAI:
    """Mimics `client.embeddings.create`; plays a script of exceptions/responses."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_client(script, **cfg):
    sleeps: list[float] = []
    fake = FakeOpenAI(script)
    clock = iter(range(0, 10_000_000, 10)).__next__  # requests are always far apart
    client = FireworksEmbeddingClient(
        SETTINGS,
        EmbeddingConfig(**{"max_attempts": 4, "initial_backoff_s": 1, "max_backoff_s": 8, **cfg}),
        openai_client=fake,
        sleep=sleeps.append,
        rng=random.Random(0),
        clock=lambda: float(clock()),
    )
    return client, fake, sleeps


# --------------------------------------------------------------- credentials


def test_missing_credentials_error_is_safe(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.setenv("FIREWORKS_EMBEDDING_MODEL", "some-model")
    with pytest.raises(MissingCredentialsError, match="FIREWORKS_API_KEY") as excinfo:
        FireworksSettings.from_env(dotenv_path=None)
    assert "FIREWORKS_EMBEDDING_MODEL" not in str(excinfo.value)


def test_model_id_is_required(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", SECRET)
    monkeypatch.delenv("FIREWORKS_EMBEDDING_MODEL", raising=False)
    with pytest.raises(MissingCredentialsError, match="FIREWORKS_EMBEDDING_MODEL") as excinfo:
        FireworksSettings.from_env(dotenv_path=None)
    assert SECRET not in str(excinfo.value)


def test_settings_from_env_and_repr_hides_key(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", SECRET)
    monkeypatch.setenv("FIREWORKS_EMBEDDING_MODEL", "m")
    monkeypatch.delenv("FIREWORKS_BASE_URL", raising=False)
    settings = FireworksSettings.from_env(dotenv_path=None)
    assert settings.base_url == "https://api.fireworks.ai/inference/v1"
    assert SECRET not in repr(settings)


# --------------------------------------------------------------- parsing


def test_parses_response_in_index_order():
    resp = response([[1.0, 2.0], [3.0, 4.0]])
    resp.data.reverse()
    vectors, prompt, total = parse_embedding_response(resp, 2)
    assert vectors.dtype == np.float32 and vectors.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert (prompt, total) == (7, 7)


@pytest.mark.parametrize(
    ("resp", "message"),
    [
        (response([[1.0, 2.0]]), "expected 2 embeddings"),
        (response([[1.0, 2.0], [1.0]]), "inconsistent embedding dimensions"),
        (response([[1.0, float("nan")], [1.0, 2.0]]), "non-finite"),
        (
            SimpleNamespace(
                data=[
                    SimpleNamespace(index=0, embedding="abc"),
                    SimpleNamespace(index=1, embedding=[]),
                ]
            ),
            "non-empty list",
        ),
    ],
)
def test_rejects_invalid_responses(resp, message):
    with pytest.raises(InvalidEmbeddingResponseError, match=message):
        parse_embedding_response(resp, 2)


def test_missing_usage_is_reported_as_unknown():
    resp = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0])], usage=None)
    assert parse_embedding_response(resp, 1)[1:] == (None, None)


# --------------------------------------------------------------- retries & errors


def test_success_sends_model_input_and_float_encoding():
    client, fake, _ = make_client([response([[0.1, 0.2]])])
    batch = client.embed(["hello"])
    assert fake.calls == [{"model": SETTINGS.model, "input": ["hello"], "encoding_format": "float"}]
    assert batch.vectors.shape == (1, 2) and batch.attempts == 1 and batch.prompt_tokens == 7


def test_dimensions_parameter_only_sent_when_configured():
    client, fake, _ = make_client([response([[0.1, 0.2]])], dimensions=2)
    client.embed(["x"])
    assert fake.calls[0]["dimensions"] == 2


def test_rate_limit_is_retried_with_retry_after():
    client, fake, sleeps = make_client(
        [status_error(openai.RateLimitError, 429, {"retry-after": "3"}), response([[1.0]])]
    )
    batch = client.embed(["x"])
    assert batch.attempts == 2 and len(fake.calls) == 2
    assert sleeps == [3.0]  # honoured Retry-After (above the jittered backoff, below the cap)


@pytest.mark.parametrize(
    "error",
    [
        status_error(openai.InternalServerError, 500),
        openai.APITimeoutError(request=REQUEST),
        openai.APIConnectionError(request=REQUEST),
    ],
    ids=["5xx", "timeout", "connection"],
)
def test_transient_errors_are_retried(error):
    client, fake, sleeps = make_client([error, response([[1.0]])])
    assert client.embed(["x"]).attempts == 2
    assert len(sleeps) == 1 and 0 <= sleeps[0] <= 1  # first backoff cap is initial_backoff_s


def test_retries_are_bounded_with_capped_backoff():
    client, fake, sleeps = make_client([status_error(openai.RateLimitError, 429)] * 4)
    with pytest.raises(RetriesExhaustedError, match="after 4 attempts"):
        client.embed(["x"])
    assert len(fake.calls) == 4
    caps = [1, 2, 4]
    assert len(sleeps) == 3 and all(0 <= s <= c for s, c in zip(sleeps, caps, strict=True))


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (status_error(openai.AuthenticationError, 401), AuthenticationFailedError),
        (status_error(openai.PermissionDeniedError, 403), AuthenticationFailedError),
        (status_error(openai.NotFoundError, 404), ModelNotAvailableError),
        (status_error(openai.BadRequestError, 400), RequestRejectedError),
    ],
)
def test_permanent_errors_are_not_retried_and_never_leak_the_key(error, expected):
    client, fake, sleeps = make_client([error])
    with pytest.raises(expected) as excinfo:
        client.embed(["x"])
    assert len(fake.calls) == 1 and sleeps == []
    assert SECRET not in str(excinfo.value)
    assert excinfo.value.__cause__ is None  # original exception (with request details) dropped


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_client_side_rate_limit_spaces_requests():
    clock = FakeClock()
    client = FireworksEmbeddingClient(
        SETTINGS,
        EmbeddingConfig(requests_per_minute=60),
        openai_client=FakeOpenAI([response([[1.0]])] * 3),
        sleep=clock.sleep,
        clock=clock,
    )
    client.embed(["a"])
    clock.now += 0.25
    client.embed(["b"])  # 0.25 s after the first: must wait 0.75 s
    clock.now += 5.0
    client.embed(["c"])  # long after: no wait
    assert clock.sleeps == [pytest.approx(0.75)]  # 60 rpm = 1 request per second


def test_connectivity_test_saves_vector_and_metadata(tmp_path):
    client, _, _ = make_client([response([[0.6, 0.8]], prompt_tokens=5)])
    meta = connectivity_test(client, "synthetic", tmp_path / "conn")
    assert meta["dimension"] == 2 and meta["prompt_tokens"] == 5
    assert np.load(tmp_path / "conn" / "vector.npy").tolist() == pytest.approx([0.6, 0.8])
    saved = (tmp_path / "conn" / "metadata.json").read_text()
    assert SECRET not in saved and json.loads(saved)["model"] == SETTINGS.model
