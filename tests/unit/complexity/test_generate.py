from __future__ import annotations

import json

import numpy as np
import pytest

from llm_router.data.datasets import AlignmentError, load_embedded_split
from llm_router.data.preprocess import load_processed
from llm_router.embeddings.fake import FakeEmbeddingClient
from llm_router.embeddings.fireworks_client import (
    EmbeddingBatch,
    InvalidEmbeddingResponseError,
    RequestRejectedError,
)
from llm_router.embeddings.generate import EmbeddingJob, EmbeddingJobError


def test_requires_successful_connectivity_test(workspace, fake_client):
    with pytest.raises(EmbeddingJobError, match="connectivity test"):
        EmbeddingJob(workspace, fake_client)


def test_connectivity_for_another_model_is_rejected(connected):
    with pytest.raises(EmbeddingJobError, match="not 'other/model'"):
        EmbeddingJob(connected, FakeEmbeddingClient(model="other/model"))


def test_refuses_to_run_without_verified_split_isolation(connected, fake_client):
    path = connected.paths.processed_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["split_isolation"]["verified"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(EmbeddingJobError, match="isolation"):
        EmbeddingJob(connected, fake_client)


def test_rows_align_with_processed_examples(connected, fake_client):
    meta = EmbeddingJob(connected, fake_client).run()
    assert meta["complete"] and meta["dimension"] == fake_client.dimension
    examples, _ = load_processed(connected.paths.processed_dir)
    emb_dir = connected.paths.embeddings_dir / meta["cache_key"]
    for split in ("train", "validation", "test"):
        loaded = load_embedded_split(emb_dir, split)
        assert loaded.example_ids == tuple(e["example_id"] for e in examples[split])
        assert loaded.y.tolist() == [e["label"] for e in examples[split]]
        expected = fake_client.embed([e["query"] for e in examples[split]]).vectors
        np.testing.assert_allclose(loaded.X, expected)


def test_resume_does_not_re_embed_completed_batches(connected, fake_client):
    first = EmbeddingJob(connected, fake_client).run(max_batches=2)
    assert not first["complete"]
    embedded_first = [t for call in fake_client.calls for t in call]
    fake_client.calls.clear()
    EmbeddingJob(connected, fake_client).run()
    embedded_second = [t for call in fake_client.calls for t in call]
    assert not set(embedded_first) & set(embedded_second)
    examples, _ = load_processed(connected.paths.processed_dir)
    assert len(embedded_first) + len(embedded_second) == sum(len(v) for v in examples.values())


def test_completed_cache_is_reused_without_api_calls(connected, fake_client):
    EmbeddingJob(connected, fake_client).run()
    fake_client.calls.clear()
    meta = EmbeddingJob(connected, fake_client).run()
    assert meta["complete"] and fake_client.calls == []


def test_different_models_use_separate_caches(connected, fake_client):
    a = EmbeddingJob(connected, fake_client).key
    b = EmbeddingJob(
        connected, FakeEmbeddingClient(model="other/model"), require_connectivity=False
    ).key
    assert a != b


def test_changing_batch_size_requires_a_new_cache(connected, fake_client):
    EmbeddingJob(connected, fake_client).run(max_batches=1)
    changed = connected.model_copy(
        update={"embeddings": connected.embeddings.model_copy(update={"batch_size": 8})}
    )
    with pytest.raises(EmbeddingJobError, match="batch size"):
        EmbeddingJob(changed, fake_client)


class PoisonClient(FakeEmbeddingClient):
    """Rejects any batch containing a poisoned query (like an over-long input)."""

    def __init__(self, poison: str) -> None:
        super().__init__()
        self.poison = poison

    def embed(self, texts):
        if self.poison in texts:
            raise RequestRejectedError("input too long", 400)
        return super().embed(texts)


def test_rejected_inputs_are_isolated_and_recorded(connected):
    examples, _ = load_processed(connected.paths.processed_dir)
    poisoned = examples["train"][5]
    client = PoisonClient(poisoned["query"])
    meta = EmbeddingJob(connected, client).run()
    assert meta["failed_examples"]["train"] == {poisoned["example_id"]: "rejected (HTTP 400)"}
    loaded = load_embedded_split(connected.paths.embeddings_dir / meta["cache_key"], "train")
    assert poisoned["example_id"] not in loaded.example_ids
    assert len(loaded.example_ids) == len(examples["train"]) - 1


class ShrinkingClient(FakeEmbeddingClient):
    def embed(self, texts):
        batch = super().embed(texts)
        if len(self.calls) > 1:
            return EmbeddingBatch(batch.vectors[:, :32], None, None, self.model, 1, 0.0)
        return batch


def test_dimension_change_mid_job_is_an_error(connected):
    with pytest.raises(InvalidEmbeddingResponseError, match="dimension 32 != expected 64"):
        EmbeddingJob(connected, ShrinkingClient()).run()


def test_plan_makes_no_api_calls_and_estimates_cost_only_with_price(connected, fake_client):
    job = EmbeddingJob(connected, fake_client)
    plan = job.plan()
    assert fake_client.calls == []
    assert plan["remaining_requests"] > 0 and plan["estimated_remaining_cost_usd"] is None
    job.run(max_batches=2)
    priced = connected.model_copy(
        update={
            "embeddings": connected.embeddings.model_copy(update={"price_per_million_tokens": 0.01})
        }
    )
    plan = EmbeddingJob(priced, fake_client).plan()
    assert plan["measured_tokens_per_char"] > 0
    assert plan["estimated_remaining_cost_usd"] is not None


def test_misaligned_mapping_is_detected(connected, fake_client):
    meta = EmbeddingJob(connected, fake_client).run()
    emb_dir = connected.paths.embeddings_dir / meta["cache_key"]
    X = np.load(emb_dir / "train.npy")
    np.save(emb_dir / "train.npy", X[:-1])
    with pytest.raises(AlignmentError, match="vectors but"):
        load_embedded_split(emb_dir, "train")
