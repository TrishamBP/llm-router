from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm_router.embeddings.fake import FakeEmbeddingClient
from llm_router.embeddings.generate import EmbeddingJob
from llm_router.inference.classifier import (
    ComplexityRouter,
    ComplexityRouterError,
    EmbeddingModelMismatchError,
    OfflineModeError,
)
from llm_router.training import train_complexity
from llm_router.training.evaluate_complexity import evaluate
from llm_router.training.smoke_complexity import run_smoke
from llm_router.training.train_complexity import classification_metrics, train


@pytest.fixture
def trained(connected, fake_client):
    EmbeddingJob(connected, fake_client).run()
    summary = train(connected, kinds=("logreg", "svm"))
    return connected, Path(summary["classifiers"]["logreg"]["model_dir"]), summary


def test_training_never_loads_the_test_split(connected, fake_client, monkeypatch):
    EmbeddingJob(connected, fake_client).run()
    loaded = []
    original = train_complexity.load_embedded_split

    def spy(emb_dir, split):
        loaded.append(split)
        return original(emb_dir, split)

    monkeypatch.setattr(train_complexity, "load_embedded_split", spy)
    train(connected)
    assert sorted(set(loaded)) == ["train", "validation"]


def test_training_saves_model_encoder_and_metadata(trained):
    _, model_dir, summary = trained
    assert {p.name for p in model_dir.iterdir()} == {
        "model.joblib",
        "label_encoder.joblib",
        "metadata.json",
    }
    meta = json.loads((model_dir / "metadata.json").read_text())
    assert meta["classes"] == ["easy", "hard", "medium"]
    assert meta["embedding_model"] == "fake/hashed-bow-64" and meta["embedding_dimension"] == 64
    assert len(meta["grid_results"]) == 8  # 4 C values x 2 class-weight settings
    assert meta["validation_calibration"]["ece_top_class"] >= 0
    assert set(summary["classifiers"]) == {"logreg", "svm"}


def test_reloaded_router_predicts_with_probabilities(trained, fake_client):
    _, model_dir, _ = trained
    router = ComplexityRouter.from_pretrained(model_dir, embedding_client=fake_client)
    result = router.predict("Design a provably optimal distributed system for biology.")
    assert result.label in {"easy", "medium", "hard"}
    assert result.probabilities is not None
    assert sum(result.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.to_dict()["classifier"] == "logistic_regression"
    batch = router.predict_batch(["What is python?"] * 70)  # spans more than one API batch
    assert len(batch) == 70 and len(fake_client.calls) >= 2


def test_offline_mode_uses_cached_embeddings_only(trained, fake_client):
    cfg, model_dir, _ = trained
    router = ComplexityRouter.from_pretrained(model_dir, offline=True)
    with pytest.raises(OfflineModeError):
        router.predict("anything")
    vectors = np.load(next(cfg.paths.embeddings_dir.glob("*/test.npy")))
    assert len(router.predict_embeddings(vectors)) == len(vectors)


def test_mismatched_embedding_model_is_refused(trained):
    _, model_dir, _ = trained
    with pytest.raises(EmbeddingModelMismatchError):
        ComplexityRouter.from_pretrained(
            model_dir, embedding_client=FakeEmbeddingClient(model="x/y")
        )


def test_dimension_mismatch_is_refused(trained):
    _, model_dir, _ = trained
    router = ComplexityRouter.from_pretrained(model_dir, offline=True)
    with pytest.raises(ComplexityRouterError, match="dimension 32"):
        router.predict_embeddings(np.zeros((1, 32)))


def test_svm_router_has_no_probabilities(trained):
    _, _, summary = trained
    router = ComplexityRouter.from_pretrained(
        summary["classifiers"]["svm"]["model_dir"], offline=True
    )
    assert router.predict_embeddings(np.ones((1, 64)) / 8)[0].probabilities is None


def test_evaluation_writes_reports(trained):
    cfg, model_dir, _ = trained
    result = evaluate(cfg, model_dir, latency_samples=10)
    reports = cfg.paths.reports_dir
    for name in (
        "metrics.json",
        "classification_report.txt",
        "confusion_matrix.png",
        "evaluation_report.md",
    ):
        assert (reports / name).exists(), name
    assert result["test"]["n"] > 0 and result["local_inference_latency_ms"]["median"] > 0
    assert "hard_predicted_easy" in result["test"]


def test_hard_predicted_easy_is_counted():
    metrics = classification_metrics(
        ["hard", "hard", "hard", "easy"], ["easy", "hard", "medium", "easy"]
    )
    assert metrics["hard_predicted_easy"] == {"count": 1, "rate_of_true_hard": pytest.approx(1 / 3)}
    assert metrics["hard_recall"] == pytest.approx(1 / 3)


def test_smoke_pipeline_runs_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    result = run_smoke(tmp_path)
    assert result["complete_after_resume"] is True
    assert result["example_prediction"]["embedding_model"] == "fake/hashed-bow-64"
