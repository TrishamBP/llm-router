"""XGBoost / MLP training, serialisation, inference, config selection, and the runner.

All data is synthetic and embeddings come from the offline fake client: no API calls.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pydantic
import pytest
import yaml

from llm_router.config import LABELS, load_config
from llm_router.embeddings.fake import FakeEmbeddingClient
from llm_router.embeddings.generate import resolve_embeddings_dir
from llm_router.features import complexity_features
from llm_router.features.complexity_features import load_split_features
from llm_router.inference.classifier import (
    ArtifactError,
    ComplexityRouter,
    ComplexityRouterError,
    EmbeddingModelMismatchError,
)
from llm_router.training import complexity_common as common
from llm_router.training.evaluate_classifiers import evaluate_model_dir
from llm_router.training.run_comparison import ComparisonRunner, _in_process
from llm_router.training.train_xgboost import ResourceBudgetError, train_xgboost

QUERY = "Design a provably optimal distributed system for biology."


def _val(cfg, feature_set):
    split, _ = load_split_features(
        cfg.paths.processed_dir, resolve_embeddings_dir(cfg), "validation", feature_set
    )
    return split


def _spy_splits(monkeypatch):
    loaded: list[str] = []
    original = complexity_features.load_embedded_split

    def spy(emb_dir, split, **kwargs):
        loaded.append(split)
        return original(emb_dir, split, **kwargs)

    monkeypatch.setattr(complexity_features, "load_embedded_split", spy)
    return loaded


def _write_config(cfg, tmp_path: Path, **classifier) -> Path:
    data = cfg.model_dump(mode="json")
    data["classifier"] = {**data["classifier"], **classifier}
    path = tmp_path / "configs" / "complexity.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def xgb_model(small_cfg):
    meta = train_xgboost(small_cfg, feature_set="embeddings+engineered")
    return small_cfg, Path(meta["model_dir"]), meta


# ---------------------------------------------------------------- XGBoost
def test_xgboost_saves_reloads_and_matches_the_booster(xgb_model):
    import xgboost as xgb

    cfg, model_dir, meta = xgb_model
    assert {p.name for p in model_dir.iterdir()} == {
        "model.ubj",
        "metadata.json",
        "feature_schema.json",
    }
    assert meta["classes"] == meta["probability_columns"] == list(LABELS)
    assert [v["class_weight"] for v in meta["variants"]] == ["none", "balanced"]
    schema = json.loads((model_dir / "feature_schema.json").read_text())
    assert schema["feature_set"] == "embeddings+engineered" and schema["n_features"] == 64 + len(
        complexity_features.text_features.FEATURE_NAMES
    )
    val = _val(cfg, "embeddings+engineered")
    X = val.matrix.rows(0, val.matrix.n_rows)
    booster = xgb.Booster()
    booster.load_model(model_dir / "model.ubj")
    direct = booster.inplace_predict(X)

    router = ComplexityRouter.from_pretrained(model_dir, offline=True)
    emb = np.load(resolve_embeddings_dir(cfg) / "validation.npy")
    indices, proba = router.predict_arrays(emb, val.texts)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba, direct, atol=1e-6)
    assert np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    assert np.array_equal(indices, proba.argmax(axis=1))
    # Saved validation metrics are reproduced exactly through the inference path.
    assert common.evaluate_proba(val.y, proba)["macro_f1"] == pytest.approx(
        meta["validation_metrics"]["macro_f1"]
    )


def test_xgboost_training_never_loads_the_test_split(small_cfg, monkeypatch):
    loaded = _spy_splits(monkeypatch)
    train_xgboost(small_cfg, feature_set="embeddings")
    assert sorted(set(loaded)) == ["train", "validation"]


def test_xgboost_smoke_and_budgets(small_cfg):
    meta = train_xgboost(small_cfg, feature_set="engineered", smoke=True)
    assert meta["output_name"].endswith("_smoke") and meta["smoke"] is True
    tight_time = small_cfg.model_copy(
        update={"resources": small_cfg.resources.model_copy(update={"max_train_seconds": 1e-9})}
    )
    with pytest.raises(ResourceBudgetError, match="budget"):
        train_xgboost(tight_time, feature_set="engineered")
    tight_memory = small_cfg.model_copy(
        update={"resources": small_cfg.resources.model_copy(update={"memory_budget_gb": 1e-9})}
    )
    with pytest.raises(ResourceBudgetError, match="exceeds"):
        train_xgboost(tight_memory, feature_set="embeddings")


def test_class_weighting_only_when_imbalance_justifies_it():
    balanced = np.repeat(np.arange(3, dtype=np.int32), 10)
    options, plan = common.class_weight_plan(balanced, ("none", "balanced"), 2.0)
    assert options == ["none"] and plan["train_max_min_ratio"] == 1.0
    skewed = np.array([0] * 30 + [1] * 10 + [2] * 5, dtype=np.int32)
    options, _ = common.class_weight_plan(skewed, ("none", "balanced"), 2.0)
    assert options == ["none", "balanced"]
    weights = common.balanced_class_weights(skewed)
    assert weights.dtype == np.float32 and weights[2] > weights[1] > weights[0]


# ---------------------------------------------------------------- inference router
def test_online_prediction_schema_and_label_order(xgb_model, fake_client):
    _, model_dir, _ = xgb_model
    router = ComplexityRouter.from_pretrained(model_dir, embedding_client=fake_client)
    result = router.predict(QUERY).to_dict()
    assert set(result) == {"label", "probabilities", "classifier", "embedding_model"}
    assert list(result["probabilities"]) == list(LABELS)
    assert result["label"] == max(result["probabilities"], key=result["probabilities"].get)
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert result["classifier"] == "xgboost"
    assert result["embedding_model"] == fake_client.model
    assert router.predict(QUERY) == router.predict(QUERY)  # deterministic


def test_engineered_model_requires_texts_and_matching_embedder(xgb_model):
    cfg, model_dir, _ = xgb_model
    router = ComplexityRouter.from_pretrained(model_dir, offline=True)
    emb = np.load(resolve_embeddings_dir(cfg) / "validation.npy")
    with pytest.raises(ComplexityRouterError, match="texts"):
        router.predict_embeddings(emb[:2])
    with pytest.raises(ComplexityRouterError, match="dimension 32"):
        router.predict_embeddings(emb[:2, :32], texts=["a", "b"])
    with pytest.raises(EmbeddingModelMismatchError):
        ComplexityRouter.from_pretrained(
            model_dir, embedding_client=FakeEmbeddingClient(model="other/model")
        )


def test_engineered_only_model_needs_no_embedding_api(small_cfg):
    meta = train_xgboost(small_cfg, feature_set="engineered")
    router = ComplexityRouter.from_pretrained(meta["model_dir"])  # no client, no key needed
    assert router.predict(QUERY).label in LABELS


@pytest.mark.parametrize(
    "damage",
    [
        "missing_dir",
        "missing_model",
        "corrupt_model",
        "corrupt_metadata",
        "schema_dim",
        "feature_hash",
        "label_order",
        "unknown_type",
    ],
)
def test_missing_or_corrupt_artifacts_are_refused(xgb_model, tmp_path, damage):
    _, model_dir, _ = xgb_model
    target = tmp_path / "copy"
    shutil.copytree(model_dir, target)
    meta_path, schema_path = target / "metadata.json", target / "feature_schema.json"
    meta, schema = json.loads(meta_path.read_text()), json.loads(schema_path.read_text())
    if damage == "missing_dir":
        shutil.rmtree(target)
    elif damage == "missing_model":
        (target / "model.ubj").unlink()
    elif damage == "corrupt_model":
        (target / "model.ubj").write_bytes(b"not a model")
    elif damage == "corrupt_metadata":
        meta_path.write_text("{ truncated")
    elif damage == "schema_dim":
        schema["embedding_dimension"] = 128
        schema_path.write_text(json.dumps(schema))
    elif damage == "feature_hash":
        schema["engineered"]["definition_hash"] = "0" * 16
        schema_path.write_text(json.dumps(schema))
    elif damage == "label_order":
        meta["probability_columns"] = ["easy", "hard", "medium"]
        meta_path.write_text(json.dumps(meta))
    else:
        meta["classifier"] = "random_forest"
        meta_path.write_text(json.dumps(meta))
    with pytest.raises(ArtifactError):
        ComplexityRouter.from_pretrained(target, offline=True)


def test_batched_evaluation_matches_single_pass(xgb_model):
    cfg, model_dir, _ = xgb_model
    small = evaluate_model_dir(cfg, model_dir, split="validation", batch_rows=5, latency_samples=3)
    large = evaluate_model_dir(cfg, model_dir, split="validation", latency_samples=3)
    assert small["validation"] == large["validation"]
    assert small["artifact_bytes"] > 0 and small["local_inference_latency_ms"]["median"] > 0


# ---------------------------------------------------------------- config selection
def test_classifier_selected_through_config(xgb_model, tmp_path):
    cfg, model_dir, _ = xgb_model
    path = _write_config(cfg, tmp_path, type="xgboost", model_dir=str(model_dir))
    router = ComplexityRouter.from_config(path, offline=True)
    assert router.classifier == "xgboost" and router.feature_set == "embeddings+engineered"

    wrong = _write_config(cfg, tmp_path, type="mlp", model_dir=str(model_dir))
    with pytest.raises(ArtifactError, match="'mlp' was configured"):
        ComplexityRouter.from_config(wrong, offline=True)

    default = load_config(_write_config(cfg, tmp_path, type="mlp", model_dir=None))
    assert default.classifier_dir() == cfg.paths.models_dir / "mlp"

    invalid = _write_config(cfg, tmp_path, type="random_forest")
    with pytest.raises(pydantic.ValidationError):
        load_config(invalid)


def test_shipped_config_defaults(repo_root):
    cfg = load_config(repo_root / "configs" / "complexity.yaml")
    assert cfg.resources.threads == 2
    assert cfg.xgboost.n_estimators == 200 and cfg.xgboost.max_depth == 4
    assert cfg.mlp.hidden_dims == (256, 128) and cfg.mlp.batch_size == 128
    assert cfg.mlp.max_epochs == 30


# ---------------------------------------------------------------- MLP (needs torch)
def test_mlp_checkpoint_save_reload_cpu_and_determinism(small_cfg, monkeypatch):
    torch = pytest.importorskip("torch")
    from llm_router.training.train_mlp import train_mlp

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # must still stay on CPU
    loaded = _spy_splits(monkeypatch)
    meta = train_mlp(small_cfg, feature_set="embeddings+engineered", output_name="mlp_a")
    assert sorted(set(loaded)) == ["train", "validation"]  # test split never loaded
    model_dir = Path(meta["model_dir"])
    assert {p.name for p in model_dir.iterdir()} == {
        "model.pt",
        "standardizer.npz",
        "metadata.json",
        "feature_schema.json",
    }  # only the best checkpoint
    assert meta["environment"]["device"] == "cpu"

    router = ComplexityRouter.from_pretrained(model_dir, offline=True)
    params = list(router._backend.backend.model.parameters())
    assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in params)
    val = _val(small_cfg, "embeddings+engineered")
    emb = np.load(resolve_embeddings_dir(small_cfg) / "validation.npy")
    _, proba = router.predict_arrays(emb, val.texts)
    assert proba.shape == (len(emb), 3) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
    # The saved checkpoint is the selected epoch: its validation metrics are reproduced.
    assert common.evaluate_proba(val.y, proba)["macro_f1"] == pytest.approx(
        meta["validation_metrics"]["macro_f1"]
    )
    _, again = router.predict_arrays(emb, val.texts)
    assert np.array_equal(proba, again)

    second = train_mlp(small_cfg, feature_set="embeddings+engineered", output_name="mlp_b")
    first_state = torch.load(model_dir / "model.pt", weights_only=True)["state_dict"]
    second_state = torch.load(Path(second["model_dir"]) / "model.pt", weights_only=True)[
        "state_dict"
    ]
    assert all(torch.equal(first_state[k], second_state[k]) for k in first_state)


def test_corrupt_mlp_checkpoint_is_refused(small_cfg):
    pytest.importorskip("torch")
    from llm_router.training.train_mlp import train_mlp

    meta = train_mlp(small_cfg, feature_set="embeddings")
    (Path(meta["model_dir"]) / "model.pt").write_bytes(b"garbage")
    with pytest.raises(ArtifactError):
        ComplexityRouter.from_pretrained(meta["model_dir"], offline=True)


# ---------------------------------------------------------------- comparison runner
def test_runner_records_failures_and_continues(small_cfg):
    def runner(cfg, exp, output_name, smoke):
        if exp.family == "mlp":
            raise RuntimeError("simulated MLP failure")
        return _in_process(cfg, exp, output_name, smoke)

    report = ComparisonRunner(
        small_cfg, None, only=("B", "C", "D"), baseline="skip", in_process=runner
    ).run()
    exps = report["experiments"]
    assert exps["B"]["status"] == exps["D"]["status"] == "trained"
    assert exps["C"]["status"] == "failed"
    assert any(f["stage"] == "train C" for f in report["failures"])
    assert report["selected"]["xgboost"]["experiment"] in {"B", "D"}
    assert "mlp" not in report["selected"]
    assert report["evaluation_split"] == "test"
    assert "test" in exps["B"]["held_out_evaluation"]
    out = small_cfg.paths.reports_dir / "comparison"
    assert (out / "comparison.json").exists() and (out / "comparison.md").exists()
    assert (small_cfg.paths.models_dir / "xgboost" / "model.ubj").exists()


def test_runner_smoke_never_evaluates_on_the_test_split(small_cfg, monkeypatch):
    from llm_router.training import evaluate_classifiers

    evaluated: list[str] = []
    original = evaluate_classifiers.load_embedded_split

    def spy(emb_dir, split, **kwargs):
        evaluated.append(split)
        return original(emb_dir, split, **kwargs)

    monkeypatch.setattr(evaluate_classifiers, "load_embedded_split", spy)
    report = ComparisonRunner(
        small_cfg,
        None,
        smoke=True,
        only=("B",),
        in_process=_in_process,
        near_duplicate_audit=False,
    ).run()
    assert report["evaluation_split"] == "validation"
    assert evaluated and set(evaluated) == {"validation"}
    assert (small_cfg.paths.models_dir / "xgboost_smoke").exists()


def test_training_cli_runs_in_a_child_process(small_cfg, tmp_path):
    path = _write_config(small_cfg, tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_router.training.train_xgboost",
            "--config",
            str(path),
            "--feature-set",
            "engineered",
            "--threads",
            "1",
            "--max-seconds",
            "0",
            "--memory-budget-gb",
            "0",
            "--output-name",
            "cli_xgb",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3] / "src")},
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    summary = json.loads(proc.stdout[proc.stdout.index("{") :])
    assert summary["classifier"] == "xgboost" and Path(summary["model_dir"]).name == "cli_xgb"
