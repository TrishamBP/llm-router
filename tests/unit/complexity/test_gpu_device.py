"""Device selection for XGBoost / MLP training. GPU tests skip when no CUDA GPU is usable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm_router.config import load_config
from llm_router.models.complexity_classifiers import XGBoostBackend
from llm_router.training import resources
from llm_router.training.resources import DeviceUnavailableError, resolve_device
from llm_router.training.train_mlp import train_mlp
from llm_router.training.train_xgboost import train_xgboost


def _cuda_usable() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


needs_cuda = pytest.mark.skipif(not _cuda_usable(), reason="no usable CUDA GPU")


def on(cfg, device):
    return cfg.model_copy(update={"resources": cfg.resources.model_copy(update={"device": device})})


def test_cpu_is_the_schema_default_and_shipped_config_uses_auto(repo_root):
    from llm_router.config import ResourcesConfig

    assert ResourcesConfig().device == "cpu"
    assert load_config(repo_root / "configs" / "complexity.yaml").resources.device == "auto"


@pytest.mark.parametrize("library", ["torch", "xgboost"])
def test_cpu_request_never_probes_the_gpu(library, monkeypatch):
    def boom():
        raise AssertionError("GPU probed")

    monkeypatch.setattr(resources, "_torch_cuda_ok", boom)
    monkeypatch.setattr(resources, "_xgboost_cuda_ok", boom)
    assert resolve_device("cpu", library) == ("cpu", "cpu requested")


@pytest.mark.parametrize("library", ["torch", "xgboost"])
def test_explicit_cuda_fails_loudly_without_a_gpu(library, monkeypatch):
    monkeypatch.setattr(resources, "_torch_cuda_ok", lambda: (False, "no GPU here"))
    monkeypatch.setattr(resources, "_xgboost_cuda_ok", lambda: (False, "no GPU here"))
    with pytest.raises(DeviceUnavailableError, match="no GPU here"):
        resolve_device("cuda", library)


def test_auto_falls_back_to_cpu_and_says_why(monkeypatch):
    monkeypatch.setattr(resources, "_torch_cuda_ok", lambda: (False, "driver too old"))
    device, note = resolve_device("auto", "torch")
    assert device == "cpu" and "driver too old" in note


def test_auto_uses_cuda_when_usable(monkeypatch):
    monkeypatch.setattr(resources, "_xgboost_cuda_ok", lambda: (True, "cuda:0"))
    assert resolve_device("auto", "xgboost") == ("cuda", "cuda:0")


def _booster_device(model_dir: Path) -> str:
    backend = XGBoostBackend.load(model_dir, threads=1)
    config = json.loads(backend.booster.save_config())
    return str(config["learner"]["generic_param"]["device"])


@needs_cuda
def test_xgboost_trains_on_gpu_and_saved_model_predicts_on_cpu(small_cfg):
    meta = train_xgboost(on(small_cfg, "cuda"), feature_set="embeddings", output_name="xgb_gpu")
    assert meta["training_device"] == "cuda"
    assert meta["hyperparameters"]["device"] == "cuda"
    model_dir = Path(meta["model_dir"])
    assert _booster_device(model_dir) == "cpu"  # portable artifact
    backend = XGBoostBackend.load(model_dir, threads=1)
    proba = backend.predict_proba(np.zeros((3, backend.n_features), np.float32))
    assert proba.shape == (3, 3) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)


@needs_cuda
def test_gpu_and_cpu_xgboost_reach_similar_validation_quality(small_cfg):
    cpu = train_xgboost(on(small_cfg, "cpu"), feature_set="embeddings", output_name="xgb_c")
    gpu = train_xgboost(on(small_cfg, "cuda"), feature_set="embeddings", output_name="xgb_g")
    # Not bit-identical across devices, but the same data and settings should agree closely.
    assert abs(cpu["validation_metrics"]["macro_f1"] - gpu["validation_metrics"]["macro_f1"]) < 0.15


@needs_cuda
def test_mlp_trains_on_gpu_and_checkpoint_loads_on_cpu(small_cfg):
    import torch

    from llm_router.models.complexity_classifiers import MLPBackend

    meta = train_mlp(on(small_cfg, "cuda"), feature_set="embeddings", output_name="mlp_gpu")
    assert meta["training_device"] == "cuda"
    assert meta["peak_gpu_memory_gib"] is not None and meta["peak_gpu_memory_gib"] > 0
    backend = MLPBackend.load(Path(meta["model_dir"]), standardize=meta["standardized"], threads=1)
    assert all(p.device.type == "cpu" for p in backend.model.parameters())
    n_features = meta["n_features"]
    proba = backend.predict_proba(np.zeros((2, n_features), np.float32))
    assert proba.shape == (2, 3)
    assert torch.cuda.memory_allocated() < 256 * 2**20  # nothing large left on the GPU
