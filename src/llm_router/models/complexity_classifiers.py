"""Serialisable XGBoost and MLP complexity classifiers with one `predict_proba` interface.

Probability columns are always in `LABELS` order (easy, medium, hard); the saved metadata
records that order and loading refuses anything else. torch is imported only for the MLP.

Artifacts per model directory:
  xgboost: model.ubj (XGBoost native format, trimmed to the best iteration)
  mlp:     model.pt (state_dict + architecture, loaded with weights_only=True)
           standardizer.npz (training-split mean/std; absent if standardisation is off)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from llm_router.config import LABELS
from llm_router.features.complexity_features import Standardizer

XGBOOST_MODEL_FILE = "model.ubj"
MLP_MODEL_FILE = "model.pt"
STANDARDIZER_FILE = "standardizer.npz"


class ClassifierArtifactError(Exception):
    """Missing, corrupt, or incompatible classifier artifact."""


def _check_proba(proba: np.ndarray, n_rows: int) -> np.ndarray:
    """(n, 3) float64 rows that sum to 1 (float32 outputs are off by ~1e-7)."""
    proba = np.asarray(proba, dtype=np.float64).reshape(n_rows, len(LABELS))
    out: np.ndarray = proba / proba.sum(axis=1, keepdims=True)
    return out


class XGBoostBackend:
    name = "xgboost"

    def __init__(self, booster: Any, *, threads: int = 1) -> None:
        self.booster = booster
        # Inference always runs on CPU, even for GPU-trained boosters, so artifacts are portable.
        self.booster.set_param({"nthread": threads, "device": "cpu"})
        self.n_features = int(booster.num_features())

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features:
            raise ClassifierArtifactError(f"model expects {self.n_features} features")
        return _check_proba(self.booster.inplace_predict(X), X.shape[0])

    def save(self, directory: Path) -> None:
        self.booster.save_model(directory / XGBOOST_MODEL_FILE)

    @classmethod
    def load(cls, directory: Path, *, threads: int = 1) -> XGBoostBackend:
        import xgboost as xgb

        path = directory / XGBOOST_MODEL_FILE
        if not path.exists():
            raise ClassifierArtifactError(f"{path} not found")
        booster = xgb.Booster()
        try:
            booster.load_model(path)
        except xgb.core.XGBoostError as exc:
            raise ClassifierArtifactError(f"cannot load {path}: {exc}") from None
        return cls(booster, threads=threads)


class MLPBackend:
    name = "mlp"

    def __init__(
        self,
        model: Any,
        architecture: dict[str, Any],
        standardizer: Standardizer | None,
        *,
        batch_size: int = 4096,
    ) -> None:
        self.model = model.to("cpu").eval()
        self.architecture = architecture
        self.standardizer = standardizer
        self.batch_size = batch_size
        self.n_features = int(architecture["input_dim"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != self.n_features:
            raise ClassifierArtifactError(f"model expects {self.n_features} features")
        out = np.empty((X.shape[0], len(LABELS)), dtype=np.float64)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                block = X[start : start + self.batch_size]
                if self.standardizer is not None:
                    block = self.standardizer.transform(block)
                logits = self.model(torch.from_numpy(np.ascontiguousarray(block)))
                out[start : start + len(block)] = torch.softmax(logits.double(), dim=1).numpy()
        return _check_proba(out, X.shape[0])

    def save(self, directory: Path) -> None:
        import torch

        state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        torch.save(
            {"state_dict": state, "architecture": self.architecture}, directory / MLP_MODEL_FILE
        )
        if self.standardizer is not None:
            self.standardizer.save(directory / STANDARDIZER_FILE)

    @classmethod
    def load(cls, directory: Path, *, standardize: bool, threads: int = 1) -> MLPBackend:
        import torch

        from llm_router.models.complexity_mlp import build_mlp

        torch.set_num_threads(threads)
        path = directory / MLP_MODEL_FILE
        if not path.exists():
            raise ClassifierArtifactError(f"{path} not found")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            arch = checkpoint["architecture"]
            model = build_mlp(
                arch["input_dim"], arch["hidden_dims"], arch["dropouts"], arch["n_classes"]
            )
            model.load_state_dict(checkpoint["state_dict"])
        except Exception as exc:  # torch raises many types for truncated/corrupt files
            raise ClassifierArtifactError(
                f"cannot load {path}: {type(exc).__name__}: {exc}"
            ) from None
        if arch["n_classes"] != len(LABELS):
            raise ClassifierArtifactError("checkpoint does not have one output per label")
        standardizer = None
        if standardize:
            std_path = directory / STANDARDIZER_FILE
            if not std_path.exists():
                raise ClassifierArtifactError(f"{std_path} not found")
            try:
                standardizer = Standardizer.load(std_path)
            except Exception as exc:
                raise ClassifierArtifactError(f"cannot load {std_path}: {exc}") from None
            if standardizer.mean.shape[0] != arch["input_dim"]:
                raise ClassifierArtifactError("standardizer and model input sizes differ")
        return cls(model, dict(arch), standardizer)
