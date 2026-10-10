"""Typed configuration for the complexity-classifier pipeline (configs/complexity.yaml).

Relative paths in the YAML file are resolved against the repository root, i.e. the parent of
the `configs/` directory that contains the file. Secrets are never part of this config; they
come from environment variables (see `llm_router.embeddings.fireworks_client`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LABELS = ("easy", "medium", "hard")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_Section):
    repo_id: str = "regolo/brick-complexity-extractor"
    revision: str = Field(min_length=7)  # pinned commit SHA
    # The card's `configs` section points at files that do not exist, so files are explicit.
    data_files: dict[str, str]
    text_field: str = "text"
    label_field: str = "label"
    confidence_field: str | None = "confidence"  # used only if the column exists
    license: str = "cc-by-nc-4.0"


class PreprocessingConfig(_Section):
    version: str = "1"
    duplicate_policy: Literal["keep_first", "keep_all"] = "keep_first"
    conflict_policy: Literal["drop", "majority"] = "drop"
    min_confidence: float | None = Field(default=None, ge=0, le=1)  # None: no filtering
    split_ratios: dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1}
    seed: int = 20261010

    @field_validator("split_ratios")
    @classmethod
    def _ratios(cls, value: dict[str, float]) -> dict[str, float]:
        if set(value) != {"train", "validation", "test"}:
            raise ValueError("split_ratios needs exactly train, validation, test")
        if any(r <= 0 for r in value.values()) or abs(sum(value.values()) - 1) > 1e-9:
            raise ValueError("split ratios must be positive and sum to 1")
        return value


class EmbeddingConfig(_Section):
    batch_size: int = Field(default=64, ge=1, le=2048)
    requests_per_minute: float = Field(default=60, gt=0)
    timeout_s: float = Field(default=60, gt=0)
    max_attempts: int = Field(default=6, ge=1, le=20)
    initial_backoff_s: float = Field(default=1.0, gt=0)
    max_backoff_s: float = Field(default=60.0, gt=0)
    dimensions: int | None = Field(default=None, gt=0)  # sent only if set (model must support it)
    # USD per million input tokens. Leave null unless taken from the provider's pricing page.
    price_per_million_tokens: float | None = Field(default=None, ge=0)
    connectivity_query: str = "Synthetic connectivity check: what is two plus two?"


class TrainingConfig(_Section):
    seed: int = 20261010
    logreg_c: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    svm_c: tuple[float, ...] = (0.01, 0.1, 1.0)
    class_weights: tuple[Literal["none", "balanced"], ...] = ("none", "balanced")
    max_iter: int = Field(default=3000, ge=100)
    selection_metric: Literal["macro_f1"] = "macro_f1"


ClassWeight = Literal["none", "balanced"]


class ResourcesConfig(_Section):
    """Compute budget shared by the XGBoost and MLP trainers.

    `device`: "cpu" (default, fully reproducible), "cuda" (GPU required; fails if unusable),
    or "auto" (GPU if usable, else CPU). Logistic regression / SVM always run on CPU.
    """

    threads: int = Field(default=2, ge=1, le=256)  # BLAS, OpenMP, XGBoost, and torch alike
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    seed: int = 42
    max_train_seconds: float | None = Field(default=3600, gt=0)  # per training job; None: off
    memory_budget_gb: float | None = Field(default=8.0, gt=0)  # stop training above this RSS
    # Class weighting is only evaluated when max/min training class count reaches this ratio.
    imbalance_threshold: float = Field(default=2.0, ge=1)


class XGBoostConfig(_Section):
    """Native-API equivalent of XGBClassifier(objective="multi:softprob", tree_method="hist")."""

    n_estimators: int = Field(default=200, ge=1)
    max_depth: int = Field(default=4, ge=1, le=12)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    subsample: float = Field(default=0.8, gt=0, le=1)
    colsample_bytree: float = Field(default=0.7, gt=0, le=1)
    min_child_weight: float = Field(default=1.0, ge=0)
    reg_lambda: float = Field(default=1.0, ge=0)
    max_bin: int = Field(default=256, ge=2, le=1024)
    early_stopping_rounds: int | None = Field(default=20, ge=1)  # on validation mlogloss
    class_weights: tuple[ClassWeight, ...] = ("none", "balanced")
    chunk_rows: int = Field(default=8192, ge=64)  # rows per DataIter batch (bounds copies)
    smoke_rows: int = Field(default=2000, ge=30)
    smoke_estimators: int = Field(default=10, ge=1)


class MLPConfig(_Section):
    hidden_dims: tuple[int, ...] = (256, 128)
    dropouts: tuple[float, ...] = (0.2, 0.1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-2, ge=0)
    batch_size: int = Field(default=128, ge=1)
    eval_batch_size: int = Field(default=2048, ge=1)
    max_epochs: int = Field(default=30, ge=1)
    patience: int = Field(default=5, ge=1)  # epochs without validation macro-F1 improvement
    class_weights: tuple[ClassWeight, ...] = ("none", "balanced")
    standardize: bool = True  # per-feature mean/std fitted on the training split only
    smoke_rows: int = Field(default=2000, ge=30)
    smoke_epochs: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _dropouts_match(self) -> MLPConfig:
        if len(self.dropouts) != len(self.hidden_dims):
            raise ValueError("mlp.dropouts needs one value per hidden layer")
        if any(not 0 <= p < 1 for p in self.dropouts):
            raise ValueError("dropout must be in [0, 1)")
        return self


ClassifierType = Literal["logistic_regression", "xgboost", "mlp"]


class ClassifierConfig(_Section):
    """Which trained classifier `ComplexityRouter.from_config` loads."""

    type: ClassifierType = "logistic_regression"
    model_dir: Path | None = None  # default: paths.models_dir / <type>


class PathsConfig(_Section):
    raw_dir: Path = Path("data/raw/brick-complexity-extractor")
    processed_dir: Path = Path("data/processed/brick-complexity")
    embeddings_dir: Path = Path("artifacts/embeddings")
    models_dir: Path = Path("artifacts/models")
    reports_dir: Path = Path("artifacts/reports")


class ComplexityConfig(_Section):
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    embeddings: EmbeddingConfig = EmbeddingConfig()
    training: TrainingConfig = TrainingConfig()
    resources: ResourcesConfig = ResourcesConfig()
    xgboost: XGBoostConfig = XGBoostConfig()
    mlp: MLPConfig = MLPConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    paths: PathsConfig = PathsConfig()

    @model_validator(mode="after")
    def _paths_absolute(self) -> ComplexityConfig:
        paths = list(self.paths.model_dump().values())
        if self.classifier.model_dir is not None:
            paths.append(self.classifier.model_dir)
        if not all(p.is_absolute() for p in paths):
            raise ValueError("paths must be absolute; use load_config() to resolve them")
        return self

    def classifier_dir(self) -> Path:
        return self.classifier.model_dir or self.paths.models_dir / self.classifier.type

    def preprocessing_hash(self) -> str:
        """Stable hash of everything that determines the processed dataset."""
        payload = {
            "dataset": self.dataset.model_dump(mode="json"),
            "preprocessing": self.preprocessing.model_dump(mode="json"),
        }
        return sha256_json(payload)[:16]


def load_config(path: str | Path = "configs/complexity.yaml") -> ComplexityConfig:
    path = Path(path).resolve()
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    paths = {k: str((root / v).resolve()) for k, v in (data.get("paths") or {}).items()}
    defaults = {k: str((root / v).resolve()) for k, v in PathsConfig().model_dump().items()}
    data["paths"] = {**defaults, **paths}
    classifier = data.get("classifier") or {}
    if classifier.get("model_dir"):
        model_dir = str((root / classifier["model_dir"]).resolve())
        data["classifier"] = {**classifier, "model_dir": model_dir}
    return ComplexityConfig.model_validate(data)


def sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
