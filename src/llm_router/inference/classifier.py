"""Complexity prediction for incoming queries.

    from llm_router.inference.classifier import ComplexityRouter

    router = ComplexityRouter.from_pretrained("artifacts/models/mlp")      # or xgboost, ...
    result = router.predict("Design a distributed scheduler that minimizes GPU idle time.")
    router = ComplexityRouter.from_config("configs/complexity.yaml")       # classifier.type

Supported classifiers: `logistic_regression` / `linear_svm` (scikit-learn, joblib),
`xgboost`, and `mlp` (needs torch). The type is read from the model directory's metadata.

Online mode embeds the query with the Fireworks model the classifier was trained on (read from
FIREWORKS_EMBEDDING_MODEL and required to match the saved metadata). Offline mode
(`offline=True`) never calls an API and only accepts precomputed embeddings (plus the query
texts, for models that use engineered features). Every input goes through the same
normalisation and feature construction as training, as recorded in `feature_schema.json`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from llm_router.config import LABELS, ComplexityConfig
from llm_router.data.splitting import normalize_text
from llm_router.embeddings.fireworks_client import EmbeddingClient

SKLEARN_CLASSIFIERS = ("logistic_regression", "linear_svm")
CLASSIFIERS = (*SKLEARN_CLASSIFIERS, "xgboost", "mlp")


class ComplexityRouterError(Exception):
    pass


class EmbeddingModelMismatchError(ComplexityRouterError):
    pass


class OfflineModeError(ComplexityRouterError):
    pass


class ArtifactError(ComplexityRouterError):
    """Missing, corrupt, or incompatible model artifacts."""


@dataclass(frozen=True)
class ComplexityPrediction:
    label: str
    probabilities: dict[str, float] | None  # None for classifiers without probabilities
    classifier: str
    embedding_model: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _SklearnBackend:
    """Existing scikit-learn pipelines; classes follow the saved LabelEncoder."""

    def __init__(self, model: Any, classes: Sequence[str]) -> None:
        self.model = model
        self.classes = list(classes)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        indices = np.asarray(self.model.predict(X))
        proba = self.model.predict_proba(X) if hasattr(self.model, "predict_proba") else None
        return indices, proba


class _ProbaBackend:
    """XGBoost / MLP backends: probability columns are in LABELS order."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.classes = list(LABELS)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        proba = self.backend.predict_proba(X)
        return proba.argmax(axis=1), proba


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactError(f"{path} not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"{path} is corrupt: {exc}") from None
    if not isinstance(data, dict):
        raise ArtifactError(f"{path} is not a JSON object")
    return data


def _embeddings_only_schema(metadata: dict[str, Any]) -> dict[str, Any]:
    from llm_router.features.complexity_features import build_schema

    return build_schema(
        "embeddings",
        {
            "dimension": metadata["embedding_dimension"],
            "embedding_model": metadata["embedding_model"],
            "cache_key": metadata.get("embedding_cache_key"),
        },
    )


class ComplexityRouter:
    def __init__(
        self,
        model: Any,
        classes: Sequence[str],
        metadata: dict[str, Any],
        embedding_client: EmbeddingClient | None,
        *,
        batch_size: int = 64,
        feature_schema: dict[str, Any] | None = None,
    ) -> None:
        self._backend: Any = (
            model
            if isinstance(model, (_SklearnBackend, _ProbaBackend))
            else _SklearnBackend(model, classes)
        )
        self._classes = list(self._backend.classes)
        self.metadata = metadata
        self.embedding_model: str = metadata["embedding_model"]
        self.dimension: int = int(metadata["embedding_dimension"])
        self.classifier: str = metadata["classifier"]
        self.feature_schema = feature_schema or _embeddings_only_schema(metadata)
        self.feature_set: str = self.feature_schema["feature_set"]
        self._client = embedding_client
        self._batch_size = batch_size
        if self.feature_schema["embedding_model"] != self.embedding_model:
            raise ArtifactError("feature schema and metadata name different embedding models")
        if int(self.feature_schema["embedding_dimension"]) != self.dimension:
            raise ArtifactError("feature schema and metadata disagree on embedding dimension")
        if (
            embedding_client is not None
            and self.needs_embeddings
            and embedding_client.model != self.embedding_model
        ):
            raise EmbeddingModelMismatchError(
                f"classifier was trained on embeddings from '{self.embedding_model}', but the "
                f"configured embedding model is '{embedding_client.model}'"
            )

    @property
    def needs_embeddings(self) -> bool:
        return self.feature_set != "engineered"

    @property
    def needs_texts(self) -> bool:
        return self.feature_set != "embeddings"

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        embedding_client: EmbeddingClient | None = None,
        offline: bool = False,
        batch_size: int = 64,
        threads: int = 1,
        expected_classifier: str | None = None,
        embedding_config: Any = None,
    ) -> ComplexityRouter:
        path = Path(path)
        if not path.is_dir():
            raise ArtifactError(f"model directory {path} not found")
        metadata = _read_json(path / "metadata.json")
        kind = metadata.get("classifier")
        if kind not in CLASSIFIERS:
            raise ArtifactError(f"{path}: unknown classifier type {kind!r}")
        if expected_classifier is not None and kind != expected_classifier:
            raise ArtifactError(
                f"{path} holds a '{kind}' model but '{expected_classifier}' was configured"
            )
        for key in ("embedding_model", "embedding_dimension", "classes"):
            if key not in metadata:
                raise ArtifactError(f"{path}/metadata.json lacks '{key}'")
        backend, schema = cls._load_backend(path, kind, metadata, threads)
        router_needs_client = schema is None or schema["feature_set"] != "engineered"
        if embedding_client is None and not offline and router_needs_client:
            from llm_router.config import EmbeddingConfig
            from llm_router.embeddings.fireworks_client import (
                FireworksEmbeddingClient,
                FireworksSettings,
            )

            embedding_client = FireworksEmbeddingClient(
                FireworksSettings.from_env(), embedding_config or EmbeddingConfig()
            )
        return cls(
            backend,
            backend.classes,
            metadata,
            embedding_client,
            batch_size=batch_size,
            feature_schema=schema,
        )

    @staticmethod
    def _load_backend(
        path: Path, kind: str, metadata: dict[str, Any], threads: int
    ) -> tuple[Any, dict[str, Any] | None]:
        if kind in SKLEARN_CLASSIFIERS:
            try:
                model = joblib.load(path / "model.joblib")
                encoder = joblib.load(path / "label_encoder.joblib")
            except FileNotFoundError as exc:
                raise ArtifactError(f"{path}: {exc}") from None
            except Exception as exc:  # pickle/joblib raise many types on corrupt files
                raise ArtifactError(f"{path}: cannot load model: {type(exc).__name__}") from None
            if list(encoder.classes_) != metadata["classes"]:
                raise ComplexityRouterError("label encoder does not match saved metadata")
            return _SklearnBackend(model, encoder.classes_), None

        from llm_router.features.complexity_features import FeatureSchemaError, check_schema
        from llm_router.models.complexity_classifiers import (
            ClassifierArtifactError,
            MLPBackend,
            XGBoostBackend,
        )

        if metadata["classes"] != list(LABELS) or metadata.get("probability_columns") != list(
            LABELS
        ):
            raise ArtifactError(f"{path}: probability columns are not in {list(LABELS)} order")
        schema = _read_json(path / "feature_schema.json")
        try:
            check_schema(schema)
            if kind == "xgboost":
                backend: Any = XGBoostBackend.load(path, threads=threads)
            else:
                backend = MLPBackend.load(
                    path, standardize=bool(metadata.get("standardized", True)), threads=threads
                )
        except (FeatureSchemaError, ClassifierArtifactError) as exc:
            raise ArtifactError(f"{path}: {exc}") from None
        if backend.n_features != schema["n_features"]:
            raise ArtifactError(
                f"{path}: model expects {backend.n_features} features, schema has "
                f"{schema['n_features']}"
            )
        return _ProbaBackend(backend), schema

    @classmethod
    def from_config(
        cls,
        config: str | Path | ComplexityConfig = "configs/complexity.yaml",
        **kwargs: Any,
    ) -> ComplexityRouter:
        """Load the classifier named by `classifier.type` (and optional `classifier.model_dir`)."""
        from llm_router.config import load_config

        cfg = config if isinstance(config, ComplexityConfig) else load_config(config)
        kwargs.setdefault("threads", cfg.resources.threads)
        kwargs.setdefault("embedding_config", cfg.embeddings)
        return cls.from_pretrained(
            cfg.classifier_dir(), expected_classifier=cfg.classifier.type, **kwargs
        )

    # ------------------------------------------------------------------ prediction
    def predict(self, query: str) -> ComplexityPrediction:
        return self.predict_batch([query])[0]

    def predict_batch(self, queries: Sequence[str]) -> list[ComplexityPrediction]:
        cleaned = [normalize_text(q) for q in queries]
        if not all(cleaned):
            raise ValueError("queries must be non-empty")
        if not self.needs_embeddings:
            return self.predict_embeddings(None, texts=cleaned)
        if self._client is None:
            raise OfflineModeError("offline mode: use predict_embeddings() with cached vectors")
        results: list[ComplexityPrediction] = []
        for start in range(0, len(cleaned), self._batch_size):
            chunk = cleaned[start : start + self._batch_size]
            vectors = self._client.embed(chunk).vectors
            results.extend(self.predict_embeddings(vectors, texts=chunk))
        return results

    def predict_arrays(
        self, vectors: np.ndarray | None, texts: Sequence[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """(class indices, probabilities or None) with columns in `self.classes` order.

        `texts` must already be normalised (as in the processed dataset or `predict_batch`).
        """
        if self.needs_embeddings:
            if vectors is None:
                raise ComplexityRouterError("this classifier needs embeddings")
            vectors = np.asarray(vectors, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors[None, :]
            if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
                raise ComplexityRouterError(
                    f"embedding dimension {vectors.shape[-1]} != classifier dimension "
                    f"{self.dimension}"
                )
            if not np.isfinite(vectors).all():
                raise ComplexityRouterError("embeddings contain non-finite values")
        if self.needs_texts and texts is None:
            raise ComplexityRouterError(
                f"the '{self.feature_set}' feature set needs the query texts as well"
            )
        from llm_router.features.complexity_features import FeatureSchemaError, build_features

        try:
            X = build_features(self.feature_schema, vectors, texts)
        except FeatureSchemaError as exc:
            raise ComplexityRouterError(str(exc)) from None
        result: tuple[np.ndarray, np.ndarray | None] = self._backend.predict(X)
        return result

    def predict_embeddings(
        self, vectors: np.ndarray | None, texts: Sequence[str] | None = None
    ) -> list[ComplexityPrediction]:
        indices, proba = self.predict_arrays(vectors, texts)
        order = [self._classes.index(label) for label in LABELS if label in self._classes]
        return [
            ComplexityPrediction(
                label=self._classes[int(index)],
                probabilities=(
                    {self._classes[j]: float(proba[row, j]) for j in order}
                    if proba is not None
                    else None
                ),
                classifier=self.classifier,
                embedding_model=self.embedding_model,
            )
            for row, index in enumerate(indices)
        ]

    @property
    def classes(self) -> list[str]:
        return list(self._classes)
