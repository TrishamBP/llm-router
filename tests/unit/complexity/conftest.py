from __future__ import annotations

from pathlib import Path

import pytest

from llm_router.config import ComplexityConfig, MLPConfig, ResourcesConfig, XGBoostConfig
from llm_router.data import preprocess
from llm_router.embeddings.fake import FakeEmbeddingClient
from llm_router.embeddings.fireworks_client import connectivity_test
from llm_router.embeddings.generate import EmbeddingJob
from llm_router.training.smoke_complexity import build_workspace


@pytest.fixture
def workspace(tmp_path: Path) -> ComplexityConfig:
    """Synthetic raw snapshot + preprocessed splits in a temp dir (no network)."""
    cfg = build_workspace(tmp_path)
    cfg = cfg.model_copy(
        update={"embeddings": cfg.embeddings.model_copy(update={"batch_size": 16})}
    )
    preprocess.run(cfg)
    return cfg


@pytest.fixture
def fake_client() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def connected(workspace: ComplexityConfig, fake_client: FakeEmbeddingClient) -> ComplexityConfig:
    connectivity_test(
        fake_client, "synthetic check", workspace.paths.embeddings_dir / "connectivity"
    )
    fake_client.calls.clear()  # tests count only the calls made by the code under test
    return workspace


@pytest.fixture
def embedded(connected: ComplexityConfig, fake_client: FakeEmbeddingClient) -> ComplexityConfig:
    """Workspace with a complete (fake, 64-d) embedding cache for every split."""
    EmbeddingJob(connected, fake_client).run()
    fake_client.calls.clear()
    return connected


@pytest.fixture
def small_cfg(embedded: ComplexityConfig) -> ComplexityConfig:
    """Tiny, fast XGBoost / MLP settings; no time or memory budget (pytest RSS varies)."""
    return embedded.model_copy(
        update={
            "resources": ResourcesConfig(
                threads=1,
                seed=7,
                max_train_seconds=None,
                memory_budget_gb=None,
                imbalance_threshold=1.0,  # synthetic data is balanced; exercise both weightings
            ),
            "xgboost": XGBoostConfig(n_estimators=8, early_stopping_rounds=3, chunk_rows=64),
            "mlp": MLPConfig(max_epochs=3, patience=2, batch_size=16, eval_batch_size=32),
        }
    )
