"""Merged training set: label mapping, cross-dataset dedup, unchanged eval splits, alignment.

Synthetic complexity + router workspaces with the same offline fake embedding model.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from llm_router.config import PathsConfig
from llm_router.data import prepare_router_dataset as prep
from llm_router.data import preprocess
from llm_router.data.build_merged_dataset import LABEL_MAP, build
from llm_router.data.datasets import load_embedded_split
from llm_router.data.leakage import verify_embedding_artifacts
from llm_router.data.preprocess import load_processed
from llm_router.data.router_smoke import build_workspace as build_router_workspace
from llm_router.embeddings import generate_router_embeddings as router_gen
from llm_router.embeddings.fake import FakeEmbeddingClient
from llm_router.embeddings.fireworks_client import connectivity_test
from llm_router.embeddings.generate import EmbeddingJob
from llm_router.training.smoke_complexity import build_workspace as build_complexity_workspace


@pytest.fixture
def sources(tmp_path):
    client = FakeEmbeddingClient()
    base = build_complexity_workspace(tmp_path / "complexity")
    base = base.model_copy(
        update={"embeddings": base.embeddings.model_copy(update={"batch_size": 16})}
    )
    preprocess.run(base)
    connectivity_test(client, "check", base.paths.embeddings_dir / "connectivity")
    EmbeddingJob(base, client).run()
    router = build_router_workspace(tmp_path / "router")
    prep.run(router, save_hf=False)
    router_gen.run_connectivity(router, client)
    router_gen.make_job(router, client).run()
    root = tmp_path / "merged"
    merged = base.model_copy(
        update={
            "paths": PathsConfig(
                raw_dir=base.paths.raw_dir,
                processed_dir=root / "processed",
                embeddings_dir=root / "embeddings",
                models_dir=root / "models",
                reports_dir=root / "reports",
            )
        }
    )
    return base, router, merged


def test_merged_dataset_maps_labels_and_keeps_eval_splits(sources):
    base, router, merged = sources
    summary = build(merged, complexity_config=base, router_config=router)
    assert summary["verified"] is True
    base_examples, _ = load_processed(base.paths.processed_dir)
    merged_examples, manifest = load_processed(merged.paths.processed_dir)
    for split in ("validation", "test"):  # evaluation splits are byte-for-byte the same rows
        assert [e["example_id"] for e in merged_examples[split]] == [
            e["example_id"] for e in base_examples[split]
        ]
    router_rows = [e for e in merged_examples["train"] if e["example_id"].startswith("router-")]
    assert router_rows and all(e["label"] == LABEL_MAP[e["original_label"]] for e in router_rows)
    assert {e["label"] for e in router_rows} <= {"medium", "hard"}  # never easy
    with (router.paths.processed_dir / "validation.jsonl").open(encoding="utf-8") as fh:
        router_val_ids = {f"router-{json.loads(line)['example_id']}" for line in fh}
    assert not router_val_ids & {e["example_id"] for e in merged_examples["train"]}
    assert manifest["label_map"] == {"large_llm": "hard", "small_llm": "medium"}


def test_merged_rows_align_with_source_vectors(sources):
    base, router, merged = sources
    summary = build(merged, complexity_config=base, router_config=router)
    from pathlib import Path

    emb = Path(summary["embeddings_dir"])
    train = load_embedded_split(emb, "train")
    base_train = load_embedded_split(
        base.paths.embeddings_dir
        / json.loads((base.paths.embeddings_dir / "latest.json").read_text())["cache_key"],
        "train",
    )
    n = len(base_train.example_ids)
    np.testing.assert_array_equal(train.X[:n], base_train.X)
    merged_examples, _ = load_processed(merged.paths.processed_dir)
    router_texts = [e["query"] for e in merged_examples["train"][n:]]
    np.testing.assert_allclose(train.X[n:], FakeEmbeddingClient().embed(router_texts).vectors)
    assert verify_embedding_artifacts(merged.paths.processed_dir, emb)["example_ids_disjoint"]


def test_router_rows_duplicating_complexity_prompts_are_dropped(tmp_path):
    client = FakeEmbeddingClient()
    base = build_complexity_workspace(tmp_path / "complexity")
    preprocess.run(base)
    connectivity_test(client, "check", base.paths.embeddings_dir / "connectivity")
    EmbeddingJob(base, client).run()
    base_examples, _ = load_processed(base.paths.processed_dir)
    test_prompt = base_examples["test"][0]["query"]
    # Plant the same prompt (different case/spacing) under its own router ID.
    router = build_router_workspace(
        tmp_path / "router",
        extra_train_rows=[{"id": "planted", "prompt": test_prompt.upper() + "  ", "label": 1}],
    )
    prep.run(router, save_hf=False)
    router_gen.run_connectivity(router, client)
    router_gen.make_job(router, client).run()
    merged = base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={
                    "processed_dir": tmp_path / "m" / "processed",
                    "embeddings_dir": tmp_path / "m" / "embeddings",
                    "reports_dir": tmp_path / "m" / "reports",
                }
            )
        }
    )
    planted = [
        json.loads(line)
        for split in ("train", "validation", "test")
        for line in (router.paths.processed_dir / f"{split}.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line)["source_id"] == "planted"
    ]
    summary = build(merged, complexity_config=base, router_config=router)
    dropped = json.loads((merged.paths.processed_dir / "dropped_router_rows.json").read_text())
    if planted[0]["split"] == "train":  # only router *train* rows are merged
        assert dropped[planted[0]["example_id"]].startswith("same normalised prompt")
    merged_examples, _ = load_processed(merged.paths.processed_dir)
    keys = {e["group"] for e in merged_examples["train"]}
    assert not keys & {e["group"] for e in merged_examples["test"]}
    assert summary["router_train_kept"] == summary["router_train_rows"] - len(dropped)


def test_incompatible_embedding_models_are_refused(sources):
    base, router, merged = sources
    meta_path = next(router.paths.embeddings_dir.glob("*/metadata.json"))
    meta = json.loads(meta_path.read_text())
    meta["embedding_model"] = "other/model"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="incompatible"):
        build(merged, complexity_config=base, router_config=router)


def test_merged_train_has_no_easy_rows_from_router(sources):
    base, router, merged = sources
    build(merged, complexity_config=base, router_config=router)
    merged_examples, _ = load_processed(merged.paths.processed_dir)
    sources_count = Counter(e.get("source_dataset") for e in merged_examples["train"])
    assert set(sources_count) == {
        "regolo/brick-complexity-extractor",
        "DevQuasar/llm_router_dataset-synth",
    }
