"""Router dataset: schema/label validation, cleaning, grouped splits, embeddings, loader.

Synthetic data and the offline fake embedding client only: no network, no API charges.
"""

from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest

from llm_router.config import EmbeddingConfig
from llm_router.data import prepare_router_dataset as prep
from llm_router.data.inspect_router_dataset import inspect_records
from llm_router.data.router_dataset import (
    RouterDatasetError,
    build_label_mapping,
    load_router_config,
    load_router_dataset,
    parse_card_label_names,
)
from llm_router.data.router_smoke import CARD, build_workspace, run_smoke
from llm_router.embeddings import generate_router_embeddings as gen
from llm_router.embeddings.fake import FakeEmbeddingClient
from llm_router.embeddings.fireworks_client import (
    FireworksEmbeddingClient,
    FireworksSettings,
    InvalidEmbeddingResponseError,
    ModelNotAvailableError,
    parse_embedding_response,
)
from llm_router.embeddings.generate import EmbeddingJobError

NAMES = {0: "small_llm", 1: "large_llm"}


def row(id_, prompt, label):
    return {"id": id_, "prompt": prompt, "label": label}


@pytest.fixture
def cfg(tmp_path):
    return build_workspace(tmp_path)


def prepared_cfg(cfg):
    prep.run(cfg, save_hf=False)
    return cfg


@pytest.fixture
def embedded(cfg):
    prepared_cfg(cfg)
    client = FakeEmbeddingClient()
    gen.run_connectivity(cfg, client)
    gen.make_job(cfg, client).run()
    return cfg


# ------------------------------------------------------------------ schema & labels


def test_shipped_config_uses_seed_42_and_80_10_10(repo_root):
    cfg = load_router_config(repo_root / "configs" / "router_dataset.yaml")
    assert cfg.preparation.seed == 42
    assert cfg.preparation.split_ratios == {"train": 0.8, "validation": 0.1, "test": 0.1}
    assert cfg.dataset.repo_id == "DevQuasar/llm_router_dataset-synth"


def test_card_label_names_are_parsed_and_required():
    assert parse_card_label_names(CARD) == NAMES
    with pytest.raises(RouterDatasetError, match="no YAML front matter"):
        parse_card_label_names("just text")
    with pytest.raises(RouterDatasetError, match="no class-label names"):
        parse_card_label_names("---\ndataset_info: {features: []}\n---\n")


def test_label_mapping_is_stable_and_validated():
    mapping = build_label_mapping(NAMES, {0, 1})
    assert mapping["label_to_id"] == {"small_llm": 0, "large_llm": 1}
    assert mapping["id_to_label"] == {"0": "small_llm", "1": "large_llm"}
    assert mapping["mapping_hash"] == build_label_mapping(NAMES, {1, 0})["mapping_hash"]
    with pytest.raises(RouterDatasetError, match="not declared"):
        build_label_mapping(NAMES, {0, 2})
    with pytest.raises(RouterDatasetError, match="must be integers"):
        build_label_mapping(NAMES, {0, True})
    assert build_label_mapping(NAMES, {1})["declared_but_unobserved"] == [0]


def test_inspection_discovers_schema_duplicates_and_conflicts():
    records = {
        "train": [
            row("a", "Same prompt", 0),
            row("a", "Other", 0),
            row("b", "same prompt!", 0),
            row("c", "Clash", 0),
            row("d", " ", 1),
        ],
        "test": [row("e", "clash", 1), row("f", "Same prompt", 0)],
    }
    report = inspect_records(records, NAMES)
    assert report["schema_valid"] is True
    assert report["total_rows"] == 7
    assert report["splits"]["train"]["empty_prompt"] == 1
    assert report["exact_duplicate_prompts"]["groups_spanning_official_splits"] == 1
    assert report["normalized_duplicate_prompts"]["groups"] == 2
    assert report["conflicting_labels"]["groups"] == 1
    assert report["ids"]["ids_in_multiple_official_splits"] == {"test&train": 0}


def test_inspection_flags_missing_fields_and_bad_labels():
    report = inspect_records({"train": [{"id": "a", "text": "x", "label": "big"}]}, NAMES)
    assert report["schema_valid"] is False
    assert any("prompt" in p for p in report["schema_problems"])
    assert any("integers" in p for p in report["schema_problems"])


def test_inspection_detects_ids_with_multiple_labels():
    report = inspect_records({"train": [row("a", "x", 0), row("a", "y", 1)]}, NAMES)
    assert report["ids"]["ids_with_multiple_labels"] == 1


# ------------------------------------------------------------------ cleaning & splits


def test_cleaning_policies_are_applied_and_recorded(cfg):
    records = {
        "train": [
            row("a", "What is X?", 0),
            row("b", "what is x", 0),
            row("c", "Clash?", 0),
            row("d", "clash", 1),
            row("e", "", 0),
            row("f", "ok", 5),
        ]
        + [row(f"g{i}", f"large prompt {i}", 1) for i in range(30)],
        "test": [row("h", "What is X?", 0)]
        + [row(f"s{i}", f"small prompt {i}", 0) for i in range(30)],
    }
    out = prep.prepare_records(records, NAMES, cfg)
    reasons = Counter(r.reason for r in out.removed)
    assert reasons == {
        "duplicate_prompt": 2,
        "conflicting_labels": 2,
        "empty_prompt": 1,
        "invalid_label": 1,
    }
    kept = {e.example_id: e for rows in out.splits.values() for e in rows}
    assert kept["train-000000"].duplicate_ids == ["train-000001", "test-000000"]
    assert all(e.label in ("small_llm", "large_llm") for e in kept.values())
    assert all(e.label_id == {"small_llm": 0, "large_llm": 1}[e.label] for e in kept.values())


def test_groups_keep_source_ids_together_and_never_cross_splits(cfg):
    records = {
        "train": [row(f"id{g}", f"prompt {g}-{j}", g % 2) for g in range(200) for j in range(5)]
    }
    out = prep.prepare_records(records, NAMES, cfg)
    split_of_id: dict[str, str] = {}
    for name, rows in out.splits.items():
        for e in rows:
            assert split_of_id.setdefault(e.source_id, name) == name
    assert out.metadata["split_isolation"]["verified"]
    assert out.metadata["split_isolation"]["overlaps"] == {
        "example_id": 0,
        "group_id": 0,
        "prompt_key": 0,
        "source_id": 0,
    }
    for name, ratio in {"train": 0.8, "validation": 0.1, "test": 0.1}.items():
        assert abs(out.metadata["splits"][name]["proportion"] - ratio) < 0.02
    train_props = out.metadata["splits"]["train"]["label_proportions"]
    test_props = out.metadata["splits"]["test"]["label_proportions"]
    assert abs(train_props["large_llm"] - test_props["large_llm"]) < 0.05


def test_stock_prompts_do_not_chain_ids_into_one_group(cfg):
    stock = "What is the capital of France?"
    records = {"train": [row(f"id{g}", p, 0) for g in range(50) for p in (stock, f"own {g}")]}
    out = prep.prepare_records(records, NAMES, cfg)
    assert out.metadata["grouping"]["equals_source_id_grouping"] is True
    assert out.metadata["grouping"]["group_size"]["max"] <= 2
    assert out.metadata["removed"] == {"duplicate_prompt": 49}


def test_preparation_is_deterministic(cfg, tmp_path):
    a = prep.run(cfg, save_hf=False).metadata["processed_fingerprint"]
    b = prep.run(cfg, save_hf=False).metadata["processed_fingerprint"]
    assert a == b


def test_isolation_check_catches_a_leak(cfg):
    records = {"train": [row(f"id{g}", f"p{g}", g % 2) for g in range(60)]}
    out = prep.prepare_records(records, NAMES, cfg)
    out.splits["test"].append(out.splits["train"][0])
    assert prep.verify_isolation(out.splits)["verified"] is False


def test_outputs_include_mapping_metadata_and_hf_dataset(cfg):
    prep.run(cfg, save_hf=True)
    out = cfg.paths.processed_dir
    for name in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "label_mapping.json",
        "split_metadata.json",
        "removed.jsonl",
    ):
        assert (out / name).exists(), name
    from datasets import load_from_disk

    hf = load_from_disk(str(out / "hf_dataset"))
    assert set(hf) == {"train", "validation", "test"}
    assert hf["train"].features["label_id"].names == ["small_llm", "large_llm"]
    record = json.loads((out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert {"source_id", "prompt", "label", "source_split", "source_row_index", "group_id"} <= set(
        record
    )


# ------------------------------------------------------------------ embeddings


def test_connectivity_is_required_before_embedding(cfg):
    prepared_cfg(cfg)
    with pytest.raises(EmbeddingJobError, match="connectivity"):
        gen.make_job(cfg, FakeEmbeddingClient())


def test_embedding_rows_align_with_prompts(embedded):
    data = load_router_dataset(config=embedded)
    client = FakeEmbeddingClient()
    for split, X in (("train", data.X_train), ("test", data.X_test)):
        with (embedded.paths.processed_dir / f"{split}.jsonl").open(encoding="utf-8") as fh:
            prompts = [json.loads(line)["prompt"] for line in fh]
        np.testing.assert_allclose(np.asarray(X), client.embed(prompts).vectors)
    assert data.X_train.dtype == np.float32 and isinstance(data.X_train, np.memmap)


def test_resume_and_cache_reuse_make_no_repeat_requests(cfg):
    prepared_cfg(cfg)
    client = FakeEmbeddingClient()
    gen.run_connectivity(cfg, client)
    client.calls.clear()
    gen.make_job(cfg, client).run(max_batches=2)
    first = [t for c in client.calls for t in c]
    client.calls.clear()
    gen.make_job(cfg, client).run()
    second = [t for c in client.calls for t in c]
    assert not set(first) & set(second)
    client.calls.clear()
    assert gen.make_job(cfg, client).run()["complete"] and client.calls == []


def test_estimate_makes_no_api_calls(embedded):
    from llm_router.embeddings.generate import _PlanOnlyClient

    job = gen.make_job(embedded, _PlanOnlyClient("fake/hashed-bow-64"), require_connectivity=False)
    summary = gen.estimate(embedded, job)
    assert summary["remaining_requests"] == 0 and summary["rows"]["train"] > 0


def test_embedding_response_validation():
    good = SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[0.1, 0.2])], usage=None)
    assert parse_embedding_response(good, 1)[0].shape == (1, 2)
    bad = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[float("inf"), 0.0])], usage=None
    )
    with pytest.raises(InvalidEmbeddingResponseError):
        parse_embedding_response(bad, 1)


def test_missing_credentials_stop_the_command(monkeypatch, tmp_path):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_EMBEDDING_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)  # no .env here
    with pytest.raises(SystemExit, match="FIREWORKS_API_KEY"):
        gen.main(["--test", "--config", str(_shipped_config())])


def _shipped_config():
    from pathlib import Path

    return Path(__file__).resolve().parents[3] / "configs" / "router_dataset.yaml"


def test_invalid_model_fails_fast(cfg):
    request = httpx.Request("POST", "https://api.fireworks.ai/inference/v1/embeddings")
    error = openai.NotFoundError(
        "model not found", response=httpx.Response(404, request=request), body=None
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise error

    client = FireworksEmbeddingClient(
        FireworksSettings(api_key="fw_fake", model="bad/model"),
        EmbeddingConfig(),
        openai_client=SimpleNamespace(embeddings=SimpleNamespace(create=create)),
        sleep=lambda s: None,
    )
    with pytest.raises(ModelNotAvailableError, match="bad/model"):
        gen.run_connectivity(cfg, client)
    assert len(calls) == 1  # not retried


# ------------------------------------------------------------------ loader


def test_loader_returns_aligned_model_ready_arrays(embedded):
    data = load_router_dataset(config=embedded)
    assert data.label_names == ("small_llm", "large_llm") and data.n_classes == 2
    for X, y in (
        (data.X_train, data.y_train),
        (data.X_validation, data.y_validation),
        (data.X_test, data.y_test),
    ):
        assert X.shape[0] == y.shape[0] and X.shape[1] == 64
        assert y.dtype == np.int64 and set(np.unique(y)) <= {0, 1}
    ids = [set(v) for v in data.example_ids.values()]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


@pytest.mark.parametrize("tamper", ["label_mapping", "fingerprint", "row_order", "label"])
def test_loader_rejects_inconsistent_artifacts(embedded, tamper):
    processed = embedded.paths.processed_dir
    emb_dir = load_router_dataset(config=embedded).embeddings_dir
    if tamper == "label_mapping":
        path = processed / "label_mapping.json"
        m = json.loads(path.read_text())
        m["mapping_hash"] = "deadbeef"
        path.write_text(json.dumps(m))
    elif tamper == "fingerprint":
        path = emb_dir / "metadata.json"
        m = json.loads(path.read_text())
        m["processed_fingerprint"] = "other"
        path.write_text(json.dumps(m))
    elif tamper == "row_order":
        path = emb_dir / "train.row_mapping.jsonl"
        lines = path.read_text().splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        path.write_text("\n".join(lines) + "\n")
    else:
        path = processed / "test.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        r = json.loads(lines[0])
        r["label"] = "large_llm" if r["label"] == "small_llm" else "small_llm"
        lines[0] = json.dumps(r)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(RouterDatasetError):
        load_router_dataset(config=embedded)


def test_smoke_runs_without_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    result = run_smoke(tmp_path)
    assert result["isolation"] and result["resumed_complete"] and not result["interrupted_complete"]
    assert result["loader_rows"] == result["splits"]
