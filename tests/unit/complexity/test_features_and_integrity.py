"""Engineered features, feature/label alignment, split isolation, and artifact validation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from llm_router.data.datasets import AlignmentError
from llm_router.data.leakage import near_duplicate_audit, verify_embedding_artifacts
from llm_router.embeddings.generate import EmbeddingJob, resolve_embeddings_dir
from llm_router.features import text_features
from llm_router.features.complexity_features import (
    FeatureMatrix,
    Standardizer,
    decode_labels,
    encode_labels,
    load_split_features,
)

SAMPLE = (
    "Write a function `f(x)` that squares x.\n"
    "```python\ndef f(x):\n    return x ** 2;\n```\n"
    "Is it O(1)? Use $\\frac{a}{b}$.\n"
    "| a | b |\n|---|---|\n"
    '{"key": 1}'
)


def _rows(cfg, split):
    with (cfg.paths.processed_dir / f"{split}.jsonl").open(encoding="utf-8") as fh:
        return {r["example_id"]: r for r in map(json.loads, fh)}


# ---------------------------------------------------------------- engineered features
def test_engineered_feature_values_on_a_known_prompt():
    raw = text_features.extract_raw([SAMPLE])[0]
    feats = dict(zip(text_features.FEATURE_NAMES, raw, strict=True))
    assert feats["n_chars"] == len(SAMPLE)
    assert feats["n_code_fences"] == 2
    assert feats["n_inline_code"] == 1
    assert feats["n_question_marks"] == 1
    assert feats["n_latex_commands"] == 1
    assert feats["n_dollar_math"] == 1
    assert feats["n_table_rows"] == 2
    assert feats["n_json_keys"] == 1
    assert feats["n_instruction_verbs"] == 1  # "Write"
    assert feats["n_code_tokens"] > 0


def test_engineered_features_are_deterministic_rowwise_and_finite():
    texts = [SAMPLE, "What is 2 + 2?", "", "Explain recursion."]
    batch = text_features.extract(texts)
    rowwise = np.vstack([text_features.extract([t]) for t in texts])
    assert batch.dtype == np.float32 and batch.shape == (4, len(text_features.FEATURE_NAMES))
    assert np.array_equal(batch, rowwise)
    assert np.array_equal(batch, text_features.extract(texts))
    assert np.isfinite(batch).all() and (batch[2] == 0).all()


def test_feature_schema_hash_detects_changed_definitions(monkeypatch):
    before = text_features.schema()
    assert before == text_features.schema()
    monkeypatch.setattr(text_features, "_PATTERNS", [*text_features._PATTERNS, "changed"])
    assert text_features.schema()["definition_hash"] != before["definition_hash"]


# ---------------------------------------------------------------- labels and alignment
def test_label_encoding_uses_fixed_order():
    encoded = encode_labels(["easy", "medium", "hard", "easy"])
    assert encoded.dtype == np.int32 and encoded.tolist() == [0, 1, 2, 0]
    assert decode_labels(encoded).tolist() == ["easy", "medium", "hard", "easy"]
    with pytest.raises(AlignmentError, match="trivial"):
        encode_labels(["trivial"])


def test_features_align_with_labels_ids_and_texts(embedded):
    emb_dir = resolve_embeddings_dir(embedded)
    split, meta = load_split_features(
        embedded.paths.processed_dir, emb_dir, "train", "embeddings+engineered"
    )
    rows = _rows(embedded, "train")
    assert [rows[i]["label"] for i in split.example_ids] == decode_labels(split.y).tolist()
    X = split.matrix.rows(0, split.matrix.n_rows)
    assert X.dtype == np.float32 and X.shape[1] == meta["dimension"] + len(
        text_features.FEATURE_NAMES
    )
    assert np.array_equal(X[:, : meta["dimension"]], np.load(emb_dir / "train.npy"))
    queries = [rows[i]["query"] for i in split.example_ids]
    assert np.array_equal(X[:, meta["dimension"] :], text_features.extract(queries))
    assert isinstance(split.matrix.embeddings, np.memmap)  # not copied into RAM


@pytest.mark.parametrize("field,value", [("label", "hard"), ("query_sha256", "0" * 16)])
def test_tampered_row_mapping_is_detected(embedded, field, value):
    emb_dir = resolve_embeddings_dir(embedded)
    path = emb_dir / "validation.row_mapping.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    if field == "label":
        value = next(lab for lab in ("easy", "medium", "hard") if lab != first["label"])
    first[field] = value
    path.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n", encoding="utf-8")
    with pytest.raises(AlignmentError, match="mismatch"):
        load_split_features(embedded.paths.processed_dir, emb_dir, "validation", "embeddings")


# ---------------------------------------------------------------- integrity checks
def test_verify_artifacts_reports_disjoint_splits(embedded):
    report = verify_embedding_artifacts(
        embedded.paths.processed_dir, resolve_embeddings_dir(embedded)
    )
    assert report["example_ids_disjoint"] and report["duplicate_groups_disjoint"]
    assert report["dimension"] == 64
    assert report["total_rows"] == sum(s["rows"] for s in report["splits"].values())


def test_duplicate_group_spanning_splits_is_detected(embedded):
    test_group = next(iter(_rows(embedded, "test").values()))["group"]
    path = embedded.paths.processed_dir / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["group"] = test_group
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(AlignmentError, match="spans"):
        verify_embedding_artifacts(embedded.paths.processed_dir, resolve_embeddings_dir(embedded))


def test_incomplete_embedding_cache_is_refused(connected, fake_client):
    job = EmbeddingJob(connected, fake_client)
    job.run(max_batches=2)
    with pytest.raises(AlignmentError, match="incomplete"):
        verify_embedding_artifacts(connected.paths.processed_dir, job.dir)


@pytest.mark.parametrize("problem", ["dimension", "nan", "dtype"])
def test_invalid_embedding_matrices_are_refused(embedded, problem):
    emb_dir = resolve_embeddings_dir(embedded)
    X = np.load(emb_dir / "validation.npy")
    if problem == "dimension":
        X = X[:, :32]
    elif problem == "nan":
        X[0, 0] = np.nan
    else:
        X = X.astype(np.float64)
    np.save(emb_dir / "validation.npy", X)
    with pytest.raises(AlignmentError):
        load_split_features(embedded.paths.processed_dir, emb_dir, "validation", "embeddings")


def test_near_duplicate_audit_flags_copied_vectors(embedded):
    emb_dir = resolve_embeddings_dir(embedded)
    val = np.load(emb_dir / "validation.npy")
    val[0] = np.load(emb_dir / "train.npy")[0] * 3.0  # same direction: cosine 1
    np.save(emb_dir / "validation.npy", val)
    first_id = json.loads(
        (emb_dir / "validation.row_mapping.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )["example_id"]
    report = near_duplicate_audit(emb_dir, chunk_rows=7)
    flagged = {f["example_id"]: f for f in report["validation"]["flagged"]}
    assert flagged[first_id]["cosine"] >= 0.9999
    assert report["validation"]["rows_at_or_above"]["0.99"] >= 1


# ---------------------------------------------------------------- memory-conscious access
def test_feature_matrix_batches_are_bounded_and_complete(embedded):
    emb_dir = resolve_embeddings_dir(embedded)
    split, _ = load_split_features(
        embedded.paths.processed_dir, emb_dir, "train", "embeddings+engineered"
    )
    matrix = split.matrix
    batches = list(matrix.iter_batches(7))
    assert all(len(b) <= 7 for b in batches)
    full = matrix.rows(0, matrix.n_rows)
    assert np.array_equal(np.vstack(batches), full)
    order = np.random.default_rng(0).permutation(matrix.n_rows)[:20]
    assert np.array_equal(matrix.take(order), full[order])


def test_standardizer_is_fitted_on_the_given_rows_only():
    rng = np.random.default_rng(1)
    train = rng.normal(3, 2, size=(50, 4)).astype(np.float32)
    std = Standardizer.fit(FeatureMatrix(train, None), chunk_rows=7)
    assert np.allclose(std.mean, train.mean(axis=0), atol=1e-5)
    assert np.allclose(std.scale, train.std(axis=0), atol=1e-4)
    assert np.allclose(std.transform(train).mean(axis=0), 0, atol=1e-5)
