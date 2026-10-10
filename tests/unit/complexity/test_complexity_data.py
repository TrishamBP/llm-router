from __future__ import annotations

from collections import Counter

import pytest

from llm_router.config import LABELS, PreprocessingConfig, load_config
from llm_router.data.inspect import inspect_records
from llm_router.data.preprocess import preprocess_records, verify_split_isolation
from llm_router.data.splitting import group_key, normalize_text, stratified_group_split
from llm_router.training.smoke_complexity import build_workspace

RATIOS = {"train": 0.8, "validation": 0.1, "test": 0.1}


def test_shipped_config_loads_with_80_10_10_splits(repo_root):
    cfg = load_config(repo_root / "configs" / "complexity.yaml")
    assert cfg.preprocessing.split_ratios == RATIOS
    assert cfg.preprocessing.min_confidence is None  # filtering disabled by default
    assert cfg.dataset.text_field == "text"
    assert cfg.paths.raw_dir.is_absolute()


def test_normalize_preserves_code_and_math():
    code = "def f(x):\r\n    return x ** 2  # square  \r\n\r\n\r\n\r\nprint(f(3))"
    assert normalize_text(code) == "def f(x):\n    return x ** 2  # square\n\nprint(f(3))"
    math_text = "  Prove ∑_{i=1}^n i = n(n+1)/2 for n ≥ 1;  x\u200b+y  "
    assert normalize_text(math_text) == "Prove ∑_{i=1}^n i = n(n+1)/2 for n ≥ 1;  x+y"


def test_group_key_matches_near_identical_queries():
    assert group_key("What is Python?") == group_key("  what is  PYTHON ")
    assert group_key("What is Python?") != group_key("What is Java?")


@pytest.mark.parametrize("seed", [1, 2])
def test_stratified_group_split_ratios_isolation_and_determinism(seed):
    labels, groups = [], []
    for label, n in (("easy", 500), ("medium", 800), ("hard", 300)):
        for i in range(n):
            labels.append(label)
            groups.append(f"{label}-{i // 2}")  # pairs of rows share a group
    first = stratified_group_split(labels, groups, RATIOS, seed)
    assert first == stratified_group_split(labels, groups, RATIOS, seed)
    group_split = {}
    for g, s in zip(groups, first, strict=True):
        assert group_split.setdefault(g, s) == s  # a group never spans splits
    for label in ("easy", "medium", "hard"):
        counts = Counter(s for lab, s in zip(labels, first, strict=True) if lab == label)
        total = sum(counts.values())
        for name, ratio in RATIOS.items():
            assert abs(counts[name] / total - ratio) < 0.01


def test_split_rejects_label_conflicts_within_a_group():
    with pytest.raises(ValueError, match="conflicting labels"):
        stratified_group_split(["easy", "hard"], ["g", "g"], RATIOS, 0)


def records_cfg(tmp_path, **pp):
    cfg = build_workspace(tmp_path)
    return cfg.model_copy(update={"preprocessing": PreprocessingConfig(**pp)})


def test_preprocess_validates_labels_and_handles_duplicates(tmp_path):
    cfg = records_cfg(tmp_path)
    records = {
        "train": [
            {"text": "What is Python?", "label": "easy"},
            {"text": "what is python", "label": "easy"},  # duplicate after normalisation
            {"text": "Is it hard?", "label": "easy"},
            {"text": "is it hard", "label": "hard"},  # conflicting label
            {"text": "", "label": "easy"},
            {"text": None, "label": "easy"},
            {"text": "Valid but bad label", "label": "EASY"},
        ]
        + [{"text": f"medium question {i}", "label": "medium"} for i in range(20)],
        "test": [{"text": f"hard question {i}", "label": "hard"} for i in range(20)],
    }
    result = preprocess_records(records, cfg)
    reasons = Counter(r.reason for r in result.removed)
    assert reasons == {
        "duplicate": 1,
        "conflicting_labels": 2,
        "empty_query": 1,
        "missing_query": 1,
        "invalid_label": 1,
    }
    kept = [e for rows in result.splits.values() for e in rows]
    python = next(e for e in kept if e.query == "What is Python?")
    assert python.example_id == "train-000000" and python.duplicate_ids == ["train-000001"]
    assert {e.label for e in kept} <= set(LABELS)
    assert {e.source_split for e in kept} == {"train", "test"}
    assert result.manifest["rows_in"] == 47 and result.manifest["rows_out"] == 41
    assert result.manifest["split_isolation"]["verified"] is True
    assert result.manifest["confidence"] == {"present": False, "invalid": 0}


def test_majority_conflict_policy(tmp_path):
    cfg = records_cfg(tmp_path, conflict_policy="majority", duplicate_policy="keep_all")
    records = {
        "train": [
            {"text": "q", "label": "hard"},
            {"text": "Q", "label": "hard"},
            {"text": "q?", "label": "easy"},
        ]
    }
    result = preprocess_records(records, cfg)
    assert sorted(e.label for rows in result.splits.values() for e in rows) == ["hard"] * 3


def test_confidence_validation_and_optional_filtering(tmp_path):
    rows = [
        {"text": f"q{i}", "label": "medium", "confidence": c}
        for i, c in enumerate([1.0, 0.67, 1.5, "x"])
    ]
    rows += [{"text": f"h{i}", "label": "hard", "confidence": 1.0} for i in range(10)]
    unfiltered = preprocess_records({"train": rows}, records_cfg(tmp_path))
    assert unfiltered.manifest["confidence"] == {"present": True, "invalid": 2}
    assert unfiltered.manifest["rows_out"] == 14  # invalid confidence is reported, not dropped
    filtered = preprocess_records({"train": rows}, records_cfg(tmp_path / "f", min_confidence=1.0))
    assert Counter(r.reason for r in filtered.removed) == {"below_min_confidence": 3}


def test_confidence_filter_requires_a_confidence_column(tmp_path):
    with pytest.raises(ValueError, match="no confidence column"):
        preprocess_records(
            {"train": [{"text": "q", "label": "easy"}]}, records_cfg(tmp_path, min_confidence=0.9)
        )


def test_preprocessing_is_deterministic(tmp_path):
    records = {"train": [{"text": f"q {i}", "label": LABELS[i % 3]} for i in range(90)]}
    a = preprocess_records(records, records_cfg(tmp_path / "a"))
    b = preprocess_records(records, records_cfg(tmp_path / "b"))
    assert a.manifest["processed_fingerprint"] == b.manifest["processed_fingerprint"]
    assert {k: [e.example_id for e in v] for k, v in a.splits.items()} == {
        k: [e.example_id for e in v] for k, v in b.splits.items()
    }


def test_isolation_check_detects_leaks(tmp_path):
    result = preprocess_records(
        {"train": [{"text": f"q {i}", "label": "easy"} for i in range(30)]}, records_cfg(tmp_path)
    )
    leaked = result.splits["train"][0]
    result.splits["test"].append(leaked)
    assert verify_split_isolation(result.splits)["verified"] is False


def test_inspection_reports_schema_problems():
    records = {
        "train": [
            {"text": "Same", "label": "easy"},
            {"text": "Same", "label": "hard"},
            {"text": " ", "label": "medium"},
            {"text": "Other", "label": "bogus"},
        ],
        "test": [{"text": "same", "label": "easy"}],
    }
    report = inspect_records(
        records, text_field="text", label_field="label", confidence_field="confidence"
    )
    assert report["expected_fields_missing"] == ["confidence"]
    assert report["unexpected_labels"] == ["bogus"]
    assert report["exact_duplicates"] == {"groups": 1, "extra_rows": 1}
    assert report["normalized_duplicates"]["groups_spanning_official_splits"] == 1
    assert report["conflicting_labels"]["groups"] == 1
    assert report["splits"]["train"]["empty_text"] == 1
