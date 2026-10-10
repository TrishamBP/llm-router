"""Synthetic GPT-5 generation: budget guarantee, parsing, cost, export. No API calls."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from llm_router.data import synthetic_gpt5 as sg


@pytest.fixture
def cfg(tmp_path, repo_root):
    base = sg.load_synth_config(repo_root / "configs" / "synthetic_gpt5.yaml")
    paths = {"work_dir": tmp_path / "work", "output": tmp_path / "items.jsonl"}
    return sg.SynthConfig(**{**base.__dict__, **paths})


def completed_body(items, usage=None):
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning"},
            {
                "type": "message",
                "content": [{"type": "output_text", "text": json.dumps({"items": items})}],
            },
        ],
        "usage": usage
        or {
            "input_tokens": 600,
            "output_tokens": 1500,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def test_shipped_config_is_gpt5_batch_with_hard_cap(cfg):
    assert cfg.model == "gpt-5" and cfg.reasoning_effort == "minimal"
    assert (
        cfg.budget_usd <= 1.29
        and cfg.prices["batch"]["output"] == cfg.prices["standard"]["output"] / 2
    )


def test_cells_rotate_labels_evenly_and_are_unique(cfg):
    all_cells = sg.cells(cfg)
    assert len({c["cell_id"] for c in all_cells}) == len(all_cells) == 3 * 30 * 6
    first_90 = Counter(c["label"] for c in all_cells[:90])
    assert set(first_90.values()) == {30}


def test_request_body_is_structured_and_bounded(cfg):
    body = sg.request_body(cfg, sg.cells(cfg)[0])
    assert body["max_output_tokens"] == cfg.max_output_tokens
    assert body["reasoning"] == {"effort": "minimal"}
    assert body["text"]["format"]["strict"] is True


def test_parse_response_and_failures():
    items = [{"prompt": "What is 2+2?", "label": "easy"}]
    assert sg.parse_response(completed_body(items)) == items
    with pytest.raises(ValueError, match="incomplete"):
        sg.parse_response(
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}
        )
    with pytest.raises(ValueError, match="no output_text"):
        sg.parse_response({"status": "completed", "output": []})


def test_usage_cost_counts_cached_and_reasoning_tokens():
    prices = {"input": 1.0, "cached_input": 0.1, "output": 10.0}
    usage = {
        "input_tokens": 1000,
        "input_tokens_details": {"cached_tokens": 400},
        "output_tokens": 2000,
    }
    assert sg.usage_cost(usage, prices) == pytest.approx(
        (600 * 1.0 + 400 * 0.1 + 2000 * 10.0) / 1e6
    )


def test_plan_never_exceeds_budget_even_at_worst_case(cfg):
    state = sg.load_state(cfg)
    with pytest.raises(SystemExit, match="pilot"):
        sg.plan_batch(cfg, state)
    state["pilot"] = {"input_tokens": 700}
    state["spent_usd"] = 0.02
    plan = sg.plan_batch(cfg, state)
    assert plan["requests"] > 0
    assert state["spent_usd"] + plan["worst_case_total_usd"] <= cfg.budget_usd
    # A pending (uncollected) batch's worst case is reserved before planning more.
    state["batches"].append({"worst_case_usd": cfg.budget_usd - 0.02, "collected": False})
    assert sg.plan_batch(cfg, state)["requests"] == 0


def test_export_validates_and_deduplicates(cfg):
    cfg.work_dir.mkdir(parents=True)
    cell = sg.cells(cfg)[0]
    rec = {
        "cell": cell,
        "source": "batch_00",
        "response_id": "r1",
        "items": [
            {"prompt": "Explain recursion with an example.", "label": "medium"},
            {"prompt": "explain recursion with an example", "label": "medium"},  # duplicate
            {"prompt": "hi", "label": "easy"},  # too short
            {"prompt": "Prove Fermat's little theorem.", "label": "extreme"},  # invalid label
        ],
    }
    (cfg.work_dir / "raw_items.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    out = sg.cmd_export(cfg)
    assert out["rows"] == 1
    assert out["removed"] == {"duplicate_prompt": 1, "bad_length": 1, "invalid_label": 1}
    row = json.loads(cfg.output.read_text(encoding="utf-8"))
    assert row["generator"] == "gpt-5" and row["target_label"] == cell["label"]
