"""Generate a synthetic easy/medium/hard prompt dataset with GPT-5 under a hard budget cap.

Usage (each step is explicit; nothing spends money without a command):
    uv run python -m llm_router.data.synthetic_gpt5 pilot     # 1 standard request; measures cost
    uv run python -m llm_router.data.synthetic_gpt5 submit    # 1 batch sized to remaining budget
    uv run python -m llm_router.data.synthetic_gpt5 status    # poll batches (free)
    uv run python -m llm_router.data.synthetic_gpt5 collect   # download, validate, account cost
    uv run python -m llm_router.data.synthetic_gpt5 export    # clean + dedupe -> items.jsonl

Budget guarantee: every request has `max_output_tokens`, so its worst-case cost is
input_tokens * input_price + max_output_tokens * output_price (reasoning tokens are output).
`submit` only sends as many requests as fit in `budget - spent - reserved_for_pending` at
worst-case cost. After `collect`, actual cost replaces the reservation, and `submit` can use
what is left. Spend = pilot (standard price) + batch usage (batch price, 50% off).

Each request asks for N prompts targeting one (label, domain, style) cell; GPT-5 also returns
its own label for each prompt using the same rubric as regolo/brick-complexity-extractor.
The key is read from OPENAI_API_KEY (.env) and never logged or written to artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llm_router.data.splitting import group_key, normalize_text
from llm_router.logging import configure_logging, get_logger

log = get_logger(__name__)

RUBRIC = """Label each prompt by the cognitive effort needed to produce a high-quality answer
(not by answer length):
- easy: 1-2 reasoning steps. Surface knowledge, factual recall, direct lookups, simple
  calculations or conversions.
- medium: 3-5 reasoning steps. Needs domain familiarity, multi-step reasoning, comparison,
  analysis, or moderate synthesis.
- hard: 6+ reasoning steps. Deep expertise, multi-constraint optimisation, creative synthesis,
  advanced mathematics or proofs, or system design under complex constraints.
Consider reasoning depth, required expertise, constraint satisfaction, ambiguity, creativity."""

INSTRUCTIONS = f"""You create realistic user prompts for training an LLM-routing classifier.

{RUBRIC}

Write prompts the way real users type them to an AI assistant: varied phrasing, lengths
(a few words up to a short paragraph), and specificity. English only. Self-contained (no
attachments). No personal data, no harmful requests. Do not number prompts or mention
difficulty inside a prompt. Each prompt must be distinct from the others.
For every prompt, return the label YOU judge correct under the rubric, even if it differs
from the requested target."""

DOMAINS = [
    "mathematics",
    "programming and software engineering",
    "data science and machine learning",
    "physics",
    "chemistry",
    "biology and medicine",
    "earth and environmental science",
    "history",
    "geography and travel",
    "economics and finance",
    "business and management",
    "law and policy",
    "philosophy and ethics",
    "psychology",
    "linguistics and languages",
    "literature and creative writing",
    "music and arts",
    "cooking and nutrition",
    "health and fitness",
    "personal productivity",
    "education and studying",
    "computer networking and security",
    "databases and data engineering",
    "cloud and distributed systems",
    "electrical and mechanical engineering",
    "astronomy and space",
    "sports",
    "parenting and relationships",
    "home and DIY",
    "everyday trivia and general knowledge",
]
STYLES = [
    "a direct question",
    "an instruction or task request",
    "a multi-part request with explicit constraints",
    "a request that includes a short code snippet, formula, or data sample",
    "a casual conversational message",
    "a request for an explanation aimed at a specific audience",
]

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["prompt", "label"],
                "properties": {
                    "prompt": {"type": "string"},
                    "label": {"type": "string", "enum": ["easy", "medium", "hard"]},
                },
            },
        }
    },
}


@dataclass(frozen=True)
class SynthConfig:
    model: str
    reasoning_effort: str
    prompts_per_request: int
    max_output_tokens: int
    labels: tuple[str, ...]
    per_label: dict[str, dict[str, int]]  # label -> {prompts, max_output_tokens}
    seed: int
    budget_usd: float
    prices: dict[str, dict[str, float]]
    work_dir: Path
    output: Path


def load_synth_config(path: str | Path = "configs/synthetic_gpt5.yaml") -> SynthConfig:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    return SynthConfig(
        model=data["model"],
        reasoning_effort=data["reasoning_effort"],
        prompts_per_request=int(data["prompts_per_request"]),
        max_output_tokens=int(data["max_output_tokens"]),
        labels=tuple(data["labels"]),
        per_label={
            label: {
                "prompts": int(
                    (data.get("per_label") or {})
                    .get(label, {})
                    .get("prompts", data["prompts_per_request"])
                ),
                "max_output_tokens": int(
                    (data.get("per_label") or {})
                    .get(label, {})
                    .get("max_output_tokens", data["max_output_tokens"])
                ),
            }
            for label in data["labels"]
        },
        seed=int(data["seed"]),
        budget_usd=float(data["budget"]["total_usd"]),
        prices=data["prices"],
        work_dir=(root / data["paths"]["work_dir"]).resolve(),
        output=(root / data["paths"]["output"]).resolve(),
    )


# ---------------------------------------------------------------- requests


def cells(cfg: SynthConfig) -> list[dict[str, str]]:
    """All (label, domain, style) cells, in a seeded order that rotates labels evenly."""
    rng = random.Random(cfg.seed)
    per_label = []
    for label in cfg.labels:
        combos = [{"label": label, "domain": d, "style": s} for d in DOMAINS for s in STYLES]
        rng.shuffle(combos)
        per_label.append(combos)
    ordered = [c for group in zip(*per_label, strict=True) for c in group]
    for i, c in enumerate(ordered):
        c["cell_id"] = f"c{i:04d}-{c['label']}"
    return ordered


def request_body(cfg: SynthConfig, cell: dict[str, str]) -> dict[str, Any]:
    user = (
        f"Write {cfg.per_label[cell['label']]['prompts']} different user prompts.\n"
        f"Target difficulty: {cell['label']}\nDomain: {cell['domain']}\nStyle: {cell['style']}\n"
        "Return JSON matching the schema."
    )
    return {
        "model": cfg.model,
        "reasoning": {"effort": cfg.reasoning_effort},
        "instructions": INSTRUCTIONS,
        "input": user,
        "max_output_tokens": cfg.per_label[cell["label"]]["max_output_tokens"],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "synthetic_prompts",
                "strict": True,
                "schema": SCHEMA,
            }
        },
    }


def parse_response(body: dict[str, Any]) -> list[dict[str, str]]:
    """Items from a Responses API body; raises ValueError if incomplete or malformed."""
    if body.get("status") != "completed":
        reason = (body.get("incomplete_details") or {}).get("reason")
        raise ValueError(f"response status {body.get('status')!r} ({reason})")
    texts = [
        part.get("text", "")
        for item in body.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    ]
    if not texts:
        raise ValueError("no output_text in response")
    items = json.loads("".join(texts))["items"]
    if not isinstance(items, list):
        raise ValueError("'items' is not a list")
    return items


def usage_cost(usage: dict[str, Any], prices: dict[str, float]) -> float:
    inp = int(usage.get("input_tokens", 0))
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
    out = int(usage.get("output_tokens", 0))  # includes reasoning tokens
    return (
        (inp - cached) * prices["input"] + cached * prices["cached_input"] + out * prices["output"]
    ) / 1e6


# ---------------------------------------------------------------- state & budget


def _state_path(cfg: SynthConfig) -> Path:
    return cfg.work_dir / "state.json"


def load_state(cfg: SynthConfig) -> dict[str, Any]:
    path = _state_path(cfg)
    if path.exists():
        state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return state
    return {"model": cfg.model, "spent_usd": 0.0, "pilot": None, "batches": [], "used_cells": []}


def save_state(cfg: SynthConfig, state: dict[str, Any]) -> None:
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    tmp = _state_path(cfg).with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, _state_path(cfg))


def reserved_usd(state: dict[str, Any]) -> float:
    return float(sum(b["worst_case_usd"] for b in state["batches"] if not b.get("collected")))


def worst_case_request_usd(cfg: SynthConfig, input_tokens: int, label: str | None = None) -> float:
    p = cfg.prices["batch"]
    ceiling = cfg.per_label[label]["max_output_tokens"] if label else cfg.max_output_tokens
    return (input_tokens * p["input"] + ceiling * p["output"]) / 1e6


def label_counts(cfg: SynthConfig) -> Counter[str]:
    """Prompts collected so far, by GPT-5's returned label."""
    counts: Counter[str] = Counter()
    path = cfg.work_dir / "raw_items.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            counts.update(i.get("label") for i in json.loads(line)["items"])
    return counts


def plan_batch(cfg: SynthConfig, state: dict[str, Any]) -> dict[str, Any]:
    if not state.get("pilot"):
        raise SystemExit("run `pilot` first (it measures input tokens and validates the format)")
    available = cfg.budget_usd - state["spent_usd"] - reserved_usd(state)
    input_tokens = int(state["pilot"]["input_tokens"] * 1.05) + 1
    used = set(state["used_cells"])
    pools = {
        lab: [c for c in cells(cfg) if c["label"] == lab and c["cell_id"] not in used]
        for lab in cfg.labels
    }
    have = label_counts(cfg)
    chosen: list[dict[str, str]] = []
    total = 0.0
    while True:  # always top up the label furthest below its share (balanced target)
        options = [lab for lab in cfg.labels if pools[lab]]
        if not options:
            break
        lab = min(options, key=lambda x: (have[x], cfg.labels.index(x)))
        cost = worst_case_request_usd(cfg, input_tokens, lab)
        if total + cost > available:
            break
        chosen.append(pools[lab].pop(0))
        total += cost
        have[lab] += cfg.per_label[lab]["prompts"]
    return {
        "available_usd": round(available, 4),
        "requests": len(chosen),
        "requests_by_label": dict(Counter(c["label"] for c in chosen)),
        "worst_case_total_usd": round(total, 4),
        "prompts_requested": sum(cfg.per_label[c["label"]]["prompts"] for c in chosen),
        "cells": chosen,
    }


# ---------------------------------------------------------------- commands


def _client() -> Any:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(".env", override=False)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (put it in the local .env, never in code)")
    return OpenAI(api_key=key, max_retries=2, timeout=600)


def _safe_error(exc: Exception) -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    text = f"{type(exc).__name__}: {getattr(exc, 'status_code', '')} {getattr(exc, 'message', exc)}"
    return (text.replace(key, "***") if key else text)[:300]


def cmd_pilot(cfg: SynthConfig) -> dict[str, Any]:
    state = load_state(cfg)
    if state.get("pilot"):
        return {"pilot": state["pilot"], "note": "pilot already done"}
    cell = cells(cfg)[0]
    body = request_body(cfg, cell)
    if state["spent_usd"] + worst_case_request_usd(cfg, 2000) * 2 > cfg.budget_usd:
        raise SystemExit("budget too small for even a pilot")
    import openai

    try:
        response = _client().responses.create(**body)
    except openai.APIError as exc:
        raise SystemExit(f"pilot request failed: {_safe_error(exc)}") from None
    raw = response.model_dump()
    usage = raw.get("usage") or {}
    cost = usage_cost(usage, cfg.prices["standard"])
    state["spent_usd"] += cost
    items: list[dict[str, str]] = []
    error = None
    try:
        items = parse_response(raw)
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        error = str(exc)
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    (cfg.work_dir / "pilot_response.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    state["pilot"] = {
        "cell": cell,
        "response_id": raw.get("id"),
        "model_reported": raw.get("model"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": (usage.get("output_tokens_details") or {}).get("reasoning_tokens"),
        "items": len(items),
        "cost_usd_standard": round(cost, 5),
        "est_cost_per_request_batch_usd": round(usage_cost(usage, cfg.prices["batch"]), 5),
        "parse_error": error,
    }
    state["used_cells"].append(cell["cell_id"])
    if items:
        _append_items(
            cfg, [{"cell": cell, "items": items, "source": "pilot", "response_id": raw.get("id")}]
        )
    save_state(cfg, state)
    return {"pilot": state["pilot"], "spent_usd": round(state["spent_usd"], 5)}


def cmd_submit(cfg: SynthConfig, *, dry_run: bool = False) -> dict[str, Any]:
    state = load_state(cfg)
    plan = plan_batch(cfg, state)
    summary = {k: v for k, v in plan.items() if k != "cells"}
    if dry_run or plan["requests"] == 0:
        return {"submitted": False, **summary}
    lines = [
        json.dumps(
            {
                "custom_id": f"{c['cell_id']}-b{len(state['batches']):02d}",
                "method": "POST",
                "url": "/v1/responses",
                "body": request_body(cfg, c),
            }
        )
        for c in plan["cells"]
    ]
    index = len(state["batches"])
    req_path = cfg.work_dir / f"batch_{index:02d}_requests.jsonl"
    req_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    import openai

    client = _client()
    try:
        with req_path.open("rb") as fh:
            file = client.files.create(file=fh, purpose="batch")
        batch = client.batches.create(
            input_file_id=file.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"project": "llm-router", "dataset": "synthetic_gpt5"},
        )
    except openai.APIError as exc:
        raise SystemExit(f"batch submission failed: {_safe_error(exc)}") from None
    state["batches"].append(
        {
            "index": index,
            "batch_id": batch.id,
            "input_file_id": file.id,
            "requests": plan["requests"],
            "cells": [c["cell_id"] for c in plan["cells"]],
            "worst_case_usd": plan["worst_case_total_usd"],
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "collected": False,
        }
    )
    state["used_cells"].extend(c["cell_id"] for c in plan["cells"])
    save_state(cfg, state)
    return {"submitted": True, "batch_id": batch.id, **summary}


def cmd_status(cfg: SynthConfig) -> list[dict[str, Any]]:
    state = load_state(cfg)
    client = _client()
    out = []
    for b in state["batches"]:
        info = client.batches.retrieve(b["batch_id"])
        counts = info.request_counts.model_dump() if info.request_counts else {}
        out.append(
            {
                "index": b["index"],
                "batch_id": b["batch_id"],
                "status": info.status,
                "request_counts": counts,
                "collected": b.get("collected", False),
            }
        )
    return out


def cmd_collect(cfg: SynthConfig) -> dict[str, Any]:
    state = load_state(cfg)
    client = _client()
    collected = []
    for b in state["batches"]:
        if b.get("collected"):
            continue
        info = client.batches.retrieve(b["batch_id"])
        if info.status not in ("completed", "expired", "cancelled", "failed"):
            continue
        cost, ok, failed, incomplete, records = 0.0, 0, 0, 0, []
        failed_cells: list[str] = []
        for file_id, kind in ((info.output_file_id, "output"), (info.error_file_id, "error")):
            if not file_id:
                continue
            text = client.files.content(file_id).text
            (cfg.work_dir / f"batch_{b['index']:02d}_{kind}.jsonl").write_text(
                text, encoding="utf-8"
            )
            for line in text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                body = ((rec.get("response") or {}).get("body")) or {}
                usage = body.get("usage") or {}
                cost += usage_cost(usage, cfg.prices["batch"]) if usage else 0.0
                cell_id = rec["custom_id"].split("-b")[0]  # strip the per-batch suffix
                try:
                    items = parse_response(body)
                except (ValueError, KeyError, json.JSONDecodeError):
                    if body.get("status") == "incomplete":
                        incomplete += 1
                    failed += 1
                    failed_cells.append(cell_id)
                    continue
                ok += 1
                cell = next(c for c in cells(cfg) if c["cell_id"] == cell_id)
                records.append(
                    {
                        "cell": cell,
                        "items": items,
                        "source": f"batch_{b['index']:02d}",
                        "response_id": body.get("id"),
                    }
                )
        _append_items(cfg, records)
        # Cells that produced nothing (truncated/failed) may be retried by a later batch.
        retry = set(failed_cells)
        state["used_cells"] = [c for c in state["used_cells"] if c not in retry]
        state["spent_usd"] += cost
        b.update(
            {
                "collected": True,
                "status": info.status,
                "actual_cost_usd": round(cost, 5),
                "ok_requests": ok,
                "failed_requests": failed,
                "incomplete_requests": incomplete,
            }
        )
        collected.append(
            {
                k: b[k]
                for k in (
                    "index",
                    "status",
                    "actual_cost_usd",
                    "ok_requests",
                    "failed_requests",
                    "incomplete_requests",
                )
            }
        )
    save_state(cfg, state)
    return {
        "collected": collected,
        "spent_usd": round(state["spent_usd"], 5),
        "remaining_budget_usd": round(cfg.budget_usd - state["spent_usd"] - reserved_usd(state), 5),
    }


def _append_items(cfg: SynthConfig, records: list[dict[str, Any]]) -> None:
    path = cfg.work_dir / "raw_items.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def cmd_export(cfg: SynthConfig) -> dict[str, Any]:
    """Validate, normalise, and deduplicate (within the synthetic set) into items.jsonl."""
    path = cfg.work_dir / "raw_items.jsonl"
    if not path.exists():
        raise SystemExit("no raw items yet; run pilot/collect first")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    removed: Counter[str] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        for j, item in enumerate(rec["items"]):
            prompt = normalize_text(str(item.get("prompt", "")))
            label = item.get("label")
            if not 5 <= len(prompt) <= 2000:
                removed["bad_length"] += 1
                continue
            if label not in cfg.labels:
                removed["invalid_label"] += 1
                continue
            key = group_key(prompt)
            if key in seen:
                removed["duplicate_prompt"] += 1
                continue
            seen.add(key)
            rows.append(
                {
                    "example_id": f"synth-{rec['cell']['cell_id']}-{j:02d}",
                    "query": prompt,
                    "label": label,
                    "target_label": rec["cell"]["label"],
                    "domain": rec["cell"]["domain"],
                    "style": rec["cell"]["style"],
                    "group": key,
                    "source": rec["source"],
                    "response_id": rec.get("response_id"),
                    "generator": cfg.model,
                }
            )
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    agree = sum(r["label"] == r["target_label"] for r in rows)
    return {
        "rows": len(rows),
        "removed": dict(removed),
        "label_distribution": dict(sorted(Counter(r["label"] for r in rows).items())),
        "label_matches_target": round(agree / len(rows), 4) if rows else None,
        "output": str(cfg.output),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=["pilot", "plan", "submit", "status", "collect", "export"]
    )
    parser.add_argument("--config", default="configs/synthetic_gpt5.yaml")
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_synth_config(args.config)
    if args.command == "pilot":
        result: Any = cmd_pilot(cfg)
    elif args.command == "plan":
        result = cmd_submit(cfg, dry_run=True)
    elif args.command == "submit":
        result = cmd_submit(cfg)
    elif args.command == "status":
        result = cmd_status(cfg)
    elif args.command == "collect":
        result = cmd_collect(cfg)
    else:
        result = cmd_export(cfg)
    json.dump(result, sys.stdout, indent=2, default=str)
    print()


if __name__ == "__main__":
    main()
