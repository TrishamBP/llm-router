"""Fireworks embeddings for the prepared router dataset (reuses the shared client and job).

Usage:
    uv run python -m llm_router.embeddings.generate_router_embeddings --test       # 1 request
    uv run python -m llm_router.embeddings.generate_router_embeddings --estimate   # no API calls
    uv run python -m llm_router.embeddings.generate_router_embeddings --yes        # full run

Same behaviour as `llm_router.embeddings.generate` (bounded retries with jitter, permanent
errors fail fast, atomic per-batch parts, resume, dimension/finiteness checks), with:
- input field `prompt`, manifest `data/processed/router_dataset/split_metadata.json`;
- cache under `artifacts/embeddings/router_dataset/<cache_key>/`, keyed by the processed
  fingerprint (raw revision + normalised prompt hashes + labels + split assignment), the
  preparation version, the embedding model ID, and the dimensions setting;
- a connectivity result of its own (`.../router_dataset/connectivity/`), required before
  any batch is embedded and checked against the configured model ID.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from llm_router.data.router_dataset import (
    DEFAULT_CONFIG,
    MANIFEST_NAME,
    TEXT_FIELD,
    RouterDatasetConfig,
    load_router_config,
)
from llm_router.embeddings.fireworks_client import (
    EmbeddingClient,
    EmbeddingError,
    FireworksEmbeddingClient,
    FireworksSettings,
    connectivity_test,
)
from llm_router.embeddings.generate import (
    PILOT_BATCH_LIMIT,
    EmbeddingJob,
    EmbeddingJobError,
    _PlanOnlyClient,
)
from llm_router.logging import configure_logging, get_logger
from llm_router.training.resources import peak_rss_bytes, to_gib

log = get_logger(__name__)

SYNTHETIC_PROMPT = "Synthetic router connectivity check: summarise the benefits of unit tests."


def make_job(
    cfg: RouterDatasetConfig, client: EmbeddingClient, *, require_connectivity: bool = True
) -> EmbeddingJob:
    return EmbeddingJob(
        cfg.pipeline_config(),
        client,
        require_connectivity=require_connectivity,
        text_field=TEXT_FIELD,
        manifest_name=MANIFEST_NAME,
    )


def run_connectivity(cfg: RouterDatasetConfig, client: EmbeddingClient) -> dict[str, Any]:
    return connectivity_test(client, SYNTHETIC_PROMPT, cfg.paths.embeddings_dir / "connectivity")


def estimate(cfg: RouterDatasetConfig, job: EmbeddingJob) -> dict[str, Any]:
    """Workload summary; never calls the API. Token estimate prefers measured usage."""
    plan = job.plan()
    conn_path = cfg.paths.embeddings_dir / "connectivity" / "metadata.json"
    ratio_source = (
        "measured on completed batches of this job" if plan["measured_tokens_per_char"] else None
    )
    ratio = plan["measured_tokens_per_char"]
    if ratio is None and conn_path.exists():
        conn = json.loads(conn_path.read_text(encoding="utf-8"))
        if (
            conn.get("prompt_tokens")
            and conn.get("query_chars")
            and conn["model"] == job.client.model
        ):
            ratio = conn["prompt_tokens"] / conn["query_chars"]
            ratio_source = "connectivity prompt only (rough; one short sentence)"
    chars = sum(s["remaining_chars"] for s in plan["splits"].values())
    tokens = round(chars * ratio) if ratio else None
    price = cfg.embeddings.price_per_million_tokens
    return {
        "model": job.client.model,
        "cache_dir": plan["cache_dir"],
        "rows": {s: v["rows"] for s, v in plan["splits"].items()},
        "remaining_requests": plan["remaining_requests"],
        "remaining_chars": chars,
        "tokens_per_char": ratio,
        "tokens_per_char_source": ratio_source,
        "estimated_tokens": tokens,
        "estimated_cost_usd": round(tokens / 1e6 * price, 4)
        if tokens and price is not None
        else None,
        "price_per_million_tokens": price,
        "minimum_minutes_at_rate_limit": round(
            plan["remaining_requests"] / cfg.embeddings.requests_per_minute, 1
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fireworks embeddings for the router dataset")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--test", action="store_true", help="embed one synthetic prompt")
    parser.add_argument("--estimate", action="store_true", help="workload/cost; no API calls")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--yes", action="store_true", help="confirm a run larger than a pilot")
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_router_config(args.config)
    try:
        if args.estimate:
            from dotenv import load_dotenv

            load_dotenv(".env", override=False)
            model = os.environ.get("FIREWORKS_EMBEDDING_MODEL", "").strip() or "<unset>"
            job = make_job(cfg, _PlanOnlyClient(model), require_connectivity=False)
            print(json.dumps(estimate(cfg, job), indent=2))
            return
        client = FireworksEmbeddingClient(FireworksSettings.from_env(), cfg.embeddings)
        if args.test:
            print(json.dumps(run_connectivity(cfg, client), indent=2))
            return
        job = make_job(cfg, client)
        summary = estimate(cfg, job)
        print(json.dumps(summary, indent=2))
        planned = summary["remaining_requests"] if args.max_batches is None else args.max_batches
        if planned > PILOT_BATCH_LIMIT and not args.yes:
            sys.exit(f"{planned} requests planned; review the estimate and re-run with --yes")
        started = time.perf_counter()
        metadata = job.run(max_batches=args.max_batches)
    except (EmbeddingError, EmbeddingJobError) as exc:
        sys.exit(f"router embedding generation stopped: {exc}")
    keys = (
        "cache_key",
        "embedding_model",
        "dimension",
        "complete",
        "rows",
        "failed_examples",
        "usage",
        "embedding_seconds",
        "estimated_cost_usd",
    )
    out = {k: metadata[k] for k in keys}
    out["wall_seconds"] = round(time.perf_counter() - started, 1)
    out["peak_rss_gib"] = to_gib(peak_rss_bytes())
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
