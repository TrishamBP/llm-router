"""Generate, cache, and resume Fireworks embeddings for the processed splits.

Usage:
    uv run python -m llm_router.embeddings.generate --plan             # no API calls
    uv run python -m llm_router.embeddings.generate --max-batches 2    # small pilot
    uv run python -m llm_router.embeddings.generate --yes              # full run

Cache layout: `paths.embeddings_dir/<cache_key>/`, where the key hashes the processed-dataset
fingerprint, preprocessing version, embedding model ID, and dimensions setting. Each completed
batch is written atomically to `parts/<split>/<batch>.npz` and `state.json` is updated, so an
interrupted job resumes without re-embedding finished batches. When a split is complete it is
assembled into `<split>.npy` plus `<split>.row_mapping.jsonl` (row -> example_id, label).
One cache directory only ever holds vectors from one model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from llm_router.config import ComplexityConfig, load_config, sha256_json
from llm_router.data.preprocess import SPLITS, load_processed
from llm_router.embeddings.fireworks_client import (
    EmbeddingClient,
    EmbeddingError,
    InvalidEmbeddingResponseError,
    RequestRejectedError,
)
from llm_router.logging import configure_logging, get_logger

log = get_logger(__name__)

PILOT_BATCH_LIMIT = 5  # runs above this many batches need --yes


class EmbeddingJobError(Exception):
    pass


def cache_key(
    manifest: dict[str, Any], model: str, dimensions: int | None, input_field: str = "query"
) -> str:
    return sha256_json(
        {
            "processed": manifest["processed_fingerprint"],
            "preprocessing": manifest["preprocessing_version"],
            "model": model,
            "dimensions": dimensions,
            "input_field": input_field,
        }
    )[:16]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_bytes(path, json.dumps(payload, indent=2).encode("utf-8"))


class EmbeddingJob:
    def __init__(
        self,
        cfg: ComplexityConfig,
        client: EmbeddingClient,
        *,
        require_connectivity: bool = True,
        text_field: str = "query",
        manifest_name: str = "manifest.json",
    ) -> None:
        """`text_field` names the record field that is embedded (e.g. "prompt" for other
        datasets); records must also carry a unique `example_id` and a `label`."""
        self.cfg = cfg
        self.client = client
        self.text_field = text_field
        self.examples, self.manifest = load_processed(cfg.paths.processed_dir, manifest_name)
        isolation = self.manifest.get("split_isolation", {})
        if not isolation.get("verified"):
            raise EmbeddingJobError("split isolation is not verified; re-run preprocessing")
        self.batch_size = cfg.embeddings.batch_size
        self.key = cache_key(self.manifest, client.model, cfg.embeddings.dimensions, text_field)
        self.dir = cfg.paths.embeddings_dir / self.key
        self.expected_dim = self._expected_dimension(require_connectivity)
        self.state = self._load_state()

    # ------------------------------------------------------------------ setup
    def _expected_dimension(self, require_connectivity: bool) -> int | None:
        path = self.cfg.paths.embeddings_dir / "connectivity" / "metadata.json"
        if not path.exists():
            if require_connectivity:
                raise EmbeddingJobError(
                    "no successful connectivity test found; run "
                    "`python -m llm_router.embeddings.fireworks_client --test` first"
                )
            return None
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta["model"] != self.client.model:
            if require_connectivity:
                raise EmbeddingJobError(
                    f"connectivity test was for model '{meta['model']}', not "
                    f"'{self.client.model}'; re-run the connectivity test"
                )
            return None
        return int(meta["dimension"])

    def _load_state(self) -> dict[str, Any]:
        path = self.dir / "state.json"
        if path.exists():
            state: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            if state["model"] != self.client.model or state["batch_size"] != self.batch_size:
                raise EmbeddingJobError(
                    "cached job used a different model or batch size; keep the original "
                    f"batch_size ({state['batch_size']}) to resume, or use a new cache"
                )
            return state
        return {
            "cache_key": self.key,
            "model": self.client.model,
            "batch_size": self.batch_size,
            "dimension": self.expected_dim,
            "reported_models": [],
            "splits": {
                name: {
                    "total_batches": math.ceil(len(rows) / self.batch_size),
                    "done": [],
                    "failed": {},
                }
                for name, rows in self.examples.items()
            },
            "usage": {
                "requests": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "tokens_complete": True,
            },
            "embedding_seconds": 0.0,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def _save_state(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.dir / "state.json", self.state)

    # ------------------------------------------------------------------ planning
    def plan(self) -> dict[str, Any]:
        """Describe the remaining work without calling the API."""
        price = self.cfg.embeddings.price_per_million_tokens
        usage = self.state["usage"]
        done_chars = 0
        remaining: dict[str, Any] = {}
        for name in SPLITS:
            rows = self.examples[name]
            done = set(self.state["splits"][name]["done"])
            todo = [b for b in range(self.state["splits"][name]["total_batches"]) if b not in done]
            field = self.text_field
            chars = sum(len(r[field]) for b in todo for r in self._batch_rows(name, b))
            done_chars += sum(len(r[field]) for b in done for r in self._batch_rows(name, b))
            remaining[name] = {
                "rows": len(rows),
                "remaining_batches": len(todo),
                "remaining_chars": chars,
            }
        tokens_per_char = (
            usage["prompt_tokens"] / done_chars
            if done_chars and usage["tokens_complete"] and usage["prompt_tokens"]
            else None
        )
        total_chars = sum(s["remaining_chars"] for s in remaining.values())
        est_tokens = round(total_chars * tokens_per_char) if tokens_per_char else None
        return {
            "cache_dir": str(self.dir),
            "model": self.client.model,
            "batch_size": self.batch_size,
            "splits": remaining,
            "remaining_requests": sum(s["remaining_batches"] for s in remaining.values()),
            "measured_tokens_per_char": tokens_per_char,
            "estimated_remaining_tokens": est_tokens,
            "estimated_remaining_cost_usd": (
                round(est_tokens / 1e6 * price, 4)
                if est_tokens is not None and price is not None
                else None
            ),
            "cost_note": (
                "estimate from tokens reported for completed batches"
                if tokens_per_char and price is not None
                else "no estimate: needs reported usage from a pilot run and "
                "embeddings.price_per_million_tokens from Fireworks' pricing page"
            ),
        }

    # ------------------------------------------------------------------ running
    def _batch_rows(self, split: str, batch: int) -> list[dict[str, Any]]:
        return self.examples[split][batch * self.batch_size : (batch + 1) * self.batch_size]

    def run(self, splits: Sequence[str] = SPLITS, max_batches: int | None = None) -> dict[str, Any]:
        processed = 0
        complete = True
        try:
            for name in splits:
                split_state = self.state["splits"][name]
                done = set(split_state["done"])
                for batch in range(split_state["total_batches"]):
                    if batch in done:
                        continue
                    if max_batches is not None and processed >= max_batches:
                        complete = False
                        break
                    self._embed_batch(name, batch)
                    processed += 1
                if not complete:
                    break
                self._assemble(name)
        finally:
            self._save_state()
        return self._finish(complete=complete)

    def _embed_batch(self, split: str, batch: int) -> None:
        rows = self._batch_rows(split, batch)
        started = time.monotonic()
        ids, vectors, failed = self._embed_rows(rows)
        self.state["embedding_seconds"] += time.monotonic() - started
        part_dir = self.dir / "parts" / split
        part_dir.mkdir(parents=True, exist_ok=True)
        payload = _npz_bytes(vectors, ids)
        _atomic_write_bytes(part_dir / f"{batch:06d}.npz", payload)
        state = self.state["splits"][split]
        state["failed"].update(failed)
        state["done"].append(batch)
        self._save_state()
        log.info("batch done", split=split, batch=batch, rows=len(ids), failed=len(failed))

    def _embed_rows(
        self, rows: list[dict[str, Any]]
    ) -> tuple[list[str], np.ndarray, dict[str, str]]:
        try:
            result = self.client.embed([r[self.text_field] for r in rows])
        except RequestRejectedError as exc:
            if len(rows) == 1:
                return (
                    [],
                    np.zeros((0, self.expected_dim or 0), np.float32),
                    {rows[0]["example_id"]: f"rejected (HTTP {exc.status})"},
                )
            mid = len(rows) // 2  # bisect to isolate the offending input(s)
            halves = [self._embed_rows(rows[:mid]), self._embed_rows(rows[mid:])]
            ids = [i for h in halves for i in h[0]]
            parts = [h[1] for h in halves if h[0]]
            vectors = np.concatenate(parts) if parts else halves[0][1]
            return ids, vectors, {**halves[0][2], **halves[1][2]}
        self._record_usage(result.prompt_tokens, result.total_tokens, result.reported_model)
        dim = result.vectors.shape[1]
        if self.expected_dim is None:
            self.expected_dim = dim
            self.state["dimension"] = dim
        if dim != self.expected_dim:
            raise InvalidEmbeddingResponseError(
                f"embedding dimension {dim} != expected {self.expected_dim}"
            )
        return [r["example_id"] for r in rows], result.vectors, {}

    def _record_usage(self, prompt: int | None, total: int | None, reported: str | None) -> None:
        usage = self.state["usage"]
        usage["requests"] += 1
        if prompt is None:
            usage["tokens_complete"] = False
        else:
            usage["prompt_tokens"] += prompt
            usage["total_tokens"] += total if total is not None else prompt
        if reported and reported not in self.state["reported_models"]:
            self.state["reported_models"].append(reported)
            if reported != self.client.model:
                log.warning("provider reported a different model name", reported=reported)

    def _assemble(self, split: str) -> None:
        """Stream parts into `<split>.npy` (memory-mapped) so peak memory is ~one batch."""
        final_npy = self.dir / f"{split}.npy"
        mapping_path = self.dir / f"{split}.row_mapping.jsonl"
        if final_npy.exists() and mapping_path.exists():
            return  # written atomically, so presence means complete
        state = self.state["splits"][split]
        parts = [self.dir / "parts" / split / f"{b:06d}.npz" for b in range(state["total_batches"])]
        ids: list[str] = []
        for path in parts:
            with np.load(path) as part:
                ids.extend(str(i) for i in part["ids"])
        expected_ids = [
            r["example_id"] for r in self.examples[split] if r["example_id"] not in state["failed"]
        ]
        if ids != expected_ids:
            raise EmbeddingJobError(f"{split}: embedded rows do not align with processed examples")

        tmp = final_npy.with_name(final_npy.name + ".tmp.npy")
        out = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float32, shape=(len(ids), self.expected_dim or 0)
        )
        row = 0
        for path in parts:
            with np.load(path) as part:
                vectors = part["vectors"]
                if not len(part["ids"]):
                    continue
                if vectors.shape[1] != self.expected_dim or not np.isfinite(vectors).all():
                    raise EmbeddingJobError(f"{split}: invalid vectors in {path.name}")
                out[row : row + len(vectors)] = vectors
                row += len(vectors)
        out.flush()
        del out  # release the memory map before renaming (required on Windows)
        if row != len(ids):
            raise EmbeddingJobError(f"{split}: wrote {row} rows, expected {len(ids)}")
        os.replace(tmp, final_npy)

        labels = {r["example_id"]: r for r in self.examples[split]}
        lines = [
            json.dumps(
                {
                    "row": i,
                    "example_id": eid,
                    "label": labels[eid]["label"],
                    "query_sha256": hashlib.sha256(
                        labels[eid][self.text_field].encode()
                    ).hexdigest()[:16],  # hash of the embedded text (any text_field)
                }
            )
            for i, eid in enumerate(ids)
        ]
        _atomic_write_bytes(mapping_path, ("\n".join(lines) + "\n").encode())
        log.info("assembled split", split=split, rows=len(ids), dimension=self.expected_dim)

    def _finish(self, *, complete: bool) -> dict[str, Any]:
        assembled = [s for s in SPLITS if (self.dir / f"{s}.npy").exists()]
        usage = self.state["usage"]
        price = self.cfg.embeddings.price_per_million_tokens
        metadata = {
            "cache_key": self.key,
            "embedding_model": self.client.model,
            "reported_models": self.state["reported_models"],
            "dimension": self.expected_dim,
            "dimensions_parameter": self.cfg.embeddings.dimensions,
            "created_at": self.state["created_at"],
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "processed_fingerprint": self.manifest["processed_fingerprint"],
            "raw_content_fingerprint": self.manifest["raw_content_fingerprint"],
            "preprocessing_version": self.manifest["preprocessing_version"],
            "batch_size": self.batch_size,
            "assembled_splits": assembled,
            "complete": complete and len(assembled) == len(SPLITS),
            "rows": {
                s: len(self.examples[s]) - len(self.state["splits"][s]["failed"]) for s in SPLITS
            },
            "failed_examples": {s: self.state["splits"][s]["failed"] for s in SPLITS},
            "usage": usage,
            "embedding_seconds": round(self.state["embedding_seconds"], 2),
            "estimated_cost_usd": (
                round(usage["prompt_tokens"] / 1e6 * price, 4)
                if price is not None and usage["tokens_complete"]
                else None
            ),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.dir / "metadata.json", metadata)
        if metadata["complete"]:
            _atomic_write_json(
                self.cfg.paths.embeddings_dir / "latest.json",
                {
                    "cache_key": self.key,
                    "embedding_model": self.client.model,
                    "dimension": self.expected_dim,
                },
            )
        return metadata


def _npz_bytes(vectors: np.ndarray, ids: list[str]) -> bytes:
    import io

    buffer = io.BytesIO()
    np.savez(buffer, vectors=vectors.astype(np.float32), ids=np.asarray(ids, dtype=str))
    return buffer.getvalue()


class _PlanOnlyClient:
    """Stands in for the API client during --plan; any embedding call is a bug."""

    def __init__(self, model: str) -> None:
        self.model = model

    def embed(self, texts: Sequence[str]) -> Any:
        raise EmbeddingJobError("planning must not call the embeddings API")


def resolve_embeddings_dir(cfg: ComplexityConfig, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    pointer = cfg.paths.embeddings_dir / "latest.json"
    if not pointer.exists():
        raise FileNotFoundError(f"{pointer} not found; generate embeddings first")
    key = str(json.loads(pointer.read_text(encoding="utf-8"))["cache_key"])
    return cfg.paths.embeddings_dir / key


def main(argv: list[str] | None = None) -> None:
    from llm_router.embeddings.fireworks_client import FireworksEmbeddingClient, FireworksSettings

    parser = argparse.ArgumentParser(description="Generate cached Fireworks embeddings")
    parser.add_argument("--config", default="configs/complexity.yaml")
    parser.add_argument("--plan", action="store_true", help="show remaining work; no API calls")
    parser.add_argument("--max-batches", type=int, default=None, help="stop after N new batches")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--yes", action="store_true", help="confirm a run larger than a pilot")
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    try:
        if args.plan:
            # Planning never calls the API, so it does not need the API key.
            from dotenv import load_dotenv

            load_dotenv(".env", override=False)
            model = os.environ.get("FIREWORKS_EMBEDDING_MODEL", "").strip() or "<unset>"
            client: EmbeddingClient = _PlanOnlyClient(model)
            job = EmbeddingJob(cfg, client, require_connectivity=False)
        else:
            settings = FireworksSettings.from_env()
            job = EmbeddingJob(cfg, FireworksEmbeddingClient(settings, cfg.embeddings))
        plan = job.plan()
        print(json.dumps(plan, indent=2))
        if args.plan:
            return
        planned = plan["remaining_requests"] if args.max_batches is None else args.max_batches
        if planned > PILOT_BATCH_LIMIT and not args.yes:
            raise SystemExit(
                f"{planned} requests planned; review the plan above and re-run with --yes, "
                f"or use --max-batches {PILOT_BATCH_LIMIT} for a pilot"
            )
        metadata = job.run(splits=args.splits, max_batches=args.max_batches)
    except (EmbeddingError, EmbeddingJobError) as exc:
        raise SystemExit(f"embedding generation stopped: {exc}") from None
    summary = {
        k: metadata[k]
        for k in (
            "cache_key",
            "embedding_model",
            "dimension",
            "complete",
            "usage",
            "embedding_seconds",
            "estimated_cost_usd",
        )
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
