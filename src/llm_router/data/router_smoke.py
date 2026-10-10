"""End-to-end smoke test of the router-dataset pipeline on synthetic data.

Usage: uv run python -m llm_router.data.router_smoke [--keep DIR]

No credentials, no network: synthetic raw files and dataset card are written to a temp
workspace, then preparation, the connectivity check, an interrupted + resumed embedding job
(offline fake embeddings), and `load_router_dataset` validation all run. The data includes
duplicate prompts across IDs, a label conflict, an empty prompt, and an undeclared label.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any

from llm_router.config import EmbeddingConfig, sha256_json
from llm_router.data import prepare_router_dataset
from llm_router.data.download import file_sha256
from llm_router.data.router_dataset import (
    RouterDatasetConfig,
    RouterPathsConfig,
    RouterSourceConfig,
    load_router_dataset,
)
from llm_router.embeddings.fake import FakeEmbeddingClient
from llm_router.embeddings.generate_router_embeddings import make_job, run_connectivity
from llm_router.logging import configure_logging

CARD = """---
license: apache-2.0
dataset_info:
  features:
    - name: id
      dtype: string
    - name: prompt
      dtype: string
    - name: label
      dtype:
        class_label:
          names:
            '0': small_llm
            '1': large_llm
---
Synthetic smoke-test card.
"""

SMALL = ["What is the capital of {t}?", "Spell {t}.", "Name a fact about {t}."]
LARGE = [
    "Derive a rigorous proof about {t} and analyse its edge cases.",
    "Design a distributed system for {t} under strict latency constraints.",
]
TOPICS = ["france", "physics", "genetics", "databases", "music", "rivers", "markets"]


def synthetic_rows(n_ids: int = 40, seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for g in range(n_ids):
        label = g % 2
        templates = LARGE if label else SMALL
        split = "test" if g % 4 == 0 else "train"
        for j in range(rng.randint(2, 6)):
            prompt = rng.choice(templates).format(t=rng.choice(TOPICS)) + f" ({g}-{j})"
            rows[split].append({"id": f"id-{g:03d}", "prompt": prompt, "label": label})
    stock = "What is the capital of France?"
    for g in (1, 3, 5):  # stock prompt reused under several small_llm IDs, in both splits
        rows["train"].append({"id": f"id-{g - 1:03d}", "prompt": stock, "label": 0})
    rows["test"].append({"id": "id-004", "prompt": stock + " ", "label": 0})
    rows["train"] += [
        {"id": "id-900", "prompt": "Is this conflicting?", "label": 0},
        {"id": "id-901", "prompt": "is this conflicting", "label": 1},  # conflict
        {"id": "id-902", "prompt": "   ", "label": 0},  # empty
        {"id": "id-903", "prompt": "Undeclared label row", "label": 7},  # invalid label
    ]
    return rows


def build_workspace(
    root: Path, *, n_ids: int = 40, extra_train_rows: list[dict[str, Any]] | None = None
) -> RouterDatasetConfig:
    revision = "smoke-rev-0001"
    raw = root / "data" / "raw" / revision
    raw.mkdir(parents=True)
    files = {}
    data = synthetic_rows(n_ids)
    data["train"] += extra_train_rows or []
    for split, rows in data.items():
        path = raw / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        files[split] = {
            "repo_path": path.name,
            "local_path": path.name,
            "sha256": file_sha256(path),
        }
    (raw / "README.md").write_text(CARD, encoding="utf-8")
    provenance = {
        "repo_id": "synthetic/router-smoke",
        "revision": revision,
        "url": "synthetic",
        "license": "apache-2.0",
        "downloaded_at": "n/a",
        "files": files,
        "content_fingerprint": sha256_json({s: f["sha256"] for s, f in files.items()})[:16],
    }
    (raw / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return RouterDatasetConfig(
        dataset=RouterSourceConfig(
            repo_id="synthetic/router-smoke",
            revision=revision,
            data_files={s: f"{s}.jsonl" for s in files},
            license="apache-2.0",
        ),
        embeddings=EmbeddingConfig(batch_size=8),
        paths=RouterPathsConfig(
            raw_dir=root / "data" / "raw",
            processed_dir=root / "data" / "processed" / "router_dataset",
            embeddings_dir=root / "artifacts" / "embeddings" / "router_dataset",
            reports_dir=root / "artifacts" / "reports",
        ),
    )


def run_smoke(root: Path) -> dict[str, Any]:
    cfg = build_workspace(root)
    prepared = prepare_router_dataset.run(cfg, save_hf=False)
    client = FakeEmbeddingClient()
    run_connectivity(cfg, client)
    first = make_job(cfg, client).run(max_batches=2)  # simulated interruption
    calls = len(client.calls)
    final = make_job(cfg, client).run()
    data = load_router_dataset(config=cfg)
    return {
        "workspace": str(root),
        "removed": prepared.metadata["removed"],
        "splits": {k: v["rows"] for k, v in prepared.metadata["splits"].items()},
        "isolation": prepared.metadata["split_isolation"]["verified"],
        "interrupted_complete": first["complete"],
        "resumed_complete": final["complete"],
        "resume_requests": len(client.calls) - calls,
        "loader_rows": data.checks["rows"],
        "label_names": list(data.label_names),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keep", default=None)
    args = parser.parse_args(argv)
    configure_logging("WARNING")
    if args.keep:
        print(json.dumps(run_smoke(Path(args.keep).resolve()), indent=2))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            print(json.dumps(run_smoke(Path(tmp)), indent=2))


if __name__ == "__main__":
    main()
