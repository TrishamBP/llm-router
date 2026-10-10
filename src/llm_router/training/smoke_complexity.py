"""End-to-end smoke test on synthetic data with a fake embedding client.

Usage: uv run python -m llm_router.training.smoke_complexity [--keep DIR]

Needs no credentials and makes no network calls. Exercises preprocessing (including a
duplicate, a label conflict, an empty query, and an invalid label), the connectivity test,
cached/resumable embedding generation, training, test evaluation, and inference.
Metrics from this run are meaningless; it only checks that the pipeline works.
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any

from llm_router.config import ComplexityConfig, DatasetConfig, PathsConfig, sha256_json
from llm_router.data import preprocess
from llm_router.data.download import file_sha256
from llm_router.embeddings.fake import FakeEmbeddingClient
from llm_router.embeddings.fireworks_client import connectivity_test
from llm_router.embeddings.generate import EmbeddingJob
from llm_router.inference.classifier import ComplexityRouter
from llm_router.logging import configure_logging
from llm_router.training.evaluate_complexity import evaluate
from llm_router.training.train_complexity import train

TOPICS = ["python", "databases", "networking", "statistics", "biology", "history", "physics"]
TEMPLATES = {
    "easy": ["What is {t}?", "Define {t} in one sentence.", "Who invented {t}?", "Spell {t}."],
    "medium": [
        "Compare two approaches in {t} and explain the tradeoffs.",
        "Explain step by step how {t} works with an example.",
        "Summarize the main debates in {t} and analyse them.",
    ],
    "hard": [
        "Design a provably optimal distributed system for {t} under adversarial constraints.",
        "Prove a rigorous theorem about {t} and derive tight asymptotic bounds.",
        "Architect a fault tolerant multi region {t} platform minimizing latency and cost.",
    ],
}


def synthetic_rows(n_per_label: int = 60, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for label, templates in TEMPLATES.items():
        for i in range(n_per_label):
            text = rng.choice(templates).format(t=rng.choice(TOPICS)) + f" (case {label}-{i})"
            rows.append({"text": text, "label": label})
    rows += [
        {"text": rows[0]["text"], "label": rows[0]["label"]},  # duplicate
        {"text": "Ambiguous conflicting query", "label": "easy"},
        {"text": "ambiguous conflicting query?", "label": "hard"},  # conflict
        {"text": "   ", "label": "easy"},  # empty
        {"text": "Unknown label row", "label": "trivial"},  # invalid label
    ]
    rng.shuffle(rows)
    return rows


def build_workspace(root: Path) -> ComplexityConfig:
    revision = "smoke000"
    raw = root / "data" / "raw" / revision
    raw.mkdir(parents=True)
    rows = synthetic_rows()
    cut = int(len(rows) * 0.8)
    files = {}
    for split, part in (("train", rows[:cut]), ("validation", rows[cut:])):
        path = raw / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in part), encoding="utf-8")
        files[split] = {
            "repo_path": path.name,
            "local_path": path.name,
            "sha256": file_sha256(path),
        }
    provenance = {
        "repo_id": "synthetic/smoke",
        "revision": revision,
        "license": "synthetic",
        "downloaded_at": "n/a",
        "files": files,
        "content_fingerprint": sha256_json({s: f["sha256"] for s, f in files.items()})[:16],
    }
    (raw / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return ComplexityConfig(
        dataset=DatasetConfig(
            repo_id="synthetic/smoke",
            revision=revision,
            data_files={s: f"{s}.jsonl" for s in files},
            license="synthetic",
        ),
        paths=PathsConfig(
            raw_dir=root / "data" / "raw",
            processed_dir=root / "data" / "processed",
            embeddings_dir=root / "artifacts" / "embeddings",
            models_dir=root / "artifacts" / "models",
            reports_dir=root / "artifacts" / "reports",
        ),
    )


def run_smoke(root: Path) -> dict[str, Any]:
    cfg = build_workspace(root)
    cfg = cfg.model_copy(
        update={"embeddings": cfg.embeddings.model_copy(update={"batch_size": 16})}
    )
    manifest = preprocess.run(cfg).manifest
    client = FakeEmbeddingClient()
    connectivity = connectivity_test(
        client, cfg.embeddings.connectivity_query, cfg.paths.embeddings_dir / "connectivity"
    )
    first = EmbeddingJob(cfg, client).run(max_batches=3)  # simulate an interrupted job
    calls_before = len(client.calls)
    final = EmbeddingJob(cfg, client).run()  # resume
    resumed_calls = len(client.calls) - calls_before
    summary = train(cfg, kinds=("logreg", "svm"))
    model_dir = Path(summary["classifiers"]["logreg"]["model_dir"])
    metrics = evaluate(cfg, model_dir, latency_samples=20)
    router = ComplexityRouter.from_pretrained(model_dir, embedding_client=client)
    prediction = router.predict("Design a provably optimal distributed system for physics.")
    return {
        "workspace": str(root),
        "removed": manifest["removed"],
        "splits": {k: v["rows"] for k, v in manifest["splits"].items()},
        "connectivity_dimension": connectivity["dimension"],
        "interrupted_after_batches": 3,
        "complete_after_resume": final["complete"] and not first["complete"],
        "resume_requests": resumed_calls,
        "test_macro_f1": round(metrics["test"]["macro_f1"], 4),
        "example_prediction": prediction.to_dict(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--keep", default=None, help="write the workspace here instead of a temp dir"
    )
    args = parser.parse_args(argv)
    configure_logging("WARNING")
    if args.keep:
        result = run_smoke(Path(args.keep).resolve())
    else:
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smoke(Path(tmp))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
