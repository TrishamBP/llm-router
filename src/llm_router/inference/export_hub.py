"""Build the Hugging Face package of the complexity router (weights as plain NumPy arrays).

Usage: uv run python -m llm_router.inference.export_hub [--out artifacts/hub/llm-complexity-router]

Exports logistic regression (scaler + multinomial softmax) and MLP C (standardiser + 3 linear
layers) to `.npz` files loadable with `allow_pickle=False`, copies the runtime and Claude Code
plugin files from `packaging/hub/`, writes `router_config.json` and the model card, then
verifies parity: the exported NumPy forward pass must reproduce the original models'
probabilities on every cached test embedding (max abs diff and identical argmax).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from llm_router.config import LABELS, load_config
from llm_router.data.datasets import load_embedded_split

PACKAGING = Path("packaging/hub")
SOURCES = {
    "logistic_regression": Path("artifacts/models/logistic_regression"),
    "mlp": Path("artifacts/models/mlp_embeddings"),  # experiment C, complexity-only
}
REPORTS = {
    "logistic_regression": Path("artifacts/reports/comparison/test/logistic_regression.json"),
    "mlp": Path("artifacts/reports/comparison/test/mlp_embeddings.json"),
}
TIERS = {"easy": "haiku", "medium": "sonnet", "hard": "opus"}


def export_logistic_regression(src: Path, out: Path) -> dict[str, Any]:
    pipe = joblib.load(src / "model.joblib")
    encoder = joblib.load(src / "label_encoder.joblib")
    scaler, clf = pipe.named_steps["scale"], pipe.named_steps["clf"]
    labels = [str(encoder.classes_[int(i)]) for i in clf.classes_]  # column order of predict_proba
    np.savez(
        out,
        kind=np.array("logistic_regression"),
        labels=np.array(labels),
        mean=scaler.mean_.astype(np.float64),
        scale=scaler.scale_.astype(np.float64),
        coef=clf.coef_.astype(np.float64),
        intercept=clf.intercept_.astype(np.float64),
    )
    return {"labels": labels, "C": float(clf.C), "class_weight": clf.class_weight}


def export_mlp(src: Path, out: Path) -> dict[str, Any]:
    import torch

    ck = torch.load(src / "model.pt", map_location="cpu", weights_only=True)
    arch, sd = ck["architecture"], ck["state_dict"]
    if arch["hidden_dims"] != [256, 128] or set(sd) != {
        f"{i}.{p}" for i in (0, 3, 6) for p in ("weight", "bias")
    }:
        raise ValueError(f"unexpected MLP layout: {arch}, {sorted(sd)}")
    std = np.load(src / "standardizer.npz")
    meta = json.loads((src / "metadata.json").read_text(encoding="utf-8"))
    np.savez(
        out,
        kind=np.array("mlp"),
        labels=np.array(meta["probability_columns"]),
        mean=std["mean"].astype(np.float64),
        scale=std["scale"].astype(np.float64),
        **{
            f"{n}{j}": sd[f"{i}.{p}"].double().numpy()
            for j, i in enumerate((0, 3, 6))
            for p, n in (("weight", "w"), ("bias", "b"))
        },
    )
    return {"labels": meta["probability_columns"], "architecture": arch}


def original_proba(name: str, X: np.ndarray) -> np.ndarray:
    """Original model probabilities, reordered to LABELS."""
    if name == "logistic_regression":
        pipe = joblib.load(SOURCES[name] / "model.joblib")
        encoder = joblib.load(SOURCES[name] / "label_encoder.joblib")
        cols = [str(encoder.classes_[int(i)]) for i in pipe.named_steps["clf"].classes_]
        proba = pipe.predict_proba(X)
    else:
        from llm_router.models.complexity_classifiers import MLPBackend

        backend = MLPBackend.load(SOURCES[name], standardize=True)
        cols = json.loads((SOURCES[name] / "metadata.json").read_text(encoding="utf-8"))[
            "probability_columns"
        ]
        proba = backend.predict_proba(X)
    reordered: np.ndarray = proba[:, [cols.index(lab) for lab in LABELS]]
    return reordered


def parity(out_dir: Path, X: np.ndarray) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("hub_runtime", out_dir / "complexity_router.py")
    assert spec and spec.loader
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(runtime)
    config = json.loads((out_dir / "router_config.json").read_text(encoding="utf-8"))
    results = {}
    for name, file in config["classifiers"].items():
        exported = runtime._Classifier(out_dir / file).predict_proba(X)
        original = original_proba(name, X)
        results[name] = {
            "rows": int(X.shape[0]),
            "max_abs_diff": float(np.abs(exported - original).max()),
            "argmax_agreement": float((exported.argmax(1) == original.argmax(1)).mean()),
        }
        if results[name]["argmax_agreement"] != 1.0 or results[name]["max_abs_diff"] > 1e-4:
            raise AssertionError(f"parity failed for {name}: {results[name]}")
    return results


def build(out_dir: Path) -> dict[str, Any]:
    cfg = load_config("configs/complexity.yaml")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    shutil.copytree(
        PACKAGING, out_dir, ignore=shutil.ignore_patterns("__pycache__", "README.template.md")
    )
    (out_dir / "models").mkdir()
    details = {
        "logistic_regression": export_logistic_regression(
            SOURCES["logistic_regression"], out_dir / "models" / "logistic_regression.npz"
        ),
        "mlp": export_mlp(SOURCES["mlp"], out_dir / "models" / "mlp.npz"),
    }
    lr_meta = json.loads(
        (SOURCES["logistic_regression"] / "metadata.json").read_text(encoding="utf-8")
    )
    metrics = {}
    for name, path in REPORTS.items():
        t = json.loads(path.read_text(encoding="utf-8"))
        t = t.get("test", t)
        metrics[name] = {
            "test_rows": t["n"],
            "accuracy": round(t["accuracy"], 4),
            "macro_f1": round(t["macro_f1"], 4),
            "weighted_f1": round(t["weighted_f1"], 4),
            "hard_recall": round(t["hard_recall"], 4),
            "hard_predicted_easy": t["hard_predicted_easy"]["count"],
            "per_class": {
                k: {m: round(v[m], 4) for m in ("precision", "recall", "f1")}
                for k, v in t["per_class"].items()
            },
        }
    config = {
        "name": "llm-complexity-router",
        "version": "0.1.0",
        "labels": list(LABELS),
        "embedding_model": lr_meta["embedding_model"],
        "embedding_dimension": lr_meta["embedding_dimension"],
        "embedding_provider": "Fireworks AI (OpenAI-compatible embeddings API)",
        "default_classifier": "logistic_regression",
        "classifiers": {
            "logistic_regression": "models/logistic_regression.npz",
            "mlp": "models/mlp.npz",
        },
        "recommended_models": TIERS,
        "training": {
            "dataset": "regolo/brick-complexity-extractor",
            "dataset_revision": cfg.dataset.revision,
            "dataset_license": "CC BY-NC 4.0",
            "split": "re-split 80/10/10, grouped by normalised prompt, seed 20261010",
            "train_rows": lr_meta["train_rows"],
            "details": details,
        },
        "test_metrics": metrics,
        "license": "CC-BY-NC-4.0",
    }
    (out_dir / "router_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    template = (PACKAGING / "README.template.md").read_text(encoding="utf-8")
    m_lr, m_mlp = metrics["logistic_regression"], metrics["mlp"]
    card = template
    for key, value in {
        "LR_ACC": m_lr["accuracy"],
        "LR_F1": m_lr["macro_f1"],
        "LR_HARD": m_lr["hard_recall"],
        "LR_H2E": m_lr["hard_predicted_easy"],
        "MLP_ACC": m_mlp["accuracy"],
        "MLP_F1": m_mlp["macro_f1"],
        "MLP_HARD": m_mlp["hard_recall"],
        "MLP_H2E": m_mlp["hard_predicted_easy"],
        "TEST_ROWS": m_lr["test_rows"],
        "TRAIN_ROWS": lr_meta["train_rows"],
        "REVISION": cfg.dataset.revision,
    }.items():
        card = card.replace("{{" + key + "}}", str(value))
    (out_dir / "README.md").write_text(card, encoding="utf-8")

    emb_dir = cfg.paths.embeddings_dir / lr_meta["embedding_cache_key"]
    X = np.asarray(load_embedded_split(emb_dir, "test", mmap=True).X)
    parity_result = parity(out_dir, X)
    shutil.rmtree(out_dir / "__pycache__", ignore_errors=True)  # left by the parity import
    report = {
        "out_dir": str(out_dir),
        "parity": parity_result,
        "files": sorted(
            str(p.relative_to(out_dir)).replace("\\", "/")
            for p in out_dir.rglob("*")
            if p.is_file()
        ),
        "size_mb": round(sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e6, 2),
    }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="artifacts/hub/llm-complexity-router")
    args = parser.parse_args(argv)
    print(json.dumps(build(Path(args.out)), indent=2))


if __name__ == "__main__":
    main()
