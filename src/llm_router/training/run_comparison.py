"""Train and compare the complexity classifiers one at a time, then evaluate on test once.

Usage:
    uv run python -m llm_router.training.run_comparison --smoke        # quick end-to-end check
    uv run python -m llm_router.training.run_comparison [--threads 2] [--max-seconds 3600]
        [--seed 42] [--only B C] [--baseline auto|train|skip] [--skip-near-duplicate-audit]

| ID | Experiment                        | Classifier          | Inputs                  |
|----|-----------------------------------|---------------------|-------------------------|
| A  | logistic_regression (baseline)    | logistic regression | embeddings              |
| B  | xgboost_embeddings                | XGBoost             | embeddings              |
| C  | mlp_embeddings                    | MLP                 | embeddings              |
| D  | xgboost_embeddings_engineered     | XGBoost             | embeddings + engineered |
| E  | mlp_embeddings_engineered         | MLP                 | embeddings + engineered |
| F1 | xgboost_engineered (ablation)     | XGBoost             | engineered only         |
| F2 | mlp_engineered (ablation)         | MLP                 | engineered only         |

Stages: (1) verify the embedding cache and split isolation, plus an embedding-space
near-duplicate audit; (2) baseline A (reused if a compatible model exists); (3) XGBoost jobs;
(4) MLP jobs; (5) per family, pick the embeddings vs embeddings+engineered variant by
validation macro-F1 and copy it to `models/xgboost` and `models/mlp`; (6) evaluate every
trained model on the test split, once; (7) write comparison.json and comparison.md.

Each job runs in its own child process, sequentially (never concurrently), with the thread
budget in its environment. That bounds and isolates memory (per-job peak RSS is measured by
the child), frees everything between jobs, and lets the runner record a failure and continue.
Outputs: `paths.reports_dir/comparison/` (`comparison_smoke/` for --smoke).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_router.config import LABELS, ComplexityConfig, load_config
from llm_router.data.leakage import near_duplicate_audit, verify_embedding_artifacts
from llm_router.embeddings.generate import resolve_embeddings_dir
from llm_router.features.complexity_features import FeatureSet
from llm_router.logging import configure_logging, get_logger
from llm_router.training import complexity_common as common
from llm_router.training.resources import thread_env
from llm_router.training.train_xgboost import with_overrides

log = get_logger(__name__)


@dataclass(frozen=True)
class Experiment:
    id: str
    name: str
    family: str  # logistic_regression | xgboost | mlp
    feature_set: FeatureSet
    description: str


EXPERIMENTS = (
    Experiment("A", "logistic_regression", "logistic_regression", "embeddings", "Baseline A"),
    Experiment("B", "xgboost_embeddings", "xgboost", "embeddings", "XGBoost B"),
    Experiment("C", "mlp_embeddings", "mlp", "embeddings", "MLP C"),
    Experiment(
        "D", "xgboost_embeddings_engineered", "xgboost", "embeddings+engineered", "XGBoost D"
    ),
    Experiment("E", "mlp_embeddings_engineered", "mlp", "embeddings+engineered", "MLP E"),
    Experiment("F1", "xgboost_engineered", "xgboost", "engineered", "XGBoost, engineered only"),
    Experiment("F2", "mlp_engineered", "mlp", "engineered", "MLP, engineered only"),
)
TRAIN_ORDER = ("A", "B", "D", "F1", "C", "E", "F2")  # all XGBoost jobs, then all MLP jobs
MODULES = {
    "xgboost": "llm_router.training.train_xgboost",
    "mlp": "llm_router.training.train_mlp",
}
SELECTABLE_FEATURE_SETS = ("embeddings", "embeddings+engineered")

# In-process job runner (used by tests): (cfg, experiment, output_name, smoke) -> summary
InProcessRunner = Callable[[ComplexityConfig, Experiment, str, bool], dict[str, Any]]


def _in_process(
    cfg: ComplexityConfig, exp: Experiment, output_name: str, smoke: bool
) -> dict[str, Any]:
    if exp.family == "logistic_regression":
        from llm_router.training.train_complexity import train

        summary = train(cfg, kinds=("logreg",))
        return {"model_dir": summary["classifiers"]["logreg"]["model_dir"]}
    if exp.family == "xgboost":
        from llm_router.training.train_xgboost import summarise, train_xgboost

        return summarise(
            train_xgboost(cfg, feature_set=exp.feature_set, output_name=output_name, smoke=smoke)
        )
    from llm_router.training.train_mlp import train_mlp
    from llm_router.training.train_xgboost import summarise

    return summarise(
        train_mlp(cfg, feature_set=exp.feature_set, output_name=output_name, smoke=smoke)
    )


class ComparisonRunner:
    def __init__(
        self,
        cfg: ComplexityConfig,
        config_path: Path | None,
        *,
        smoke: bool = False,
        only: tuple[str, ...] | None = None,
        baseline: str = "auto",
        near_duplicate_audit: bool = True,
        in_process: InProcessRunner | None = None,
    ) -> None:
        if config_path is None and in_process is None:
            raise ValueError("child processes need a config file path")
        self.cfg = cfg
        self.config_path = config_path
        self.smoke = smoke
        self.only = only
        self.baseline = baseline
        self.audit = near_duplicate_audit
        self.in_process = in_process
        # Smoke runs never read the test split; they evaluate on validation instead.
        self.eval_split = "validation" if smoke else "test"
        self._flagged: set[str] = set()
        self.out = cfg.paths.reports_dir / ("comparison_smoke" if smoke else "comparison")
        self.report: dict[str, Any] = {
            "started_at": common.now(),
            "smoke": smoke,
            "resources": cfg.resources.model_dump(mode="json"),
            "xgboost_config": cfg.xgboost.model_dump(mode="json"),
            "mlp_config": cfg.mlp.model_dump(mode="json"),
            "experiments": {},
            "failures": [],
        }

    # ------------------------------------------------------------------ helpers
    def _save(self) -> None:
        common.write_json(self.out / "comparison.json", self.report)

    def _output_name(self, exp: Experiment) -> str:
        if exp.family == "logistic_regression":
            return "logistic_regression"
        return exp.name + ("_smoke" if self.smoke else "")

    def _selected(self, exp_id: str) -> bool:
        return self.only is None or exp_id in self.only

    def _run_child(self, exp: Experiment, output_name: str) -> dict[str, Any]:
        res = self.cfg.resources
        result_path = self.out / "training" / f"{output_name}.result.json"
        log_path = self.out / "logs" / f"{output_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if result_path.exists():
            result_path.unlink()
        if exp.family == "logistic_regression":
            cmd = [
                sys.executable,
                "-m",
                "llm_router.training.train_complexity",
                "--models",
                "logreg",
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                MODULES[exp.family],
                "--feature-set",
                exp.feature_set,
                "--output-name",
                output_name,
                "--threads",
                str(res.threads),
                "--seed",
                str(res.seed),
                "--result-json",
                str(result_path),
                "--max-seconds",
                str(res.max_train_seconds or 0),
                "--memory-budget-gb",
                str(res.memory_budget_gb or 0),
                "--device",
                res.device,
            ]
            if self.smoke:
                cmd.append("--smoke")
        cmd += ["--config", str(self.config_path)]
        env = {**os.environ, **thread_env(res.threads), "PYTHONHASHSEED": str(res.seed)}
        # Soft limits are enforced inside the child; this hard limit only catches hangs.
        timeout = res.max_train_seconds * 1.5 + 900 if res.max_train_seconds else None
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(
                cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=timeout, check=False
            )
        if proc.returncode != 0:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            raise RuntimeError(f"exit code {proc.returncode}; log {log_path}:\n" + "\n".join(tail))
        summary: dict[str, Any] = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else {"model_dir": str(self.cfg.paths.models_dir / output_name)}
        )
        summary["wall_seconds"] = round(time.monotonic() - started, 2)
        summary["log"] = str(log_path)
        return summary

    def _run_job(self, exp: Experiment) -> None:
        output_name = self._output_name(exp)
        entry: dict[str, Any] = {
            "id": exp.id,
            "description": exp.description,
            "family": exp.family,
            "feature_set": exp.feature_set,
            "output_name": output_name,
        }
        log.info("job start", experiment=exp.id, name=output_name)
        try:
            if self.in_process is not None:
                summary = self.in_process(self.cfg, exp, output_name, self.smoke)
            else:
                summary = self._run_child(exp, output_name)
            entry.update(status="trained", training=summary)
            entry["model_dir"] = summary["model_dir"]
        except Exception as exc:  # record, continue with the remaining jobs
            message = f"{type(exc).__name__}: {exc}"
            log.error("job failed", experiment=exp.id, error=message.splitlines()[0])
            entry.update(status="failed", error=message)
            self.report["failures"].append({"stage": f"train {exp.id}", "error": message})
        self.report["experiments"][exp.id] = entry
        self._save()

    def _baseline(self, exp: Experiment, emb_key: str) -> None:
        """Reuse a compatible baseline A; train one only outside smoke runs."""
        model_dir = self.cfg.paths.models_dir / "logistic_regression"
        base = {
            "id": exp.id,
            "description": exp.description,
            "family": exp.family,
            "feature_set": exp.feature_set,
        }
        meta_path = model_dir / "metadata.json"
        if self.baseline != "train" and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("embedding_cache_key") == emb_key:
                self.report["experiments"][exp.id] = {
                    **base,
                    "status": "reused",
                    "model_dir": str(model_dir),
                    "training": {
                        "training_seconds": meta.get("training_seconds"),
                        "hyperparameters": meta.get("hyperparameters"),
                        "validation": common.brief(meta["validation_metrics"]),
                    },
                }
                self._save()
                return
        if self.baseline == "skip" or self.smoke:
            self.report["experiments"][exp.id] = {**base, "status": "skipped"}
            self._save()
            return
        self._run_job(exp)

    # ------------------------------------------------------------------ stages
    def verify(self) -> Path:
        emb_dir = resolve_embeddings_dir(self.cfg)
        self.report["data"] = verify_embedding_artifacts(self.cfg.paths.processed_dir, emb_dir)
        self._save()
        if self.audit:
            audit = near_duplicate_audit(emb_dir)
            common.write_json(self.out / "near_duplicates.json", audit)
            self.report["near_duplicates"] = {
                s: {
                    k: audit[s][k] for k in ("rows", "rows_at_or_above", "max_similarity_quantiles")
                }
                for s in ("validation", "test")
            }
            self.report["near_duplicates"]["flag_threshold"] = audit["flag_threshold"]
            self._flagged = {f["example_id"] for f in audit[self.eval_split]["flagged"]}
        self._save()
        return emb_dir

    def select(self) -> None:
        selected: dict[str, Any] = {}
        for family in ("xgboost", "mlp"):
            candidates = []
            for exp in EXPERIMENTS:
                entry = self.report["experiments"].get(exp.id, {})
                if exp.family != family or exp.feature_set not in SELECTABLE_FEATURE_SETS:
                    continue
                if entry.get("status") != "trained":
                    continue
                meta = json.loads(
                    (Path(entry["model_dir"]) / "metadata.json").read_text(encoding="utf-8")
                )
                key = common.selection_key(meta["validation_metrics"])
                candidates.append((key, exp.id, entry["model_dir"]))
            if not candidates:
                continue
            key, exp_id, model_dir = max(candidates)
            target = self.cfg.paths.models_dir / (family + ("_smoke" if self.smoke else ""))
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(model_dir, target)
            selected[family] = {
                "experiment": exp_id,
                "source": model_dir,
                "model_dir": str(target),
                "validation_macro_f1": key[0],
                "validation_hard_recall": key[1],
                "rule": "max validation macro-F1 (tie-break hard recall) among embeddings and "
                "embeddings+engineered; test metrics were not consulted",
            }
        self.report["selected"] = selected
        self._save()

    def evaluate(self) -> None:
        from llm_router.training.evaluate_classifiers import evaluate_many

        dirs = {
            exp_id: Path(e["model_dir"])
            for exp_id, e in self.report["experiments"].items()
            if e.get("status") in ("trained", "reused")
        }
        if not dirs:
            return
        split = self.eval_split
        out = self.out / split
        if self.in_process is not None:
            results = evaluate_many(
                self.cfg,
                list(dirs.values()),
                out,
                split=split,
                exclude_ids=self._flagged or None,
                latency_samples=50 if self.smoke else 500,
            )
        else:
            exclude_file = self.out / f"near_duplicate_{split}_ids.json"
            common.write_json(exclude_file, sorted(self._flagged))
            cmd = [
                sys.executable,
                "-m",
                "llm_router.training.evaluate_classifiers",
                "--config",
                str(self.config_path),
                "--split",
                split,
                "--out-dir",
                str(out),
                "--threads",
                str(self.cfg.resources.threads),
                "--latency-samples",
                "50" if self.smoke else "500",
                "--model-dirs",
                *map(str, dirs.values()),
            ]
            if self._flagged:
                cmd += ["--exclude-ids-file", str(exclude_file)]
            log_path = self.out / "logs" / f"evaluate_{split}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, **thread_env(self.cfg.resources.threads)}
            with log_path.open("w", encoding="utf-8") as fh:
                proc = subprocess.run(
                    cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, check=False
                )
            if proc.returncode != 0:
                self.report["failures"].append(
                    {
                        "stage": f"{split} evaluation",
                        "error": f"exit code {proc.returncode}; {log_path}",
                    }
                )
            results = {}
            for path in dirs.values():
                result_file = out / f"{path.name}.json"
                if result_file.exists():
                    results[path.name] = json.loads(result_file.read_text(encoding="utf-8"))
        for exp_id, path in dirs.items():
            result = results.get(path.name)
            if result is None:
                continue
            if "error" in result:
                self.report["failures"].append(
                    {"stage": f"{split} {exp_id}", "error": result["error"]}
                )
            self.report["experiments"][exp_id]["held_out_evaluation"] = result
        self.report["evaluation_split"] = split
        self.report["evaluated_at"] = common.now()
        self._save()

    def run(self) -> dict[str, Any]:
        try:
            emb_dir = self.verify()
        except Exception as exc:
            self.report["failures"].append(
                {"stage": "verify", "error": f"{type(exc).__name__}: {exc}"}
            )
            self._save()
            write_markdown(self.report, self.out / "comparison.md")
            raise
        emb_key = emb_dir.name
        by_id = {e.id: e for e in EXPERIMENTS}
        for exp_id in TRAIN_ORDER:
            if not self._selected(exp_id):
                continue
            exp = by_id[exp_id]
            if exp.family == "logistic_regression":
                self._baseline(exp, emb_key)
            else:
                self._run_job(exp)
        self.select()  # every configuration choice is made before the test split is read
        self.evaluate()
        self.report["finished_at"] = common.now()
        self._save()
        write_markdown(self.report, self.out / "comparison.md")
        return self.report


# ---------------------------------------------------------------------- report
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _peak(entry: dict[str, Any]) -> float | None:
    resources = (entry.get("training") or {}).get("resources") or {}
    return resources.get("peak_rss_gib")


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Complexity classifier comparison", ""]
    if report.get("smoke"):
        lines += [
            "> **Smoke run** on a small subsample with few iterations. These numbers say "
            "nothing about model quality.",
            "",
        ]
    data = report.get("data")
    if data:
        lines += [
            "## Data",
            "",
            f"- Embedding model: `{data['embedding_model']}` "
            f"(provider reported {data.get('reported_models')}), dimension {data['dimension']}",
            f"- Processed fingerprint `{data['processed_fingerprint']}`; "
            f"{data['total_rows']} rows; example IDs and duplicate groups disjoint across splits "
            f"({data['groups_checked']} groups checked)",
            f"- Training imbalance (max/min class count): {data['train_imbalance_max_min']}",
            "",
            "| Split | Rows | " + " | ".join(LABELS) + " |",
            "|---|---|" + "---|" * len(LABELS),
        ]
        for split, s in data["splits"].items():
            counts = " | ".join(str(s["label_counts"][label]) for label in LABELS)
            lines.append(f"| {split} | {s['rows']} | {counts} |")
        lines.append("")
    nd = report.get("near_duplicates")
    if nd:
        lines += [
            f"Embedding near-duplicates vs. train (cosine ≥ threshold): "
            f"validation {nd['validation']['rows_at_or_above']}, "
            f"test {nd['test']['rows_at_or_above']}. Test rows ≥ {nd['flag_threshold']} are "
            "excluded in the 'test excl. near-dup' column.",
            "",
        ]

    exps = report.get("experiments", {})
    lines += [
        "## Validation (used for all selection)",
        "",
        "| ID | Experiment | Features | Status | Accuracy | Macro F1 | Hard recall | Hard→easy | "
        "Log loss | Train s | Peak RSS GiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for exp_id, e in exps.items():
        val = (e.get("training") or {}).get("validation") or (
            (e.get("held_out_evaluation") or {}).get("validation_metrics") or {}
        )
        hpe = (val.get("hard_predicted_easy") or {}).get("rate_of_true_hard")
        lines.append(
            f"| {exp_id} | {e.get('description')} | {e.get('feature_set')} | {e.get('status')} | "
            f"{_fmt(val.get('accuracy'))} | {_fmt(val.get('macro_f1'))} | "
            f"{_fmt(val.get('hard_recall'))} | {_fmt(hpe)} | {_fmt(val.get('log_loss'))} | "
            f"{_fmt((e.get('training') or {}).get('training_seconds'), 1)} | {_fmt(_peak(e), 2)} |"
        )
    lines.append("")

    split = report.get("evaluation_split", "test")
    tested = {
        k: e["held_out_evaluation"]
        for k, e in exps.items()
        if split in e.get("held_out_evaluation", {})
    }
    if tested:
        title = (
            "Test (evaluated once, after selection)"
            if split == "test"
            else "Validation re-evaluation through the inference path (smoke run; test unused)"
        )
        lines += [
            f"## {title}",
            "",
            "| ID | Accuracy | Macro F1 | Weighted F1 | Hard recall | Hard→easy | Log loss | "
            "Excl. near-dup macro F1 | Latency median / p95 ms | Artifact MB |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for exp_id, r in tested.items():
            t = r[split]
            lat = r["local_inference_latency_ms"]
            nd_f1 = (r.get(f"{split}_excluding_near_duplicates") or {}).get("macro_f1")
            lines.append(
                f"| {exp_id} | {_fmt(t['accuracy'])} | {_fmt(t['macro_f1'])} | "
                f"{_fmt(t['weighted_f1'])} | {_fmt(t['hard_recall'])} | "
                f"{t['hard_predicted_easy']['count']} "
                f"({_fmt(t['hard_predicted_easy']['rate_of_true_hard'])}) | "
                f"{_fmt(t.get('log_loss'))} | {_fmt(nd_f1)} | "
                f"{_fmt(lat['median'], 3)} / {_fmt(lat['p95'], 3)} | "
                f"{r['artifact_bytes'] / 1e6:.2f} |"
            )
        lines += [
            "",
            f"### Per-class {split} metrics",
            "",
            "| ID | Class | Precision | Recall | F1 | Support |",
            "|---|---|---|---|---|---|",
        ]
        for exp_id, r in tested.items():
            for label, v in r[split]["per_class"].items():
                lines.append(
                    f"| {exp_id} | {label} | {_fmt(v['precision'])} | "
                    f"{_fmt(v['recall'])} | {_fmt(v['f1'])} | {v['support']} |"
                )
        lines += [
            "",
            f"### {split.capitalize()} confusion matrices (rows true, columns predicted: "
            + ", ".join(LABELS)
            + ")",
            "",
        ]
        for exp_id, r in tested.items():
            lines.append(f"- {exp_id}: `{r[split]['confusion_matrix']['rows_true_cols_pred']}`")
        lines.append("")
        deltas = []
        for with_eng, without in (("D", "B"), ("E", "C")):
            if with_eng in tested and without in tested:
                delta = tested[with_eng][split]["macro_f1"] - tested[without][split]["macro_f1"]
                deltas.append(
                    f"- {with_eng} − {without} (engineered features added): "
                    f"{split} macro-F1 {delta:+.4f}"
                )
        if deltas:
            lines += ["### Engineered-feature contribution", "", *deltas, ""]

    selected = report.get("selected") or {}
    if selected:
        lines += ["## Selected models", ""]
        for family, s in selected.items():
            lines.append(
                f"- {family}: experiment {s['experiment']} → `{s['model_dir']}` "
                f"(validation macro-F1 {s['validation_macro_f1']:.4f})"
            )
        lines.append("")
    budget = (report.get("resources") or {}).get("memory_budget_gb")
    peaks = [p for p in (_peak(e) for e in exps.values()) if p is not None]
    if peaks:
        lines += [
            f"Highest measured training peak RSS: {max(peaks):.2f} GiB (budget "
            f"{budget} GiB). RSS includes touched pages of memory-mapped embeddings.",
            "",
        ]
    if report.get("failures"):
        lines += ["## Failures", ""]
        lines += [f"- {f['stage']}: {f['error'].splitlines()[0]}" for f in report["failures"]]
        lines.append("")
    lines += [
        "Notes: labels come from an LLM judge; probabilities are not guaranteed to be calibrated "
        "(see log loss / ECE in comparison.json); latency excludes the embedding API call.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/complexity.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--only", nargs="+", default=None, choices=[e.id for e in EXPERIMENTS])
    parser.add_argument(
        "--baseline",
        default="auto",
        choices=["auto", "train", "skip"],
        help="auto: reuse a compatible logistic-regression model or train one",
    )
    parser.add_argument("--skip-near-duplicate-audit", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None, help="per job; <= 0: no limit")
    parser.add_argument("--memory-budget-gb", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    args = parser.parse_args(argv)
    configure_logging()
    cfg = with_overrides(
        load_config(args.config),
        threads=args.threads,
        max_seconds=args.max_seconds,
        seed=args.seed,
        memory_budget_gb=args.memory_budget_gb,
        device=args.device,
    )
    runner = ComparisonRunner(
        cfg,
        Path(args.config).resolve(),
        smoke=args.smoke,
        only=tuple(args.only) if args.only else None,
        baseline=args.baseline,
        near_duplicate_audit=not args.skip_near_duplicate_audit,
    )
    report = runner.run()
    print(f"report: {runner.out / 'comparison.md'}")
    print(
        json.dumps({"selected": report.get("selected"), "failures": report["failures"]}, indent=2)
    )
    if any(e.get("status") == "failed" for e in report["experiments"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
