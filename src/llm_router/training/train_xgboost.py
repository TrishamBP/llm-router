"""Train the XGBoost multiclass complexity classifier on cached features (CPU or CUDA GPU).

Usage:
    uv run python -m llm_router.training.train_xgboost --smoke              # quick check
    uv run python -m llm_router.training.train_xgboost [--feature-set embeddings]
        [--threads 2] [--max-seconds 3600] [--seed 42] [--output-name NAME]
        [--device auto|cpu|cuda]

Equivalent to XGBClassifier(objective="multi:softprob", num_class=3, tree_method="hist",
eval_metric="mlogloss", ...) with settings from the `xgboost:` config section, but uses the
native API so training data can be streamed into a QuantileDMatrix in row chunks from the
memory-mapped embeddings (no full float copy of the feature matrix is built).

Early stopping watches validation mlogloss; the class-weight setting (`none`, and `balanced`
if the training imbalance justifies it) is selected by validation macro-F1, tie-break hard
recall. The best booster is trimmed to its best iteration and saved to
`paths.models_dir/<output-name>/`. The test split is never loaded here.

With `resources.device` cuda/auto, `tree_method="hist"` runs on the GPU: the quantised
training matrix (about 1 byte per value) is what lives in VRAM, not the float32 embeddings.
The saved booster predicts on CPU (see XGBoostBackend), so GPU-trained models stay portable.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from llm_router.config import LABELS, ComplexityConfig, load_config
from llm_router.features.complexity_features import (
    FEATURE_SETS,
    FeatureMatrix,
    FeatureSet,
    build_schema,
)
from llm_router.logging import configure_logging, get_logger
from llm_router.models.complexity_classifiers import XGBoostBackend
from llm_router.training import complexity_common as common
from llm_router.training.resources import (
    GIB,
    BudgetMonitor,
    directory_size_bytes,
    limit_threads,
    resolve_device,
)

log = get_logger(__name__)


class ResourceBudgetError(RuntimeError):
    pass


def default_output_name(feature_set: str, smoke: bool = False) -> str:
    return f"xgboost_{feature_set.replace('+', '_')}" + ("_smoke" if smoke else "")


def with_overrides(
    cfg: ComplexityConfig,
    *,
    threads: int | None = None,
    max_seconds: float | None = None,
    seed: int | None = None,
    memory_budget_gb: float | None = None,
    device: str | None = None,
) -> ComplexityConfig:
    """CLI overrides of the `resources` section. `max_seconds <= 0` disables the time limit."""
    update: dict[str, Any] = {}
    if threads is not None:
        update["threads"] = threads
    if max_seconds is not None:
        update["max_train_seconds"] = max_seconds if max_seconds > 0 else None
    if seed is not None:
        update["seed"] = seed
    if memory_budget_gb is not None:
        update["memory_budget_gb"] = memory_budget_gb if memory_budget_gb > 0 else None
    if device is not None:
        update["device"] = device
    return cfg.model_copy(update={"resources": cfg.resources.model_copy(update=update)})


def _xgb_params(cfg: ComplexityConfig, device: str = "cpu") -> dict[str, Any]:
    x = cfg.xgboost
    return {
        "objective": "multi:softprob",
        "num_class": len(LABELS),
        "max_depth": x.max_depth,
        "eta": x.learning_rate,
        "subsample": x.subsample,
        "colsample_bytree": x.colsample_bytree,
        "min_child_weight": x.min_child_weight,
        "lambda": x.reg_lambda,
        "max_bin": x.max_bin,
        "tree_method": "hist",
        "device": device,
        "eval_metric": "mlogloss",
        "nthread": cfg.resources.threads,
        "seed": cfg.resources.seed,
    }


def _quantile_matrix(
    matrix: FeatureMatrix, y: np.ndarray, chunk_rows: int, max_bin: int, ref: Any = None
) -> Any:
    import xgboost as xgb

    class ChunkIter(xgb.DataIter):
        def __init__(self) -> None:
            self._pos = 0
            super().__init__()

        def next(self, input_data: Any) -> bool:
            if self._pos >= matrix.n_rows:
                return False
            stop = min(self._pos + chunk_rows, matrix.n_rows)
            input_data(data=matrix.rows(self._pos, stop), label=y[self._pos : stop])
            self._pos = stop
            return True

        def reset(self) -> None:
            self._pos = 0

    return xgb.QuantileDMatrix(ChunkIter(), max_bin=max_bin, ref=ref)


def _predict(backend: XGBoostBackend, matrix: FeatureMatrix, chunk_rows: int) -> np.ndarray:
    return np.concatenate([backend.predict_proba(b) for b in matrix.iter_batches(chunk_rows)])


def _memory_estimate_gib(
    train_rows: int, val_rows: int, n_features: int, cfg: ComplexityConfig
) -> float:
    bytes_per_bin = 1 if cfg.xgboost.max_bin <= 256 else 2
    quantised = (train_rows + val_rows) * n_features * bytes_per_bin
    chunk = cfg.xgboost.chunk_rows * n_features * 4 * 3  # a few chunk copies in flight
    return (quantised + chunk) / GIB


def train_xgboost(
    cfg: ComplexityConfig,
    *,
    feature_set: FeatureSet = "embeddings",
    embeddings_dir: Path | None = None,
    output_name: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    import xgboost as xgb

    res, xcfg = cfg.resources, cfg.xgboost
    limit_threads(res.threads)
    device, device_note = resolve_device(res.device, "xgboost")
    log.info("xgboost device", device=device, note=device_note)
    common.seed_everything(res.seed)
    monitor = BudgetMonitor(res.max_train_seconds, res.memory_budget_gb)
    output_name = output_name or default_output_name(feature_set, smoke)
    train, val, emb_meta, emb_dir = common.prepare_training_data(
        cfg, feature_set, embeddings_dir=embeddings_dir, smoke=smoke, smoke_rows=xcfg.smoke_rows
    )
    schema = build_schema(feature_set, emb_meta)
    n_features = train.matrix.n_features
    estimate = _memory_estimate_gib(train.matrix.n_rows, val.matrix.n_rows, n_features, cfg)
    log.info(
        "xgboost data",
        feature_set=feature_set,
        train_rows=train.matrix.n_rows,
        val_rows=val.matrix.n_rows,
        features=n_features,
        est_extra_gib=round(estimate, 2),
    )
    if res.memory_budget_gb is not None and estimate > res.memory_budget_gb:
        raise ResourceBudgetError(
            f"estimated {estimate:.1f} GiB for quantised data exceeds the "
            f"{res.memory_budget_gb} GiB budget; lower xgboost.max_bin or use fewer features"
        )

    dtrain = _quantile_matrix(train.matrix, train.y, xcfg.chunk_rows, xcfg.max_bin)
    dval = _quantile_matrix(val.matrix, val.y, xcfg.chunk_rows, xcfg.max_bin, ref=dtrain)
    params = _xgb_params(cfg, device)
    rounds = xcfg.smoke_estimators if smoke else xcfg.n_estimators
    weight_options, weight_plan = common.class_weight_plan(
        train.y, xcfg.class_weights, res.imbalance_threshold
    )
    class_weights = common.balanced_class_weights(train.y)
    progress_path = cfg.paths.reports_dir / "training" / f"{output_name}.json"

    variants: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], Any, dict[str, Any], dict[str, Any]] | None = None
    for weighting in weight_options:
        if monitor.exceeded():
            log.warning("budget exhausted before variant", class_weight=weighting)
            break
        row_weights = (
            class_weights[train.y] if weighting == "balanced" else np.ones(len(train.y), np.float32)
        )
        dtrain.set_weight(row_weights)

        class Budget(xgb.callback.TrainingCallback):
            reason: str | None = None

            def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
                self.reason = monitor.exceeded()
                return self.reason is not None

        budget = Budget()
        callbacks: list[Any] = [budget]
        if xcfg.early_stopping_rounds:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=xcfg.early_stopping_rounds,
                    metric_name="mlogloss",
                    data_name="validation",
                    save_best=True,
                )
            )
        evals_result: dict[str, Any] = {}
        started = time.perf_counter()
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            evals=[(dval, "validation")],
            evals_result=evals_result,
            callbacks=callbacks,
            verbose_eval=False,
        )
        fit_s = time.perf_counter() - started
        curve = [float(v) for v in evals_result["validation"]["mlogloss"]]
        best_iteration = int(np.argmin(curve)) if xcfg.early_stopping_rounds else len(curve) - 1
        backend = XGBoostBackend(booster, threads=res.threads)
        metrics = common.evaluate_proba(val.y, _predict(backend, val.matrix, xcfg.chunk_rows))
        variant = {
            "class_weight": weighting,
            "rounds_trained": len(curve),
            "best_iteration": best_iteration,
            "trees_kept": int(booster.num_boosted_rounds()),
            "stopped_reason": budget.reason,
            "fit_seconds": round(fit_s, 2),
            "validation": common.brief(metrics),
            "validation_mlogloss_curve": curve,
        }
        variants.append(variant)
        common.write_json(progress_path, {"output_name": output_name, "variants": variants})
        log.info(
            "xgboost variant",
            class_weight=weighting,
            rounds=len(curve),
            best_iteration=best_iteration,
            val_macro_f1=round(metrics["macro_f1"], 4),
            seconds=round(fit_s, 1),
            stopped=budget.reason,
        )
        if best is None or common.selection_key(metrics) > best[0]:
            best = (common.selection_key(metrics), backend, variant, metrics)
        del booster, backend
        gc.collect()
    if best is None:
        raise ResourceBudgetError("no XGBoost variant finished within the resource budget")

    _, backend, chosen, val_metrics = best
    del dtrain, dval
    gc.collect()
    out_dir = cfg.paths.models_dir / output_name
    staging = common.staging_dir(out_dir)
    backend.save(staging)
    metadata = {
        "classifier": "xgboost",
        "output_name": output_name,
        "feature_set": feature_set,
        "classes": list(LABELS),
        "probability_columns": list(LABELS),
        "supports_probabilities": True,
        "hyperparameters": {
            **{k: v for k, v in params.items() if k not in ("nthread",)},
            "n_estimators": rounds,
            "early_stopping_rounds": xcfg.early_stopping_rounds,
            "class_weight": chosen["class_weight"],
            "best_iteration": chosen["best_iteration"],
            "trees_kept": chosen["trees_kept"],
        },
        "selection": "early stopping on validation mlogloss; class weighting by max validation "
        "macro-F1, tie-break validation hard recall",
        "status": "stopped_by_budget" if chosen["stopped_reason"] else "complete",
        "stopped_reason": chosen["stopped_reason"],
        "class_weighting": weight_plan,
        "train_class_distribution": common.class_distribution(train.y),
        "validation_class_distribution": common.class_distribution(val.y),
        "validation_metrics": val_metrics,
        "validation_calibration": val_metrics["calibration"],
        "variants": variants,
        "train_rows": train.matrix.n_rows,
        "validation_rows": val.matrix.n_rows,
        "n_features": n_features,
        "training_seconds": round(sum(v["fit_seconds"] for v in variants), 2),
        "threads": res.threads,
        "training_device": device,
        "training_device_note": device_note,
        "seed": res.seed,
        "smoke": smoke,
        **common.provenance(emb_meta),
        "embeddings_dir": str(emb_dir),
        "created_at": common.now(),
        "environment": common.environment_info(
            {"xgboost_build": f"device={device}, hist; inference on cpu"}
        ),
    }
    common.write_json(staging / "feature_schema.json", schema)
    metadata["resources"] = monitor.report()
    common.write_json(staging / "metadata.json", metadata)
    common.replace_directory(staging, out_dir)
    metadata["model_dir"] = str(out_dir)
    metadata["artifact_bytes"] = directory_size_bytes(out_dir)
    common.write_json(
        progress_path,
        {
            "output_name": output_name,
            "variants": variants,
            "model_dir": str(out_dir),
            "resources": metadata["resources"],
        },
    )
    return metadata


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/complexity.yaml")
    parser.add_argument("--feature-set", default="embeddings", choices=FEATURE_SETS)
    parser.add_argument("--embeddings-dir", default=None)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--smoke", action="store_true", help="small subsample, few iterations")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None, help="<= 0: no limit")
    parser.add_argument("--memory-budget-gb", type=float, default=None, help="<= 0: no limit")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--device", default=None, choices=["auto", "cpu", "cuda"], help="override resources.device"
    )
    parser.add_argument("--result-json", default=None, help="also write the summary here")


def load_cli_config(args: argparse.Namespace) -> ComplexityConfig:
    return with_overrides(
        load_config(args.config),
        threads=args.threads,
        max_seconds=args.max_seconds,
        seed=args.seed,
        memory_budget_gb=args.memory_budget_gb,
        device=args.device,
    )


def summarise(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "classifier": metadata["classifier"],
        "model_dir": metadata["model_dir"],
        "feature_set": metadata["feature_set"],
        "status": metadata["status"],
        "hyperparameters": metadata["hyperparameters"],
        "validation": common.brief(metadata["validation_metrics"]),
        "training_seconds": metadata["training_seconds"],
        "resources": metadata["resources"],
        "artifact_bytes": metadata["artifact_bytes"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    args = parser.parse_args(argv)
    configure_logging()
    metadata = train_xgboost(
        load_cli_config(args),
        feature_set=args.feature_set,
        embeddings_dir=Path(args.embeddings_dir) if args.embeddings_dir else None,
        output_name=args.output_name,
        smoke=args.smoke,
    )
    summary = summarise(metadata)
    if args.result_json:
        common.write_json(Path(args.result_json), summary)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
