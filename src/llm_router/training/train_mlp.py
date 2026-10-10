"""Train the lightweight PyTorch MLP complexity classifier on cached features (CPU or CUDA).

Usage:
    uv run python -m llm_router.training.train_mlp --smoke                  # quick check
    uv run python -m llm_router.training.train_mlp [--feature-set embeddings]
        [--threads 2] [--max-seconds 3600] [--seed 42] [--lr 1e-3] [--batch-size 128]
        [--device auto|cpu|cuda]

CrossEntropyLoss + AdamW, batch size 128, at most 30 epochs (config `mlp:` section). Inputs
are standardised with a mean/std fitted on the training split only. After each epoch the
validation macro-F1 is computed under `torch.no_grad()`; the best epoch (tie-break hard
recall) is kept, and training stops after `patience` epochs without improvement. Class
weighting (`balanced`) is evaluated only if the training imbalance justifies it, and the
better setting by validation macro-F1 is saved. Only the best checkpoint is written, to
`paths.models_dir/<output-name>/`. The test split is never loaded here.

GPU (`resources.device` cuda/auto): only the model and the current mini-batch live in VRAM;
features stay in host memory (memory-mapped) and each batch is copied over, so the 1 GB
training matrix never has to fit on a small (e.g. 4 GB) GPU. Checkpoints are kept and saved
on CPU. Peak VRAM is recorded in the metadata. GPU results are not bit-identical to CPU ones.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from llm_router.config import LABELS, ComplexityConfig
from llm_router.features.complexity_features import (
    FeatureSet,
    SplitFeatures,
    Standardizer,
    build_schema,
)
from llm_router.logging import configure_logging, get_logger
from llm_router.models.complexity_classifiers import MLPBackend
from llm_router.training import complexity_common as common
from llm_router.training.resources import (
    BudgetMonitor,
    directory_size_bytes,
    limit_threads,
    resolve_device,
)
from llm_router.training.train_xgboost import (
    ResourceBudgetError,
    add_common_args,
    load_cli_config,
    summarise,
)

log = get_logger(__name__)


def default_output_name(feature_set: str, smoke: bool = False) -> str:
    return f"mlp_{feature_set.replace('+', '_')}" + ("_smoke" if smoke else "")


def seed_torch(seed: int) -> None:
    # Needed for deterministic cuBLAS; must be set before CUDA is initialised.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    common.seed_everything(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class BatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):  # type: ignore[misc,unused-ignore]
    """torch Dataset whose items are whole batches: `ds[list_of_rows] -> (X, y)`.

    Used with a BatchSampler so a batch is gathered from the memory map in one sorted read,
    standardised, and converted to float32 tensors; nothing else is materialised.
    """

    def __init__(self, split: SplitFeatures, standardizer: Standardizer | None) -> None:
        self.split = split
        self.standardizer = standardizer

    def __len__(self) -> int:
        return self.split.matrix.n_rows

    def __getitem__(self, rows: list[int]) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override,unused-ignore]
        idx = np.asarray(rows, dtype=np.int64)
        X = self.split.matrix.take(idx)
        if self.standardizer is not None:
            X = self.standardizer.transform(X)
        return torch.from_numpy(np.ascontiguousarray(X)), torch.from_numpy(
            self.split.y[idx].astype(np.int64)
        )


def _loader(dataset: BatchDataset, batch_size: int, *, shuffle: bool, seed: int) -> Any:
    from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

    sampler: Any
    if shuffle:
        sampler = RandomSampler(range(len(dataset)), generator=torch.Generator().manual_seed(seed))
    else:
        sampler = SequentialSampler(range(len(dataset)))
    batches = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    # batch_size=None: the dataset already returns whole batches. num_workers=0 keeps a
    # single process (no forked copies of the data).
    return DataLoader(dataset, sampler=batches, batch_size=None, num_workers=0)  # type: ignore[arg-type,unused-ignore]


def _validation_proba(model: Any, loader: Any, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    with torch.no_grad():
        for X, _ in loader:
            logits = model(X.to(device, non_blocking=True))
            chunks.append(torch.softmax(logits.double(), dim=1).cpu().numpy())
    return np.concatenate(chunks)


def _cpu_state(model: Any) -> dict[str, Any]:
    """Snapshot of the weights in host memory (keeps checkpoints off the GPU)."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _fit_variant(
    cfg: ComplexityConfig,
    train: SplitFeatures,
    val: SplitFeatures,
    standardizer: Standardizer | None,
    weighting: str,
    monitor: BudgetMonitor,
    max_epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    from torch import nn

    from llm_router.models.complexity_mlp import build_mlp

    mcfg, seed = cfg.mlp, cfg.resources.seed
    seed_torch(seed)
    model = build_mlp(train.matrix.n_features, mcfg.hidden_dims, mcfg.dropouts, len(LABELS))
    model.to(device)
    weight = (
        torch.from_numpy(common.balanced_class_weights(train.y)).to(device)
        if weighting == "balanced"
        else None
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=mcfg.learning_rate, weight_decay=mcfg.weight_decay
    )
    train_loader = _loader(
        BatchDataset(train, standardizer), mcfg.batch_size, shuffle=True, seed=seed
    )
    val_loader = _loader(
        BatchDataset(val, standardizer), mcfg.eval_batch_size, shuffle=False, seed=seed
    )

    history: list[dict[str, Any]] = []
    best_key: tuple[float, float] | None = None
    best_state: dict[str, Any] | None = None
    best_metrics: dict[str, Any] = {}
    best_epoch, stale, stopped = -1, 0, None
    started = time.perf_counter()
    for epoch in range(max_epochs):
        model.train()
        total, seen = 0.0, 0
        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(X), y)
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(y)  # a float, so no graph is retained
            seen += len(y)
            stopped = monitor.exceeded()
            if stopped:
                break
        metrics = common.evaluate_proba(val.y, _validation_proba(model, val_loader, device))
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total / max(seen, 1), 6),
                "validation": common.brief(metrics),
                "partial_epoch": bool(stopped),
            }
        )
        key = common.selection_key(metrics)
        if best_key is None or key > best_key:
            best_key, best_epoch, stale, best_metrics = key, epoch, 0, metrics
            best_state = _cpu_state(model)
        else:
            stale += 1
        log.info(
            "mlp epoch",
            class_weight=weighting,
            epoch=epoch,
            loss=history[-1]["train_loss"],
            val_macro_f1=round(metrics["macro_f1"], 4),
        )
        if stopped or stale >= mcfg.patience:
            break
    assert best_state is not None
    model.to("cpu")  # the returned model is always on CPU (saving, CPU inference)
    model.load_state_dict(best_state)
    return {
        "model": model,
        "metrics": best_metrics,
        "summary": {
            "class_weight": weighting,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "early_stopped": stopped is None and len(history) < max_epochs,
            "stopped_reason": stopped,
            "fit_seconds": round(time.perf_counter() - started, 2),
            "validation": common.brief(best_metrics),
            "history": history,
        },
    }


def train_mlp(
    cfg: ComplexityConfig,
    *,
    feature_set: FeatureSet = "embeddings",
    embeddings_dir: Path | None = None,
    output_name: str | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    from llm_router.models.complexity_mlp import count_parameters

    res, mcfg = cfg.resources, cfg.mlp
    limit_threads(res.threads)
    torch.set_num_threads(res.threads)
    seed_torch(res.seed)
    device_name, device_note = resolve_device(res.device, "torch")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    log.info("mlp device", device=device_name, note=device_note)
    monitor = BudgetMonitor(res.max_train_seconds, res.memory_budget_gb)
    output_name = output_name or default_output_name(feature_set, smoke)
    train, val, emb_meta, emb_dir = common.prepare_training_data(
        cfg, feature_set, embeddings_dir=embeddings_dir, smoke=smoke, smoke_rows=mcfg.smoke_rows
    )
    schema = build_schema(feature_set, emb_meta)
    standardizer = Standardizer.fit(train.matrix) if mcfg.standardize else None
    max_epochs = mcfg.smoke_epochs if smoke else mcfg.max_epochs
    weight_options, weight_plan = common.class_weight_plan(
        train.y, mcfg.class_weights, res.imbalance_threshold
    )
    log.info(
        "mlp data",
        feature_set=feature_set,
        train_rows=train.matrix.n_rows,
        val_rows=val.matrix.n_rows,
        features=train.matrix.n_features,
    )
    progress_path = cfg.paths.reports_dir / "training" / f"{output_name}.json"

    variants: list[dict[str, Any]] = []
    best: tuple[tuple[float, float], Any, dict[str, Any], dict[str, Any]] | None = None
    for weighting in weight_options:
        if monitor.exceeded():
            log.warning("budget exhausted before variant", class_weight=weighting)
            break
        fitted = _fit_variant(cfg, train, val, standardizer, weighting, monitor, max_epochs, device)
        variants.append(fitted["summary"])
        common.write_json(progress_path, {"output_name": output_name, "variants": variants})
        key = common.selection_key(fitted["metrics"])
        if best is None or key > best[0]:
            best = (key, fitted["model"], fitted["summary"], fitted["metrics"])
        del fitted
        gc.collect()
    if best is None:
        raise ResourceBudgetError("no MLP variant finished within the resource budget")

    _, model, chosen, val_metrics = best
    architecture = {
        "input_dim": train.matrix.n_features,
        "hidden_dims": list(mcfg.hidden_dims),
        "dropouts": list(mcfg.dropouts),
        "n_classes": len(LABELS),
    }
    backend = MLPBackend(model, architecture, standardizer)
    out_dir = cfg.paths.models_dir / output_name
    staging = common.staging_dir(out_dir)
    backend.save(staging)
    metadata = {
        "classifier": "mlp",
        "output_name": output_name,
        "feature_set": feature_set,
        "classes": list(LABELS),
        "probability_columns": list(LABELS),
        "supports_probabilities": True,
        "architecture": architecture,
        "n_parameters": count_parameters(model),
        "standardized": standardizer is not None,
        "hyperparameters": {
            "hidden_dims": list(mcfg.hidden_dims),
            "dropouts": list(mcfg.dropouts),
            "optimizer": "AdamW",
            "learning_rate": mcfg.learning_rate,
            "weight_decay": mcfg.weight_decay,
            "batch_size": mcfg.batch_size,
            "max_epochs": max_epochs,
            "patience": mcfg.patience,
            "loss": "CrossEntropyLoss",
            "class_weight": chosen["class_weight"],
            "best_epoch": chosen["best_epoch"],
        },
        "selection": "best epoch and class weighting by max validation macro-F1, tie-break "
        "validation hard recall",
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
        "n_features": train.matrix.n_features,
        "training_seconds": round(sum(v["fit_seconds"] for v in variants), 2),
        "threads": res.threads,
        "training_device": device_name,
        "training_device_note": device_note,
        "peak_gpu_memory_gib": (
            round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
            if device.type == "cuda"
            else None
        ),
        "seed": res.seed,
        "smoke": smoke,
        **common.provenance(emb_meta),
        "embeddings_dir": str(emb_dir),
        "created_at": common.now(),
        "environment": common.environment_info(
            {
                "torch": torch.__version__,
                "torch_cuda": str(torch.version.cuda),
                "num_workers": "0",
                "precision": "float32",
            }
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(parser)
    parser.add_argument("--lr", type=float, default=None, help="override mlp.learning_rate")
    parser.add_argument("--batch-size", type=int, default=None, help="override mlp.batch_size")
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_cli_config(args)
    update = {
        k: v
        for k, v in (("learning_rate", args.lr), ("batch_size", args.batch_size))
        if v is not None
    }
    if update:
        cfg = cfg.model_copy(update={"mlp": cfg.mlp.model_copy(update=update)})
    metadata = train_mlp(
        cfg,
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
