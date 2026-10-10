# Experiments

## Reproducibility requirements

- Every run records: git commit, resolved config files, `uv.lock` hash, random seeds,
  dataset version, embedding model, and hardware (CPU / GPU type).
- Runs are tracked in MLflow; artifacts are written under `artifacts/`.
- Seeds are fixed for NumPy, PyTorch, XGBoost, and data splitting.
- Dependencies are pinned via `uv.lock`; environments are created with `uv sync`.

## Train / validation / test separation

- Splits are assigned once, at the query level, and stored with the data.
- **Train** — model fitting.
- **Validation** — hyperparameter selection, early stopping, calibration, threshold tuning.
- **Test** — reported results only; evaluated once per finalised configuration.
- `data/evaluation/` benchmarks are external and never used for any fitting or tuning.

## Baseline comparisons

Every reported router is compared against:

- each enabled registry candidate as a fixed-model baseline (every version separately)
- random routing (matched to the router's model mix)
- simple heuristic routers
- oracle routing (best model per query, upper bound)

Comparisons report quality, total cost, latency, and oracle gap, plus cost–quality curves
across routing thresholds.

## Experiment log

| ID | Date | Description | Config | Result |
|----|------|-------------|--------|--------|
