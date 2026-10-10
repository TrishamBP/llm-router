# Dataset Design

Data is split into two record types joined by `query_id`. Query-level information is stored
once; model-dependent outcomes are stored per (query, model) pair.

## 1. Query records (one row per query)

Model-independent inputs available **at routing time**.

- `query_id` — stable unique identifier
- query text (or reference to it)
- source / dataset / task category
- structured metadata (e.g. conversation length, presence of code, requested format)
- split assignment (`train` / `validation` / `test`)

Derived features (`features/`) and embeddings (`data/embeddings/`) are keyed by `query_id`
and versioned by feature-config / embedding-model.

## 2. Per-model outcome records (one row per query × model)

Observed results from running a candidate model on a query. **Never** used as router inputs.

- `query_id`, canonical `model_id` from the registry, provider and exact provider model ID,
  family and version at evaluation time (see `llm_router.schemas.OutcomeRecord`)
- response (or reference to it)
- quality score(s) and how they were obtained (reference, judge, human)
- input / output token counts and inference cost
- latency (time-to-first-token, total)
- error / refusal / truncation flags
- collection timestamp and run identifier

## 3. Labels

Training targets are derived from outcome records, e.g. per-model quality, per-model utility
under a reward configuration, or pairwise preferences between models. Label derivation is
deterministic given `configs/reward.yaml`.

## 4. Rules

- Splits are assigned at the **query** level so all outcomes for a query share a split.
- Near-duplicate queries must fall in the same split.
- `data/evaluation/` holds frozen benchmark sets that are never used for training or tuning.
- Raw data is immutable; all transformations write to `interim/` or `processed/`.
