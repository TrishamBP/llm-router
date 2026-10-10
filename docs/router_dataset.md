# Router dataset: DevQuasar/llm_router_dataset-synth

A second, binary routing dataset (`small_llm` / `large_llm`), split leakage-safely and
embedded with the same Fireworks model as the complexity dataset, ready for CPU training
(XGBoost / MLP; no training happens in this stage).

## Status (2026-10-10, all steps run on the real data)

| Step | Result |
|---|---|
| Inspect + schema/label gate | ✅ valid; 20,227 rows |
| Clean, dedupe, group, split (seed 42) | ✅ 17,016 rows; 13,614 / 1,701 / 1,701; isolation verified |
| Connectivity (1 synthetic prompt) | ✅ `fireworks/qwen3-embedding-8b`, 4,096 dims |
| Embeddings | ✅ 267 requests, 390,788 tokens (≈ $0.039 at $0.10/M), 0 failed rows |
| Artifact validation | ✅ alignment, hashes, fingerprints, labels, isolation, finite float32 |

## Source

- [`DevQuasar/llm_router_dataset-synth`](https://huggingface.co/datasets/DevQuasar/llm_router_dataset-synth),
  revision `cd0aa4a0ee35c7c3a31e061e9ad240c732e7cdda`, **Apache-2.0**. Attribute the source.
- `load_dataset("DevQuasar/llm_router_dataset-synth")` works. The pipeline itself keeps
  byte-identical copies of `train.jsonl`, `test.jsonl`, and `README.md`, with SHA-256 hashes,
  in `data/raw/llm_router_dataset-synth/<revision>/`.
- Fields: `id` (string), `prompt` (string), `label` (an integer in the raw JSON). The card
  declares `class_label` names `0: small_llm`, `1: large_llm`. **The card doesn't say how
  labels were assigned or what they mean.** Treat them as published routing labels: not
  complexity scores, and not verified to mean "hard".

## What inspection found

`artifacts/reports/router_dataset_inspection.{json,md}`:

| Item | Value |
|---|---|
| Official splits | train 15,306, test 4,921 (no validation split) |
| Labels | small_llm 8,633 (42.7%), large_llm 11,594 (57.3%) |
| Missing / empty prompts | 0 / 0 |
| IDs | 3,220 unique; 1–11 rows per ID (median 6); no ID in both official splits; no ID with two labels |
| Exact duplicate prompts | 654 groups, 3,173 extra rows; **400 groups span official train and test** |
| Normalised duplicates | 664 groups, 3,211 extra rows; every group spans several IDs |
| Conflicting labels | 0 |
| Prompt length | 5–681 chars (median 101); 1–120 words (median 15) |

An ID is a generation batch of topically related prompts that share a label, which makes it
a sound grouping signal. The source also reuses stock prompts across batches: "What is the
capital of France?" appears under 167 IDs.

## Cleaning and split strategy

`uv run python -m llm_router.data.prepare_router_dataset`. It is deterministic; the fingerprint
`e22af8d804382af2` reproduced on a re-run.

1. Remove missing or empty prompts and labels the card doesn't declare. Each removal is
   recorded in `removed.jsonl` with a reason.
2. Normalise conservatively (`splitting.normalize_text`). Code, maths, punctuation, and
   indentation are preserved.
3. Remove normalised prompts with conflicting labels entirely. This revision has none.
4. **Deduplicate before grouping.** Keep the first copy of each normalised prompt (official
   train before test, then row index) and list the removed copies on the kept row. If IDs were
   linked through the reused stock prompts instead, one component would contain 4,754 rows
   (23.5%), all small_llm, which would force a distorted split.
5. Group rows by source ID and normalised prompt (connected components). After step 4 this is
   exactly source-ID grouping: 3,002 groups of 1–11 rows.
6. Split 80/10/10 with label-stratified greedy allocation (seed 42). Groups are never split.
   The official split isn't used, because it leaks 400 identical prompts and has no
   validation split.
7. Assert that example IDs, group IDs, normalised prompt keys, and source IDs never span splits.

| Split | Rows | Share | Groups | large_llm | small_llm |
|---|---|---|---|---|---|
| train | 13,614 | 80.0% | 2,392 | 9,181 (67.4%) | 4,433 (32.6%) |
| validation | 1,701 | 10.0% | 309 | 1,146 (67.4%) | 555 (32.6%) |
| test | 1,701 | 10.0% | 301 | 1,149 (67.5%) | 552 (32.5%) |

**Deduplication changes the class balance** from 57/43 to 67/33, because most of the repeated
stock prompts are small_llm. The training imbalance ratio is 2.07, so consider class
weighting when training.

**Embedding-space near-duplicates** (report only; splits are never changed afterwards): 44
validation and 38 test rows have a training row with cosine ≥ 0.98. They are paraphrases from
different batches ("What's your name?" / "What is your name?"), all with the same label. Their
IDs are in `artifacts/reports/router_near_duplicate_ids.json` for a sensitivity evaluation
that excludes them.

## Artifacts

```text
data/processed/router_dataset/
  train.jsonl validation.jsonl test.jsonl   example_id, source_id, prompt, label, label_id,
                                            source_split, source_row_index, prompt_key,
                                            group_id, duplicate_ids, split
  label_mapping.json                        {"small_llm": 0, "large_llm": 1} (+ hash, provenance)
  split_metadata.json                       counts, proportions, policy, isolation, fingerprints
  removed.jsonl                             every removed row with its reason
  hf_dataset/                               the same splits as a Hugging Face DatasetDict
artifacts/embeddings/router_dataset/
  connectivity/                             this dataset's 1-request model check
  latest.json                               -> deb8dfcdc0fc1b65
  deb8dfcdc0fc1b65/
    train.npy validation.npy test.npy       float32 (n, 4096)
    <split>.row_mapping.jsonl               row -> example_id, label, prompt hash
    metadata.json state.json parts/         model, dims, fingerprints, usage, resume state
artifacts/reports/router_dataset_inspection.{json,md}
artifacts/reports/router_artifacts_validation.{json,md}
artifacts/reports/router_near_duplicate_ids.json
```

This follows the repository's existing layout (`artifacts/embeddings/<cache_key>/`) rather than
the `data/embeddings/...` layout suggested in the task.

**Never use `id`, `source_id`, `example_id`, or `group_id` as features.**

## Embeddings, caching, and resume

These use the shared client and job (`llm_router.embeddings.fireworks_client` and
`generate.EmbeddingJob`), so retries, failure handling, and resume behave exactly like the
complexity dataset:

- Bounded exponential backoff with jitter for 429, timeouts, and 5xx errors.
- Authentication, permission, and unknown-model errors fail immediately.
- Each batch is written atomically and the job resumes from the last completed batch.
- Vector dimensions and finiteness are checked; a rejected input is isolated by bisection.

The cache key hashes the processed fingerprint, the preparation version, the model ID, and the
`dimensions` setting. The processed fingerprint covers the raw revision, the normalised prompt
hashes, the labels, the label mapping, and the split assignment. Any change to these produces
a new cache, so vectors from different models or data revisions are never mixed. Credentials
come from `.env` (`FIREWORKS_API_KEY`, `FIREWORKS_EMBEDDING_MODEL`) and are never logged.

## Loader

```python
from llm_router.data.router_dataset import load_router_dataset

data = load_router_dataset()  # or load_router_dataset("artifacts/embeddings/router_dataset")
data.X_train, data.y_train  # np.memmap float32 (13614, 4096), int64 {0, 1}
data.X_validation, data.y_validation
data.X_test, data.y_test
data.label_names  # ("small_llm", "large_llm")
```

Before it returns, the loader checks:
- the cache is complete;
- the processed and embedding fingerprints match;
- the label-mapping hash matches;
- rows are 0..n-1 and aligned with the processed records;
- each row's label, label ID, and prompt hash match;
- the shape is (n, dim) float32 and all values are finite;
- example IDs, group IDs, and normalised prompts don't overlap between splits.

Arrays are memory-mapped, so loading all three splits uses well under 1 GB of RAM.

## Commands

```bash
uv sync --group complexity
uv run python -m llm_router.data.inspect_router_dataset            # download + report + gate
uv run python -m llm_router.data.prepare_router_dataset            # clean + split (--no-hf)
uv run python -m llm_router.embeddings.generate_router_embeddings --test      # 1 request
uv run python -m llm_router.embeddings.generate_router_embeddings --estimate  # no API calls
uv run python -m llm_router.embeddings.generate_router_embeddings --yes       # resumable
uv run python -m llm_router.data.validate_router_artifacts         # full verification
uv run python -m llm_router.data.router_smoke                      # synthetic, no credentials
uv run pytest tests/unit/router                                    # mocked API, no charges
```

The commands are the same in PowerShell and bash. Runs larger than 5 batches need `--yes`.
Measured resource use: preparation in seconds; embedding peak RSS 0.34 GiB; validation peak
RSS 0.46 GiB.

## Limitations

- The labels' semantics and how they were assigned are undocumented. The data is synthetic and
  heavily templated (stock prompts, one-word paraphrases).
- Text-level grouping can't catch paraphrases; see the embedding near-duplicate report above.
- Labels are binary and not aligned with the complexity dataset's easy/medium/hard labels. Do
  not merge the two label spaces without an explicit mapping decision.

## Merged training experiment (2026-10-10)

This was a user decision: map `large_llm` → `hard` and `small_llm` → `medium`, and add the
router *train* rows to complexity training
(`uv run python -m llm_router.data.build_merged_dataset`, then
`uv run python -m llm_router.training.run_comparison --config configs/complexity_merged.yaml`).

- Train: 75,072 rows (61,461 complexity + 13,611 router). 3 router rows that duplicated
  complexity prompts were dropped, and 0 router rows were near-paraphrases (cosine ≥ 0.98) of
  validation prompts. Validation and test are the unchanged complexity splits, and all settings
  match `configs/complexity.yaml`.
- Train labels: easy 20,146 / medium 35,873 / hard 19,053. Hard is 25% of training but 16% of
  evaluation.
- Outputs are in `artifacts/models/merged/` and `artifacts/reports/merged/`. The complexity-only
  models are untouched.

Test macro-F1 for the same models on the same test split (complexity-only → merged):
A logreg 0.6951 → 0.6808, B 0.6172 → 0.5731, D 0.6309 → 0.6036, F1 0.4933 → 0.4830,
C MLP 0.7071 → 0.6913, E MLP 0.7009 → 0.6904, F2 0.5119 → 0.4946.

**Every model got worse with the mapped data** (−1.0 to −4.4 points), so the complexity-only
models remain the better choice. No significance test was run.
