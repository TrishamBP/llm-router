# Data

All dataset contents are git-ignored; only this README and `.gitkeep` placeholders are tracked.

| Directory     | Contents                                                              |
|---------------|-----------------------------------------------------------------------|
| `raw/`        | Immutable source data: queries and raw per-model responses/metadata.  |
| `interim/`    | Cleaned / joined intermediate tables.                                 |
| `processed/`  | Final feature tables and labels used for training.                    |
| `embeddings/` | Cached query embeddings, keyed by embedding model and query ID.       |
| `evaluation/` | Held-out benchmark sets. Never used for training or tuning.           |

See `docs/dataset_design.md` for the record schemas.
