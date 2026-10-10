"""Load the downloaded raw snapshot with Hugging Face Datasets (read-only)."""

from __future__ import annotations

from typing import Any

from llm_router.config import ComplexityConfig
from llm_router.data.download import raw_snapshot_dir, read_provenance


def load_raw_records(cfg: ComplexityConfig) -> dict[str, list[dict[str, Any]]]:
    """Return {split: [row, ...]} exactly as stored in the raw files (hash-verified)."""
    from datasets import load_dataset

    provenance = read_provenance(cfg)
    root = raw_snapshot_dir(cfg)
    data_files = {s: str(root / info["local_path"]) for s, info in provenance["files"].items()}
    dataset = load_dataset("json", data_files=data_files)
    return {split: dataset[split].to_list() for split in sorted(dataset)}
