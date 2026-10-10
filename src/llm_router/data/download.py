"""Download the raw dataset files, byte-for-byte, at a pinned revision, with provenance.

Usage: uv run python -m llm_router.data.download [--config configs/complexity.yaml]

Files are fetched with `huggingface_hub` (no transformation) into `paths.raw_dir/<revision>/`
and are never modified afterwards. `provenance.json` records the source, revision, license,
SHA-256 of every file, and a content fingerprint used to key all downstream caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_router.config import ComplexityConfig, load_config, sha256_json
from llm_router.logging import configure_logging, get_logger

log = get_logger(__name__)

LICENSE_NOTE = (
    "CC BY-NC 4.0: attribution required; non-commercial use only. The raw data and models "
    "trained on it must not be used commercially without permission from Regolo.ai."
)


def raw_snapshot_dir(cfg: ComplexityConfig) -> Path:
    return cfg.paths.raw_dir / cfg.dataset.revision


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_provenance(cfg: ComplexityConfig) -> dict[str, Any]:
    path = raw_snapshot_dir(cfg) / "provenance.json"
    if not path.exists():
        raise FileNotFoundError(
            f"raw dataset not found at {path.parent}; run `python -m llm_router.data.download`"
        )
    provenance: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for split, info in provenance["files"].items():
        local = path.parent / info["local_path"]
        if file_sha256(local) != info["sha256"]:
            raise RuntimeError(f"raw file for split '{split}' changed since download: {local}")
    return provenance


def download(cfg: ComplexityConfig, *, force: bool = False) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    target = raw_snapshot_dir(cfg)
    if (target / "provenance.json").exists() and not force:
        provenance = read_provenance(cfg)  # verifies hashes
        log.info("raw dataset already present and unchanged", path=str(target))
        return provenance

    target.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for split, repo_path in sorted(cfg.dataset.data_files.items()):
        cached = Path(
            hf_hub_download(
                repo_id=cfg.dataset.repo_id,
                filename=repo_path,
                repo_type="dataset",
                revision=cfg.dataset.revision,
            )
        )
        local = target / Path(repo_path).name
        shutil.copyfile(cached, local)
        files[split] = {
            "repo_path": repo_path,
            "local_path": local.name,
            "sha256": file_sha256(local),
            "bytes": local.stat().st_size,
        }
        log.info("downloaded", split=split, bytes=files[split]["bytes"])

    provenance = {
        "repo_id": cfg.dataset.repo_id,
        "revision": cfg.dataset.revision,
        "url": f"https://huggingface.co/datasets/{cfg.dataset.repo_id}",
        "license": cfg.dataset.license,
        "license_note": LICENSE_NOTE,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
        "content_fingerprint": sha256_json({s: f["sha256"] for s, f in files.items()})[:16],
    }
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/complexity.yaml")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)
    configure_logging()
    provenance = download(load_config(args.config), force=args.force)
    print(json.dumps({k: provenance[k] for k in ("revision", "content_fingerprint")}, indent=2))
    print(LICENSE_NOTE)


if __name__ == "__main__":
    main()
