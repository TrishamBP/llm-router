#!/usr/bin/env bash
# Lightning AI Studio setup for the XGBoost / MLP complexity classifiers (CPU only).
#
# Installs the `complexity` dependency group, checks that the cached data the trainers need
# has been copied over, and runs the quick smoke comparison. It makes no API calls and never
# reads .env (training uses cached embeddings only).
#
# Usage:  bash scripts/setup_lightning_complexity.sh [--no-smoke]
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed on this Studio. See https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv python install
uv sync --group complexity

uv run python - <<'PY'
import os, sys, torch, xgboost
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  xgboost {xgboost.__version__}  cpus {os.cpu_count()}")
print("CUDA visible:", torch.cuda.is_available(), "(not used: the classifiers run on CPU only)")
PY

missing=0
for path in data/processed/brick-complexity/manifest.json artifacts/embeddings/latest.json; do
    if [ ! -f "$path" ]; then echo "missing: $path" >&2; missing=1; fi
done
if [ "$missing" -eq 0 ]; then
    key=$(uv run python -c "import json; print(json.load(open('artifacts/embeddings/latest.json'))['cache_key'])")
    for split in train validation test; do
        for suffix in npy row_mapping.jsonl; do
            f="artifacts/embeddings/$key/$split.$suffix"
            if [ ! -f "$f" ]; then echo "missing: $f" >&2; missing=1; fi
        done
    done
    [ -f "artifacts/embeddings/$key/metadata.json" ] || { echo "missing: artifacts/embeddings/$key/metadata.json" >&2; missing=1; }
fi
if [ "$missing" -ne 0 ]; then
    echo "Copy the processed splits and the completed embedding cache first (see docs/complexity_classifier.md)." >&2
    exit 1
fi

if [ "${1:-}" != "--no-smoke" ]; then
    uv run python -m llm_router.training.run_comparison --smoke
fi
