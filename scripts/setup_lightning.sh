#!/usr/bin/env bash
# Lightning AI environment setup -- STARTING POINT ONLY.
#
# This script intentionally does NOT install GPU-specific (CUDA) PyTorch builds.
# GPU configuration will be added when the Lightning AI environment is prepared.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed on this Studio. See https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv python install
uv sync --group neural --group experiments

# TODO(lightning): configure CUDA-enabled PyTorch index and verify GPU visibility.

uv run python scripts/validate_environment.py --groups neural experiments
