#!/usr/bin/env bash
# Local (CPU) development environment setup using uv.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. See https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv python install
uv sync    # core dependencies + dev group

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example; fill in local values."
fi

uv run python scripts/validate_environment.py
