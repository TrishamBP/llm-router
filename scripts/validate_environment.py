"""Check the Python version and that required packages are installed.

Packages are located with ``importlib.util.find_spec`` and are not imported, so no
models are downloaded and no heavy initialisation (e.g. CUDA) is triggered.
"""

import argparse
import importlib.util
import sys

REQUIRED_PYTHON = (3, 12)

# Import names per dependency group (distribution names may differ).
GROUPS: dict[str, list[str]] = {
    "core": ["numpy", "pandas", "sklearn", "xgboost", "yaml", "pydantic", "structlog"],
    "neural": ["torch", "sentence_transformers"],
    "serving": ["fastapi", "uvicorn"],
    "experiments": ["mlflow", "dotenv", "tqdm"],
    "dev": ["pytest", "ruff", "mypy"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="*",
        default=[],
        choices=[g for g in GROUPS if g != "core"],
        help="Optional dependency groups to check in addition to core.",
    )
    args = parser.parse_args()

    ok = True
    version = sys.version_info[:2]
    if version < REQUIRED_PYTHON:
        print(f"[FAIL] Python {version[0]}.{version[1]} < required 3.12")
        ok = False
    else:
        print(f"[ OK ] Python {sys.version.split()[0]}")

    for group in ["core", *args.groups]:
        for module in GROUPS[group]:
            found = importlib.util.find_spec(module) is not None
            ok &= found
            print(f"[{' OK ' if found else 'FAIL'}] {group:<12} {module}")

    found = importlib.util.find_spec("llm_router") is not None
    ok &= found
    print(f"[{' OK ' if found else 'FAIL'}] {'package':<12} llm_router")

    print("Environment OK." if ok else "Environment check failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
