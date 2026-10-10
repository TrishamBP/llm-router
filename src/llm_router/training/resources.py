"""CPU thread limits, memory measurement, and time/memory budgets for training jobs.

One thread budget covers BLAS, OpenMP, XGBoost, and torch so libraries do not nest their
thread pools. Environment variables only affect libraries initialised afterwards, so the
comparison runner also passes them to each child process at spawn time; `limit_threads`
additionally applies threadpoolctl limits to already-loaded BLAS/OpenMP pools.

Memory figures are process-level: peak working set on Windows (psutil), `ru_maxrss` on
Linux/macOS. Both include pages of memory-mapped embedding files that were touched.
"""

from __future__ import annotations

import mmap
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

GIB = 1024**3


def thread_env(threads: int) -> dict[str, str]:
    return dict.fromkeys(THREAD_ENV_VARS, str(threads))


def limit_threads(threads: int) -> None:
    os.environ.update(thread_env(threads))
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(threads)
    except ImportError:  # pragma: no cover - threadpoolctl ships with scikit-learn
        pass
    if "torch" in sys.modules:
        import torch

        torch.set_num_threads(threads)


class DeviceUnavailableError(RuntimeError):
    pass


def _torch_cuda_ok() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "torch is not installed"
    if not torch.cuda.is_available():
        build = torch.version.cuda or "CPU-only build"
        return False, f"torch {torch.__version__} ({build}) sees no usable CUDA GPU"
    return True, torch.cuda.get_device_name(0)


def _xgboost_cuda_ok() -> tuple[bool, str]:
    import xgboost as xgb

    if not xgb.build_info().get("USE_CUDA"):
        return False, f"xgboost {xgb.__version__} was built without CUDA"
    try:  # a tiny fit proves the GPU is actually usable (driver, runtime, memory)
        X = np.zeros((8, 2), dtype=np.float32)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.float32)
        params = {"objective": "multi:softprob", "num_class": 3, "device": "cuda", "verbosity": 0}
        xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=1)
    except xgb.core.XGBoostError as exc:
        return False, f"xgboost CUDA check failed: {str(exc).splitlines()[0][:200]}"
    return True, "cuda:0"


def resolve_device(requested: str, library: str) -> tuple[str, str]:
    """Return (device, note) for `library` ("torch" or "xgboost").

    "cuda" raises DeviceUnavailableError if the GPU cannot be used (never silently falls back);
    "auto" falls back to CPU and says why in the note.
    """
    if requested == "cpu":
        return "cpu", "cpu requested"
    ok, detail = _torch_cuda_ok() if library == "torch" else _xgboost_cuda_ok()
    if ok:
        return "cuda", detail
    if requested == "cuda":
        raise DeviceUnavailableError(f"resources.device=cuda but {library} cannot use it: {detail}")
    return "cpu", f"auto: falling back to CPU ({detail})"


def current_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * mmap.PAGESIZE
    return None


def peak_rss_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            import psutil

            return int(psutil.Process().memory_info().peak_wset)
        except (ImportError, AttributeError):
            return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # Linux reports KiB


def to_gib(value: int | None) -> float | None:
    return None if value is None else round(value / GIB, 3)


def directory_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


class BudgetMonitor:
    """Tracks elapsed time and RSS against the configured budgets (checked by trainers)."""

    def __init__(self, max_seconds: float | None, memory_budget_gb: float | None) -> None:
        self.max_seconds = max_seconds
        self.memory_budget = None if memory_budget_gb is None else memory_budget_gb * GIB
        self.started = time.monotonic()

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def exceeded(self) -> str | None:
        if self.max_seconds is not None and self.elapsed > self.max_seconds:
            return f"time budget of {self.max_seconds:.0f}s exceeded"
        rss = current_rss_bytes()
        if self.memory_budget is not None and rss is not None and rss > self.memory_budget:
            return f"memory budget exceeded (RSS {rss / GIB:.2f} GiB)"
        return None

    def report(self) -> dict[str, Any]:
        return {
            "elapsed_seconds": round(self.elapsed, 2),
            "rss_gib": to_gib(current_rss_bytes()),
            "peak_rss_gib": to_gib(peak_rss_bytes()),
            "peak_rss_note": "process peak; includes touched pages of memory-mapped embeddings",
        }
