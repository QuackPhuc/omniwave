"""Environment fingerprinting for reproducibility."""

from __future__ import annotations

import platform
import subprocess
from typing import Any

import numpy as np
import torch


def _git_revision() -> dict[str, Any]:
    """Capture current Git revision and dirty state."""
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        )
        return {"revision": rev, "dirty": dirty}
    except (subprocess.SubprocessError, FileNotFoundError):
        return {"revision": "unknown", "dirty": None}


def _triton_version() -> str | None:
    """Return Triton version if installed."""
    try:
        import triton
        return str(triton.__version__)
    except ImportError:
        return None


def collect_environment() -> dict[str, Any]:
    """Capture a comprehensive environment fingerprint."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "triton": _triton_version(),
        "git": _git_revision(),
        "cuda": {
            "available": torch.cuda.is_available(),
        },
    }

    if torch.cuda.is_available():
        env["cuda"].update({
            "runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / (1024**3),
                        2,
                    ),
                }
                for i in range(torch.cuda.device_count())
            ],
        })

    env["deterministic"] = {
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }

    return env
