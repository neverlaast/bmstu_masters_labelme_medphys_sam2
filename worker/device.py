from __future__ import annotations


def detect_best_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def get_runtime_device_summary() -> str:
    return detect_best_device()
