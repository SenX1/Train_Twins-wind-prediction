"""CatBoost GPU helpers."""
from __future__ import annotations

import os

from catboost import CatBoostRegressor


def gpu_task_config() -> dict:
    """Force GPU; set CATBOOST_CPU=1 to override."""
    if os.environ.get("CATBOOST_CPU", "").strip() == "1":
        return {"task_type": "CPU", "thread_count": -1}
    device = os.environ.get("CATBOOST_GPU_DEVICE", "0")
    return {
        "task_type": "GPU",
        "devices": device,
    }


def assert_gpu_available() -> str:
    cfg = gpu_task_config()
    if cfg["task_type"] != "GPU":
        return "CPU"
    try:
        m = CatBoostRegressor(iterations=2, **cfg, verbose=0)
        m.fit([[0.0], [1.0]], [0.0, 1.0])
        return "GPU"
    except Exception as exc:
        raise RuntimeError(
            "GPU requested but CatBoost GPU failed. Install GPU build or set CATBOOST_CPU=1. "
            f"Error: {exc}"
        ) from exc
