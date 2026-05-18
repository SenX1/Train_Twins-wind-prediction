"""Competition metrics and diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

INSTALLED_CAPACITY = 90.09
LOW_POWER_THRESHOLD = 10.0


def mae_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)) / INSTALLED_CAPACITY * 100.0)


def clip_predictions(y: np.ndarray, max_capacity: np.ndarray | float | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    y = np.clip(y, 0.0, INSTALLED_CAPACITY)
    if max_capacity is not None:
        y = np.minimum(y, np.asarray(max_capacity, dtype=float))
    return y


def summarize_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df_meta: pd.DataFrame | None = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    out = {
        "mae_pct": mae_pct(y_true, y_pred),
        "mae_mw": float(err.mean()),
        "low_power_mae_pct": mae_pct(
            y_true[y_true < LOW_POWER_THRESHOLD],
            y_pred[y_true < LOW_POWER_THRESHOLD],
        )
        if np.any(y_true < LOW_POWER_THRESHOLD)
        else None,
        "high_power_mae_pct": mae_pct(
            y_true[y_true >= LOW_POWER_THRESHOLD],
            y_pred[y_true >= LOW_POWER_THRESHOLD],
        )
        if np.any(y_true >= LOW_POWER_THRESHOLD)
        else None,
    }
    if df_meta is not None and "month" in df_meta.columns:
        out["month_mae_pct"] = {
            int(m): mae_pct(y_true[df_meta["month"].values == m], y_pred[df_meta["month"].values == m])
            for m in sorted(df_meta["month"].unique())
        }
    if df_meta is not None and "wind_sector" in df_meta.columns:
        out["sector_mae_pct"] = {
            int(s): mae_pct(
                y_true[df_meta["wind_sector"].values == s],
                y_pred[df_meta["wind_sector"].values == s],
            )
            for s in sorted(df_meta["wind_sector"].dropna().unique())[:8]
        }
    return out
