"""
Ансамбль из 11 моделей CatBoost с Ridge-стекингом и изотонической калибровкой.
OOF кросс-валидация через TimeSeriesSplit с разрывом.
"""
from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.physics import generate_all_features, TARGET, DATETIME_COL, INSTALLED_CAPACITY  # noqa: E402
from models.gpu_utils import assert_gpu_available, gpu_task_config  # noqa: E402
from metrics.competition_metrics import mae_pct  # noqa: E402

ARTIFACT_DIR = ROOT / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = ROOT / "outputs"

log = logging.getLogger("ensemble_q1")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
if not log.handlers:
    log.addHandler(logging.StreamHandler())
    _fh = logging.FileHandler(ARTIFACT_DIR / "train_q1.log", encoding="utf-8")
    _fh.setFormatter(_fmt)
    log.addHandler(_fh)

RANDOM_SEED = 42
N_SPLITS = 5
CV_GAP = 24
N_BASELINE = 3
N_DIRECT = 5
N_RESID = 3
OOF_ES_ROUNDS = 80

MANUAL_PARAMS = {
    "iterations": 939,
    "learning_rate": 0.010025183496094117,
    "depth": 6,
    "l2_leaf_reg": 4.256625768288167,
    "random_strength": 2.1465008633934577,
    "bagging_temperature": 0.9995259396420547,
}


def purged_splits(n: int, n_splits: int, gap: int):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr, va in tscv.split(np.arange(n)):
        if gap and len(tr) > gap:
            tr = tr[:-gap]
        if len(tr) and len(va):
            yield tr, va


def cf_to_mw(cf: np.ndarray, max_cap: np.ndarray) -> np.ndarray:
    return np.clip(np.clip(cf, 0, 1) * max_cap, 0, max_cap)


def _model_params(base_seed: int, i: int = 0, for_fit: bool = True) -> dict:
    p = MANUAL_PARAMS.copy()
    p.update(gpu_task_config())
    p.update({
        "loss_function": "MAE",
        "random_seed": base_seed + i,
        "allow_writing_files": False,
    })
    if i > 0:
        p["random_strength"] = min(3.0, p["random_strength"] + 0.25 * i)
        p["bagging_temperature"] = (p["bagging_temperature"] + 0.08 * i) % 1.0
    p["verbose"] = (100 if i == 0 else 0) if for_fit else 0
    return p


def make_models(n: int, base_seed: int) -> list[CatBoostRegressor]:
    return [CatBoostRegressor(**_model_params(base_seed, i)) for i in range(n)]


def fit_one(model, X_tr, y_tr, cat_cols, X_va=None, y_va=None):
    if X_va is not None and y_va is not None:
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), cat_features=cat_cols,
                  early_stopping_rounds=OOF_ES_ROUNDS, use_best_model=True)
    else:
        model.fit(X_tr, y_tr, cat_features=cat_cols)
    return model


def fit_ensemble(models, X_tr, y_tr, cat_cols, X_va=None, y_va=None):
    for m in models:
        fit_one(m, X_tr, y_tr, cat_cols, X_va, y_va)


def predict_ensemble(models, X) -> np.ndarray:
    return np.mean([m.predict(X) for m in models], axis=0)


def _fit_stack_meta(phys, direct, res, y_cf):
    X = np.column_stack([phys, direct, res, direct + res])
    ridge = Ridge(alpha=0.5, positive=True, fit_intercept=True, random_state=RANDOM_SEED)
    ridge.fit(X, y_cf)
    cf = np.clip(X @ ridge.coef_ + ridge.intercept_, 0, 1)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(cf, y_cf)
    cf_cal = iso.predict(cf)
    return ridge, iso, cf_cal


def _stack_predict(phys, direct, res, coef, intercept, iso):
    X = np.column_stack([phys, direct, res, direct + res])
    cf = np.clip(X @ coef + intercept, 0, 1)
    if iso is not None:
        cf = iso.predict(cf)
    return cf


def _find_blend_alpha(y_mw, pred_a, pred_b) -> float:
    best_a, best_m = 0.5, 1e9
    for a in np.linspace(0, 1, 41):
        m = mae_pct(y_mw, a * pred_b + (1 - a) * pred_a)
        if m < best_m:
            best_m, best_a = m, float(a)
    return best_a


def run_oof(train_df, feature_cols, cat_cols) -> dict:
    df = train_df.sort_values(DATETIME_COL).reset_index(drop=True)
    X = df[feature_cols]
    y_cf = df["cf_target"].values
    y_mw = df[TARGET].values
    max_cap = df["max_capacity"].values
    phys_cf = df["physics_cf"].values
    res_tgt = df["cf_residual"].values
    n = len(df)

    oof_base = np.zeros(n)
    oof_direct = np.zeros(n)
    oof_res = np.zeros(n)
    fold_pct_base, fold_pct_stack = [], []

    for fold, (tr_idx, va_idx) in enumerate(purged_splits(n, N_SPLITS, CV_GAP)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_cf[tr_idx], y_cf[va_idx]

        direct_models = make_models(N_BASELINE, RANDOM_SEED + fold)
        fit_ensemble(direct_models, X_tr, y_tr, cat_cols, X_va, y_va)
        oof_direct[va_idx] = predict_ensemble(direct_models, X_va)
        oof_base[va_idx] = oof_direct[va_idx]

        res_models = make_models(2, RANDOM_SEED + 100 + fold)
        fit_ensemble(res_models, X_tr, res_tgt[tr_idx], cat_cols, X_va, res_tgt[va_idx])
        oof_res[va_idx] = predict_ensemble(res_models, X_va)

        m_base = mae_pct(y_mw[va_idx], cf_to_mw(oof_base[va_idx], max_cap[va_idx]))
        stack_cf = np.clip(oof_direct[va_idx] + oof_res[va_idx], 0, 1)
        m_st = mae_pct(y_mw[va_idx], cf_to_mw(stack_cf, max_cap[va_idx]))
        fold_pct_base.append(m_base)
        fold_pct_stack.append(m_st)
        log.info("Fold %d | baseline=%.4f%% | stack=%.4f%%", fold, m_base, m_st)

    mask = oof_base != 0
    ridge, iso, stack_cf_fit = _fit_stack_meta(
        phys_cf[mask], oof_direct[mask], oof_res[mask], y_cf[mask]
    )

    mw_base = cf_to_mw(oof_base[mask], max_cap[mask])
    mw_stack = cf_to_mw(stack_cf_fit, max_cap[mask])

    blend_alpha = _find_blend_alpha(y_mw[mask], mw_base, mw_stack)
    mw_blend = blend_alpha * mw_stack + (1 - blend_alpha) * mw_base

    return {
        "ridge_coef": ridge.coef_.tolist(),
        "ridge_intercept": float(ridge.intercept_),
        "iso": iso,
        "blend_alpha": blend_alpha,
        "oof_mae_pct_baseline": mae_pct(y_mw[mask], mw_base),
        "oof_mae_pct_stack": mae_pct(y_mw[mask], mw_stack),
        "oof_mae_pct_blend": mae_pct(y_mw[mask], mw_blend),
        "mean_fold_baseline_pct": float(np.mean(fold_pct_base)),
        "mean_fold_stack_pct": float(np.mean(fold_pct_stack)),
    }


def train_full(train_df, test_df, feature_cols, cat_cols, cv_meta):
    X_tr = train_df[feature_cols]
    y_cf = train_df["cf_target"].values
    res = train_df["cf_residual"].values
    X_te = test_df[feature_cols]
    max_cap_te = test_df["max_capacity"].values
    phys_te = test_df["physics_cf"].values

    log.info("Training baseline ensemble (%d models)...", N_BASELINE)
    base_models = make_models(N_BASELINE, RANDOM_SEED)
    for m in base_models:
        fit_one(m, X_tr, y_cf, cat_cols)

    log.info("Training direct ensemble (%d models)...", N_DIRECT)
    direct_models = make_models(N_DIRECT, RANDOM_SEED + 20)
    for m in direct_models:
        fit_one(m, X_tr, y_cf, cat_cols)

    log.info("Training residual ensemble (%d models)...", N_RESID)
    res_models = make_models(N_RESID, RANDOM_SEED + 40)
    for m in res_models:
        fit_one(m, X_tr, res, cat_cols)

    pred_base = predict_ensemble(base_models, X_te)
    pred_direct = predict_ensemble(direct_models, X_te)
    pred_res = predict_ensemble(res_models, X_te)

    stack_cf = _stack_predict(
        phys_te, pred_direct, pred_res,
        np.array(cv_meta["ridge_coef"]),
        cv_meta["ridge_intercept"],
        cv_meta["iso"],
    )
    mw_base = cf_to_mw(pred_base, max_cap_te)
    mw_stack = cf_to_mw(stack_cf, max_cap_te)
    a = cv_meta["blend_alpha"]
    mw_final = np.clip(a * mw_stack + (1 - a) * mw_base, 0, max_cap_te)

    return mw_stack, mw_final


def save_submission(preds_chronological_asc: np.ndarray, name: str) -> Path:
    """Save predictions reversed (last hour first) with header for submission."""
    sub = np.asarray(preds_chronological_asc, dtype=float)[::-1]
    path = OUTPUT_DIR / name
    pd.DataFrame(sub, columns=["Prediction"]).to_csv(path, index=False, float_format="%.6f")
    return path


class Q1Ensemble:
    def run(self):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        task = assert_gpu_available()
        log.info("=== Q1 Ensemble | %s ===", task)

        train_raw = pd.read_csv(ROOT / "data" / "train_dataset.csv").dropna(subset=[TARGET])
        test_raw = pd.read_csv(ROOT / "data" / "valid_features.csv")
        row_map = pd.DataFrame({
            "_orig_row": np.arange(len(test_raw)),
            DATETIME_COL: pd.to_datetime(test_raw[DATETIME_COL], format="ISO8601"),
        })

        log.info("Building features...")
        train_df, test_df, feature_cols, cat_cols = generate_all_features(train_raw, test_raw)

        # Подготовка таргетов
        train_df["physics_cf"] = np.clip(train_df["physics_prediction"] / train_df["max_capacity"], 0, 1.2)
        train_df["cf_target"] = np.clip(train_df[TARGET] / train_df["max_capacity"], 0, 1)
        train_df["cf_residual"] = train_df["cf_target"] - train_df["physics_cf"]
        test_df["physics_cf"] = np.clip(test_df["physics_prediction"] / test_df["max_capacity"], 0, 1.2)

        test_df[DATETIME_COL] = pd.to_datetime(test_df[DATETIME_COL], format="ISO8601")
        test_df = test_df.merge(row_map, on=DATETIME_COL, how="left")

        log.info("Features=%d", len(feature_cols))
        log.info("OOF CV (%d folds)...", N_SPLITS)
        cv = run_oof(train_df, feature_cols, cat_cols)
        log.info("OOF %% | baseline=%.4f | stack=%.4f | blend=%.4f | alpha=%.2f",
                 cv["oof_mae_pct_baseline"], cv["oof_mae_pct_stack"],
                 cv["oof_mae_pct_blend"], cv["blend_alpha"])

        cv_save = {k: v for k, v in cv.items() if k != "iso"}
        (ARTIFACT_DIR / "cv_q1.json").write_text(
            json.dumps(cv_save, indent=2, ensure_ascii=False), encoding="utf-8")

        test_sorted = test_df.sort_values(DATETIME_COL).reset_index(drop=True)
        log.info("Full train + predict...")
        mw_stack, mw_final = train_full(train_df, test_sorted, feature_cols, cat_cols, cv)

        p = save_submission(mw_final, "submission_q1.csv")
        log.info("DONE -> %s", p)


def main():
    Q1Ensemble().run()


if __name__ == "__main__":
    main()
