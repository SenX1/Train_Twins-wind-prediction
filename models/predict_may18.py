"""
Прогноз на 18 мая: дообучение на расширенном трейне (train + april_may) и инференс на 24 часа.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from features.physics import generate_all_features, TARGET, DATETIME_COL  # noqa: E402
from models.gpu_utils import assert_gpu_available  # noqa: E402
from models.ensemble_q1 import (  # noqa: E402
    MANUAL_PARAMS, RANDOM_SEED, N_BASELINE, N_DIRECT, N_RESID,
    make_models, fit_one, predict_ensemble, cf_to_mw, _find_blend_alpha,
)

OUTPUT_DIR = ROOT / "outputs"


class May18Predictor:
    def run(self):
        assert_gpu_available()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Загрузка данных
        train_main = pd.read_csv(ROOT / "data" / "train_dataset.csv").dropna(subset=[TARGET])
        new_data = pd.read_csv(ROOT / "data" / "april_may_data.csv")

        # Разделение нового файла
        if TARGET in new_data.columns:
            train_april = new_data[new_data[TARGET].notna()].copy()
            test_may = new_data[new_data[TARGET].isna()].copy()
            test_may = test_may.drop(columns=[TARGET])
        else:
            train_april = pd.DataFrame(columns=new_data.columns)
            test_may = new_data.copy()

        if DATETIME_COL not in test_may.columns:
            test_may[DATETIME_COL] = pd.date_range("2026-05-18 00:00", periods=len(test_may), freq="h")

        train_raw = pd.concat([train_main, train_april], ignore_index=True)
        print(f"Расширенный трейн: {train_raw.shape[0]} записей, тест 18 мая: {test_may.shape[0]} записей")

        # 2. Генерация признаков
        train_df, test_df, feature_cols, cat_cols = generate_all_features(train_raw, test_may)

        # 3. Подготовка таргетов
        train_df["physics_cf"] = np.clip(train_df["physics_prediction"] / train_df["max_capacity"], 0, 1.2)
        train_df["cf_target"] = np.clip(train_df[TARGET] / train_df["max_capacity"], 0, 1)
        train_df["cf_residual"] = train_df["cf_target"] - train_df["physics_cf"]
        test_df["physics_cf"] = np.clip(test_df["physics_prediction"] / test_df["max_capacity"], 0, 1.2)

        X = train_df[feature_cols]
        y_cf = train_df["cf_target"].values
        res = train_df["cf_residual"].values

        # 4. Обучение ансамблей
        print("Обучение baseline ансамбля...")
        base_models = make_models(N_BASELINE, RANDOM_SEED)
        for m in base_models:
            fit_one(m, X, y_cf, cat_cols)

        print("Обучение direct ансамбля...")
        direct_models = make_models(N_DIRECT, RANDOM_SEED + 20)
        for m in direct_models:
            fit_one(m, X, y_cf, cat_cols)

        print("Обучение residual ансамбля...")
        res_models = make_models(N_RESID, RANDOM_SEED + 40)
        for m in res_models:
            fit_one(m, X, res, cat_cols)

        # 5. Мета-модель Ridge + Isotonic
        phys_cf = train_df["physics_cf"].values
        pred_direct = predict_ensemble(direct_models, X)
        pred_res = predict_ensemble(res_models, X)
        stack_cf_raw = np.clip(pred_direct + pred_res, 0, 1)

        X_meta = np.column_stack([phys_cf, pred_direct, pred_res, stack_cf_raw])
        ridge = Ridge(alpha=0.5, positive=True, fit_intercept=True, random_state=RANDOM_SEED)
        ridge.fit(X_meta, y_cf)
        stack_cf_cal = np.clip(X_meta @ ridge.coef_ + ridge.intercept_, 0, 1)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(stack_cf_cal, y_cf)

        # 6. Прогноз на 18 мая
        X_te = test_df[feature_cols]
        phys_te = test_df["physics_cf"].values
        max_cap_te = test_df["max_capacity"].values

        base_pred = predict_ensemble(base_models, X_te)
        direct_pred = predict_ensemble(direct_models, X_te)
        res_pred = predict_ensemble(res_models, X_te)

        stack_raw_te = np.clip(direct_pred + res_pred, 0, 1)
        X_meta_te = np.column_stack([phys_te, direct_pred, res_pred, stack_raw_te])
        stack_cf_te = np.clip(X_meta_te @ ridge.coef_ + ridge.intercept_, 0, 1)
        stack_cf_te = iso.predict(stack_cf_te)

        mw_base = cf_to_mw(base_pred, max_cap_te)
        mw_stack = cf_to_mw(stack_cf_te, max_cap_te)

        # Подбор blend_alpha
        a = _find_blend_alpha(
            train_df[TARGET].values,
            cf_to_mw(predict_ensemble(base_models, X), train_df["max_capacity"].values),
            cf_to_mw(stack_cf_cal, train_df["max_capacity"].values)
        )
        print(f"Оптимальный blend_alpha = {a:.3f}")

        mw_final = np.clip(a * mw_stack + (1 - a) * mw_base, 0, max_cap_te)

        # Сохранение сабмита
        submission = mw_final[::-1]
        output_file = OUTPUT_DIR / "submission_may18.csv"
        pd.DataFrame(submission, columns=["Prediction"]).to_csv(output_file, index=False, float_format="%.6f")
        print(f"✅ Прогноз на 18 мая сохранён: {output_file}")


def main():
    May18Predictor().run()


if __name__ == "__main__":
    main()
