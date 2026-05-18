import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import random
import warnings

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

INSTALLED_CAPACITY = 90.09
TURBINE_CAPACITY = 3.465
N_TURBINES = 26
TARGET = "Выработка. Результирующий расчет"
DATETIME_COL = "METEOFORECASTHOUR_OPENM_Datetime"

# =============================================================================
# НАСТРОЙКИ
# =============================================================================
SHIFT = 0
USE_EXTRA_PHYSICS = True
USE_MONTH_HOUR_INTERACTION = True
USE_CLUSTER_DISTANCES = True
# =============================================================================


def apply_time_shift(df, shift=SHIFT):
    df = df.copy().sort_values(DATETIME_COL).reset_index(drop=True)
    exclude_cols = [DATETIME_COL, TARGET, "Кол-во_ВЭУ_в_ремонте", "month", "hour_of_day"]
    meteo_cols = [c for c in df.columns if c not in exclude_cols]
    if shift != 0:
        df[meteo_cols] = df[meteo_cols].shift(-shift).ffill().bfill()
    return df


def calc_physical_base(df):
    df["max_capacity"] = (N_TURBINES - df["Кол-во_ВЭУ_в_ремонте"]) * TURBINE_CAPACITY
    df["capacity_ratio"] = df["max_capacity"] / INSTALLED_CAPACITY
    df['wind_speed_84m'] = df['wind_speed_80m'] + (4/40) * (df['wind_speed_120m'] - df['wind_speed_80m'])

    R_air = 287.058
    df["air_density"] = df["pressure_msl"] * 100 / (R_air * (df["temperature_80m"] + 273.15))
    df["air_density"] = df["air_density"].fillna(df["air_density"].mean())

    A = 13685
    CP = 0.45
    df["v_cubed"] = df["wind_speed_84m"] ** 3
    df["power_flux"] = 0.5 * df["air_density"] * A * df["v_cubed"] * CP / 1e6
    df['physics_prediction'] = np.clip(df['power_flux'] * df['capacity_ratio'] * N_TURBINES, 0, df['max_capacity'])

    df.loc[df["wind_speed_84m"] < 3.0, "physics_prediction"] = 0
    df['steep_zone'] = ((df['wind_speed_84m'] >= 6) & (df['wind_speed_84m'] <= 12)).astype(int)
    df['pitch_control'] = (df['wind_speed_84m'] > 13).astype(int)
    return df


def calc_aerodynamics_and_wake(df):
    df["wind_shear_exponent"] = np.log((df["wind_speed_120m"] + 0.1) / (df["wind_speed_84m"] + 0.1)) / np.log(120/84)
    wind_std = df['wind_speed_84m'].rolling(3, min_periods=1).std().fillna(0)
    wind_mean = df['wind_speed_84m'].rolling(3, min_periods=1).mean().replace(0, 0.1)
    df["turbulence_intensity"] = wind_std / wind_mean
    df["gust_factor"] = df["wind_gusts_10m"] / (df["wind_speed_10m"] + 0.1)
    deg = df["wind_direction_80m"] * 1000
    df["wind_sector"] = np.floor(deg / 22.5).astype(int)
    rad = np.radians(deg)
    df["wind_dir_sin"] = np.sin(rad)
    df["wind_dir_cos"] = np.cos(rad)
    return df


def calc_thermodynamics(df):
    df["lapse_rate"] = (df["temperature_120m"] - df["temperature_80m"]) / (120 - 80)
    df["stability_class"] = np.select(
        [(df["lapse_rate"] < -0.01), (df["lapse_rate"] > 0)],
        ["Unstable", "Stable"], default="Neutral")
    precip = df["rain"] + df["snowfall"]
    df["icing_heuristic"] = ((df["temperature_80m"] >= -5) & (df["temperature_80m"] <= 2) & (precip > 0)).astype(int)

    # Прокси морского бриза
    df['sea_breeze_proxy'] = ((df['lapse_rate'] > 0.01) &
                              (df['hour_of_day'] >= 10) &
                              (df['hour_of_day'] <= 18)).astype(int)

    # Направление ветра с моря
    df['wind_from_sea'] = ((df['wind_sector'] >= 12) & (df['wind_sector'] <= 16)).astype(int)

    # Взаимодействие морского ветра и его силы
    df['sea_wind_interact'] = df['wind_from_sea'] * df['wind_speed_84m']

    # Улучшенный индикатор обледенения
    df['icing_risk_high'] = ((df['temperature_80m'] >= -4) &
                             (df['temperature_80m'] <= 1) &
                             ((df['rain'] + df['showers']) > 0) &
                             (df['cloud_cover_low'] > 0.5) &
                             (df['pressure_msl'] > 1015)).astype(int)

    # Сильный восточный ветер
    df['east_wind_storm'] = ((df['wind_sector'] >= 2) &
                             (df['wind_sector'] <= 4) &
                             (df['wind_speed_84m'] > 10)).astype(int)

    return df


def calc_temporal_dynamics(df):
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['wind_accel_1h'] = df['wind_speed_84m'].diff(1).fillna(0)
    df['pressure_diff_3h'] = df['pressure_msl'].diff(3).fillna(0)
    for lag in [1, 2, 3, 6]:
        df[f'wind_lag_{lag}'] = df['wind_speed_84m'].shift(lag).bfill()
        df[f'temp_lag_{lag}'] = df['temperature_80m'].shift(lag).bfill()
    for lead in [1, 2, 3]:
        df[f'wind_lead_{lead}'] = df['wind_speed_84m'].shift(-lead).ffill()
    df['wind_roll_mean_12'] = df['wind_speed_84m'].rolling(12, min_periods=1).mean()
    df['wind_roll_max_24'] = df['wind_speed_84m'].rolling(24, min_periods=1).max()
    df['temp_lag_24'] = df['temperature_80m'].shift(24).bfill()
    df['wind_dir_change_3h'] = ((df['wind_direction_80m'] - df['wind_direction_80m'].shift(3)) * 1000).abs().fillna(0)

    if USE_EXTRA_PHYSICS:
        df['wind_momentum'] = df['wind_speed_84m'] - df['wind_speed_84m'].shift(3).bfill()
        smooth_power = df['physics_prediction'].rolling(3, min_periods=1).mean()
        df['power_gradient'] = smooth_power.diff(1).fillna(0)
    if USE_MONTH_HOUR_INTERACTION:
        df['month_hour_sin_interact'] = df['month_sin'] * df['hour_sin']
        df['month_hour_cos_interact'] = df['month_cos'] * df['hour_cos']
    return df


def build_unsupervised_features(train_df, test_df):
    print("🧠 KMeans и кластерные расстояния...")
    clust_cols = ['wind_speed_84m', 'wind_accel_1h', 'turbulence_intensity', 'pressure_diff_3h', 'temperature_80m']
    scaler = StandardScaler()
    X_train_clust = scaler.fit_transform(train_df[clust_cols])
    X_test_clust = scaler.transform(test_df[clust_cols])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    train_df['regime'] = kmeans.fit_predict(X_train_clust).astype(str)
    test_df['regime'] = kmeans.predict(X_test_clust).astype(str)
    if USE_CLUSTER_DISTANCES:
        for i in range(5):
            train_df[f'cluster_dist_{i}'] = np.linalg.norm(X_train_clust - kmeans.cluster_centers_[i], axis=1)
            test_df[f'cluster_dist_{i}'] = np.linalg.norm(X_test_clust - kmeans.cluster_centers_[i], axis=1)
    return train_df, test_df


def generate_all_features(train_raw, test_raw):
    print("🛠 [1/3] Временной сдвиг...")
    train = apply_time_shift(train_raw, SHIFT)
    test = apply_time_shift(test_raw, SHIFT)

    print("⚙️ [2/3] Генерация физики и динамики...")

    def enrich(df):
        df = calc_physical_base(df)
        df = calc_aerodynamics_and_wake(df)
        df = calc_thermodynamics(df)
        df = calc_temporal_dynamics(df)
        df = df.bfill().ffill().fillna(0)
        return df

    train = enrich(train)
    test = enrich(test)

    print("🤖 [3/3] Unsupervised фичи...")
    train, test = build_unsupervised_features(train, test)

    CAT_COLS = ['wind_sector', 'stability_class', 'regime', 'hour_of_day',
                'month', 'Кол-во_ВЭУ_в_ремонте']

    exclude = [TARGET, DATETIME_COL, 'is_zero']
    FEATURE_COLS = [c for c in train.columns if c not in exclude]

    for col in CAT_COLS:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)

    print(f"✅ Готово! Признаков: {len(FEATURE_COLS)}.")
    return train, test, FEATURE_COLS, CAT_COLS
