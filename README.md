# Wind Power Generation Forecasting

*Team Train Twins | Wind Power Generation Forecasting Hackathon*

A machine learning solution for forecasting electricity generation at the **Azov Wind Farm (90.09 MW installed capacity)**.

The approach combines physics-informed feature engineering, time-series modeling, and an ensemble of 11 CatBoost models with Ridge stacking and isotonic calibration.

## Competition Results

Developed for the **Wind Power Generation Forecasting Hackathon**, using data from the Azov Wind Farm (90.09 MW installed capacity).

| Evaluation period | Result |
|---|---|
| Q1 2026 | **7.987% MAE** (normalized by installed capacity), competition leaderboard |
| Additional forecasting period | **2nd place** on the private leaderboard, MAE ≈ 2.7 (units to be confirmed) |

The solution combines physics-informed feature engineering with an ensemble of 11 CatBoost models, Ridge stacking, and isotonic calibration.

*Results are reported from separate competition evaluation periods and should not be treated as directly comparable.*

## Approach

### 1. Physics-Informed Feature Engineering

Designed meteorological and environmental features to capture factors affecting wind power generation:

- **Wind speed at hub height:** interpolation to 84 m.
- **Air density:** adjustments based on atmospheric conditions.
- **Wind power relationship:** cubic wind-speed features (V³).
- **Wind dynamics:** gusts, vertical wind shear, and atmospheric stability.
- **Local weather effects:** coastal breezes, icing conditions, and easterly storms relevant to the Azov Wind Farm.

### 2. Time-Series Features

Generated temporal features to capture historical patterns and short-term variations:

- Lagged observations
- Lead features
- Rolling statistics

### 3. Ensemble Architecture

The solution uses **11 CatBoost models**, organized into three groups:

| Model group | Number | Purpose |
|---|---:|---|
| Base models | 3 | Primary prediction models |
| Direct models | 5 | Additional direct forecasts |
| Residual models | 3 | Residual correction |

The predictions are combined using **Ridge regression stacking**, followed by **isotonic calibration**.

### 4. Validation

Used `TimeSeriesSplit` with a gap between training and validation periods to reduce temporal leakage.

**Q1 2026 validation MAE: 7.987% of installed capacity.**

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare datasets

Place the following files inside `data/`:

```text
data/
├── train_dataset.csv
├── valid_features.csv
└── april_may_data.csv
```

The datasets are not included in the repository.

### 3. Generate Q1 predictions

```bash
python pipelines/generate_submission_q1.py
```

Output: `outputs/submission_q1.csv`

### 4. Generate May 18 predictions

```bash
python pipelines/generate_submission_may18.py
```

Output: `outputs/submission_may18.csv`

## Repository Structure

```text
.
├── features/       # Feature engineering
├── models/         # Model training and inference
├── pipelines/      # Prediction entry points
├── data/           # Input datasets (not included)
├── outputs/        # Generated predictions
└── requirements.txt
```

## Reproducibility

- Random seeds are fixed to `42`.
- Model hyperparameters are defined in the source code.
- The pipeline is configured for deterministic execution.

**Note:** Reproduction requires the original competition datasets, which are not distributed with this repository.

---

*Developed by Team Train Twins for the Wind Power Generation Forecasting Hackathon.*
