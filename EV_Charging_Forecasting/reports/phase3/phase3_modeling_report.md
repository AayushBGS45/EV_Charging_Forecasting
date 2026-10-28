# Phase 3 Modeling Report: Time Series Forecasting of Non-Stationary EV Charging Demand

**Project:** Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models  
**Evaluated Dataset:** `data/processed_v2/caltech_hourly_demand_v2.parquet`  
**Evaluation Period:** March 1, 2020 – December 5, 2020 (Test Set: 6,700 hours / 279 daily Day-Ahead forecast origins)  
**Date:** October 2026  

---

## 1. Executive Summary & Master Results

Phase 3 implements, validates, and compares four distinct forecasting methodologies against corrected empirical baselines on the audited Caltech ACN hourly charging demand dataset (`caltech_hourly_demand_v2.parquet`). All model selection and hyperparameter tuning were conducted strictly on pre-COVID validation data (January 1, 2020 to February 29, 2020), and all models were subsequently evaluated on identical test timestamps from March 1, 2020 to December 5, 2020.

The primary task is **24-Hour Day-Ahead Forecasting ($H=24$)** originating daily at 00:00 UTC without intra-day telemetry feedback. The secondary task is **Rolling 1-Step-Ahead Forecasting ($H=1$)** with immediate hourly feedback.

### Master Results: Day-Ahead ($H=24$) Primary Task (Test Set: $N=6,696$ hours across 279 days)

| Rank | Model | MAE (kW) | RMSE (kW) | Filtered MAPE ($\ge 1\text{ kW}$) | sMAPE (%) | Validation H=24 MAE | Optimal Configuration / Specification |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **Chronos (chronos-t5-tiny)** | **5.87** | 8.72 | **58.46%** | 61.20% | 9.18 kW | Pretrained Transformer, Context = 504h (3 weeks) |
| **2** | **SARIMA** | 5.91 | **8.51** | 66.40% | **58.73%** | 9.34 kW | $\text{SARIMA}(2, 0, 1) \times (0, 1, 1)_{24}$ (Kalman filtered) |
| **3** | **LightGBM GBDT** | 6.68 | 9.28 | 78.81% | 60.29% | 8.00 kW | L2 loss, lr=0.05, num_leaves=31, 22 causal features |
| 4 | Seasonal Naive (Weekly $H=24$) | 7.33 | 10.54 | 79.85% | 71.20% | 11.63 kW | Lag 168h reference (same day/hour last week) |
| 5 | Seasonal Naive (Diurnal $H=24$) | 7.33 | 10.60 | 75.76% | 69.54% | 12.00 kW | Lag 24h reference (previous day profile) |
| 6 | Holt-Winters Exp. Smoothing | 10.96 | 15.10 | 148.84% | 95.83% | 11.99 kW | Damped additive trend + Additive diurnal seasonality |
| 7 | Persistence (Flat $H=24$) | 13.52 | 18.13 | 213.64% | 83.04% | 28.41 kW | Flat forward propagation from $t-1$ |

### Master Results: Rolling 1-Step-Ahead ($H=1$) Secondary Task (Test Set: $N=6,700$ hours)

| Rank | Model | MAE (kW) | RMSE (kW) | Filtered MAPE ($\ge 1\text{ kW}$) | sMAPE (%) | Validation H=1 MAE | Key Mechanism |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **Chronos (chronos-t5-tiny)** | **3.33** | **5.10** | **31.28%** | 31.88% | — | Zero-shot deep autoregressive self-attention |
| **2** | **LightGBM GBDT** | 3.42 | 5.14 | 33.95% | 37.11% | 4.24 kW | Causal lags (1..4, 24, 168) + 3h/6h/24h rolling stats |
| **3** | **SARIMA** | 3.48 | 5.11 | 36.29% | 41.18% | 4.38 kW | Kalman filter 1-step updating with $Q=1$ |
| 4 | Persistence ($t-1$) | 3.60 | 5.67 | 33.21% | **30.31%** | 5.58 kW | Immediate previous hour observation |
| 5 | Holt-Winters Exp. Smoothing | 4.12 | 5.82 | 48.32% | 54.14% | 4.92 kW | Damped additive trend recursive filter |
| 6 | Seasonal Naive ($t-24$) | 7.33 | 10.59 | 75.76% | 69.53% | 12.00 kW | Diurnal lag |
| 7 | Seasonal Naive ($t-168$) | 7.33 | 10.53 | 79.85% | 71.27% | 11.63 kW | Weekly lag |

---

## 2. Dataset Caveats and Methodological Documentation

> [!IMPORTANT]
> **1. Uniform Energy Allocation vs. Physical Instantaneous Power:**  
> The target variable `avg_power_kw` represents hourly aggregate facility energy consumption (where $1\text{ kWh/hour} \equiv 1\text{ kW}$ mean demand) reconstructed by uniformly spreading each charging session's delivered energy across its connection window. It does **not** reflect measured instantaneous charging power or real-time intra-hour fluctuations (such as constant-current / constant-voltage battery tapering, station power throttling, or dynamic curtailment).
>
> **2. Provisional Nature of the 7.7 kW Power Cap:**  
> The 7.7 kW per-session cap applied during Phase 2.1 data cleaning was derived from standard Level 2 AC EVSE electrical ceilings (32 A at 240 V). While it successfully suppresses severe reconnection-loop meter artefacts (reducing the maximum demand spike from 375.4 kW to 120.6 kW), station-specific hardware engineering specifications were not independently verified against physical electrical drawings. The cap remains provisional.
>
> **3. Corrected Dataset Baseline Benchmarks:**  
> All model comparisons are evaluated against baselines computed on the corrected v2 dataset (`caltech_hourly_demand_v2.parquet`), rather than the uncorrected v1 baselines.

### Chronological Splitting Strategy

A strictly chronological, non-overlapping split was preserved:
- **Training Set:** May 1, 2018 00:00 UTC – December 31, 2019 23:00 UTC ($14,640$ hours, 609 days)
- **Validation Set:** January 1, 2020 00:00 UTC – February 29, 2020 23:00 UTC ($1,440$ hours, 60 days)
- **Test Set:** March 1, 2020 00:00 UTC – December 5, 2020 03:00 UTC ($6,700$ hours, 279 full days + 4 hours)

All lag features, rolling statistics, and state updates are strictly causal (using only observations strictly prior to the forecast origin).

---

## 3. Model Architecture, Diagnostics, and Selection Process

### 3.1 Model 1: Holt-Winters Exponential Smoothing

- **Script:** [`train_holt_winters.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/train_holt_winters.py)
- **Validation Tuning:** Tested three candidate specifications on the 60 validation days (Jan–Feb 2020):
  1. `HW_NoTrend_AddSeason24`: Validation H=24 MAE = 17.47 kW, RMSE = 22.78 kW
  2. `HW_AddTrend_AddSeason24`: Validation H=24 MAE = 12.15 kW, RMSE = 17.39 kW
  3. `HW_DampedTrend_AddSeason24`: Validation H=24 MAE = **11.99 kW**, RMSE = **17.36 kW**
- **Selected Configuration:** `HW_DampedTrend_AddSeason24` ($\alpha = 0.9197, \beta = 0.0000, \gamma = 0.0803, \phi = 0.8217$).
- **Test Set Performance:**
  - Day-Ahead ($H=24$): MAE = **10.96 kW**, RMSE = **15.10 kW**, sMAPE = **95.83%**
  - Rolling 1-Step ($H=1$): MAE = **4.12 kW**, RMSE = **5.82 kW**, sMAPE = **54.14%**
- **Diagnostic Finding:** Standard single-season Holt-Winters struggles under sudden non-stationary structural shocks because exponential smoothing carries forward historical seasonal amplitudes without multi-period (weekly) or exogenous calendar context.

### 3.2 Model 2: SARIMA with Training-Only Diagnostics

- **Script:** [`train_sarima.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/train_sarima.py)
- **Training-Only Diagnostics:**
  - **ADF on Level Series:** Statistic = $-12.55$ ($p = 2.2 \times 10^{-23}$), but **KPSS Statistic:** $2.44$ ($p = 0.01$), rejecting trend-stationarity due to pronounced diurnal/weekly seasonality and long-term trend shifts.
  - **Seasonal Differencing ($D=1, s=24$):** ADF Statistic = **$-24.48$** ($p = 0.0$), KPSS Statistic = **$0.04$** ($p > 0.10$), establishing strong covariance-stationarity.
  - **ACF / PACF:** Seasonally differenced training series displayed significant cutoffs at non-seasonal lags 1–2 and strong seasonal spikes at lags 24 and 48, indicating $p \in \{1, 2\}$, $q \in \{0, 1\}$, and $Q \in \{0, 1\}$.
- **Validation Order Selection (Jan–Feb 2020):**
  1. $\text{SARIMA}(1, 0, 0) \times (1, 1, 0)_{24}$: AIC = 96,776.8, Val H=24 MAE = 10.84 kW
  2. $\text{SARIMA}(1, 0, 1) \times (1, 1, 0)_{24}$: AIC = 96,581.2, Val H=24 MAE = 10.58 kW
  3. $\text{SARIMA}(1, 0, 1) \times (0, 1, 1)_{24}$: AIC = 91,454.0, Val H=24 MAE = 9.58 kW
  4. $\text{SARIMA}(2, 0, 1) \times (1, 1, 0)_{24}$: AIC = 96,488.3, Val H=24 MAE = 10.27 kW
  5. $\text{SARIMA}(2, 0, 1) \times (0, 1, 1)_{24}$: AIC = **91,384.3**, Val H=24 MAE = **9.34 kW** (RMSE = 13.72 kW)
- **Selected Configuration:** $\text{SARIMA}(2, 0, 1) \times (0, 1, 1)_{24}$
- **Test Set Performance:**
  - Day-Ahead ($H=24$): MAE = **5.91 kW**, RMSE = **8.51 kW**, sMAPE = **58.73%**
  - Rolling 1-Step ($H=1$): MAE = **3.48 kW**, RMSE = **5.11 kW**, sMAPE = **41.18%**
- **Diagnostic Finding:** Incorporating a seasonal moving-average term ($Q=1$) reduced validation AIC by over 5,000 points and improved day-ahead MAE significantly. The Kalman filter update mechanism allowed SARIMA to adapt rapidly across origin points without parameter refitting.

### 3.3 Model 3: LightGBM GBDT (Causal Lag & Calendar Features)

- **Script:** [`train_lightgbm.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/train_lightgbm.py)
- **Feature Architecture:**
  - *Calendar & Cyclical:* `hour`, `day_of_week`, `is_weekend`, `month`, `sin_hour`, `cos_hour`, `sin_dow`, `cos_dow`, `step` (1..24).
  - *Causal Same-Hour Lags:* `lag_24`, `lag_48`, `lag_72`, `lag_168`, `lag_336` (strictly observed prior to origin 00:00 UTC).
  - *Origin-Level Rolling Statistics:* Mean, std, min, max over $[t-24, t-1]$, mean and std over $[t-168, t-1]$, last observed demand $y_{t-1}$, and 24h momentum delta.
- **Validation Tuning:**
  - Evaluated candidate objectives (`regression_l1` vs. `regression` L2), learning rates (0.03, 0.05), and tree depths.
  - Selected for Day-Ahead ($H=24$): `LGBM_L2_lr05_nl31` (Best iteration: 169, Val MAE = **8.00 kW**, RMSE = 11.52 kW).
  - Selected for Rolling 1-Step ($H=1$): `LGBM_H1_L1_lr05_nl31` (Best iteration: 444, Val MAE = **4.24 kW**, RMSE = 6.58 kW).
- **Test Set Performance:**
  - Day-Ahead ($H=24$): MAE = **6.68 kW**, RMSE = **9.28 kW**, sMAPE = **60.29%**
  - Rolling 1-Step ($H=1$): MAE = **3.42 kW**, RMSE = **5.14 kW**, sMAPE = **37.11%**
- **Feature Importance:** Top features driving Day-Ahead forecast gain were `origin_mean_24` (24h origin rolling level), `hour` (diurnal phase), `lag_24` (yesterday's demand), `origin_mean_168` (weekly baseline level), and `lag_168` (weekly seasonal anchor).

### 3.4 Model 4: Chronos Foundation Time Series Model

- **Script:** [`train_chronos.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/train_chronos.py)
- **Architecture:** `amazon/chronos-t5-tiny` zero-shot pretrained transformer model for probabilistic time series forecasting. Point forecasts generated via sample median quantile ($q=0.50$).
- **Validation Context Window Tuning:**
  - Context 168 hours (1 week): Val H=24 MAE = 10.53 kW, RMSE = 16.51 kW
  - Context 336 hours (2 weeks): Val H=24 MAE = 9.74 kW, RMSE = 15.02 kW
  - Context 504 hours (3 weeks): Val H=24 MAE = **9.18 kW**, RMSE = **14.39 kW**
- **Selected Context:** **504 hours (3 weeks)**
- **Test Set Performance:**
  - Day-Ahead ($H=24$): MAE = **5.87 kW**, RMSE = **8.72 kW**, Filtered MAPE = **58.46%**, sMAPE = **61.20%**
  - Rolling 1-Step ($H=1$): MAE = **3.33 kW**, RMSE = **5.10 kW**, Filtered MAPE = **31.28%**, sMAPE = **31.88%**
- **Diagnostic Finding:** Chronos achieved the lowest overall test MAE in both Day-Ahead (5.87 kW) and Rolling 1-Step (3.33 kW) regimes. The 3-week context window provided sufficient memory for the self-attention mechanism to recognize multi-day rhythms and adjust its zero-shot scaling factor to post-lockdown demand shifts.

---

## 4. Seasonal Breakdown Analysis (COVID Trajectory)

The test set covers the unprecedented demand collapse and partial resumption of 2020. Evaluating performance across distinct phases reveals how models respond to non-stationarity:

### Day-Ahead ($H=24$) Performance by Sub-Period (MAE in kW)

| Sub-Period | Persistence Flat | Seasonal Naive Diurnal | Seasonal Naive Weekly | Holt-Winters | LightGBM | SARIMA | Chronos |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **March–May 2020** (Lockdown shock) | 11.54 | 6.15 | 6.81 | 7.09 | 6.66 | 5.66 | **5.23** |
| **June–August 2020** (Summer low plateau) | 12.53 | 7.41 | 6.98 | 14.43 | 6.41 | **5.42** | 5.75 |
| **Sept–Dec 2020** (Fall partial restart) | 16.40 | 8.39 | 8.17 | 11.34 | 6.98 | 6.62 | **6.61** |

### Rolling 1-Step ($H=1$) Performance by Sub-Period (MAE in kW)

| Sub-Period | Persistence ($t-1$) | Seasonal Naive ($t-24$) | Seasonal Naive ($t-168$) | Holt-Winters | SARIMA | LightGBM | Chronos |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **March–May 2020** | 2.67 | 6.15 | 6.81 | 3.45 | 2.87 | 2.68 | **2.51** |
| **June–August 2020** | 3.67 | 7.41 | 6.98 | 4.25 | 3.51 | 3.59 | **3.47** |
| **Sept–Dec 2020** | 4.41 | 8.38 | 8.16 | 4.64 | 4.03 | **3.98** | 4.00 |

### Key Observations:
1. **The March–May Shock:** When Caltech closed campus mid-March 2020, facility demand dropped from ~50 kW peak down to ~15 kW peak. Chronos adapted fastest to this sharp regime collapse, achieving the lowest shock-period Day-Ahead MAE (5.23 kW).
2. **The Summer Plateau:** During June–August, demand was low and stable. SARIMA performed best in this regime (5.42 kW MAE / 7.35 kW RMSE) because its autoregressive differencing perfectly tracked the subdued, consistent cycle.
3. **The Fall Resumption:** As activity partially resumed in September–December, demand variance rose. Chronos (6.61 kW) and SARIMA (6.62 kW) remained virtually tied, outperforming LightGBM (6.98 kW) and the seasonal baselines (>8.1 kW).

---

## 5. Timestamp Alignment and Verification Audit

To ensure scientific integrity and eliminate evaluation bias, all test evaluations were audited for exact timestamp synchronization:

| Model Output File | Task | Timestamps ($N$) | Min Timestamp | Max Timestamp | Reference Match |
|:---|:---:|:---:|:---:|:---:|:---:|
| `baseline_v2_h24_predictions.parquet` | Day-Ahead ($H=24$) | 6,696 | 2020-03-01 00:00 UTC | 2020-12-04 23:00 UTC | ✅ MATCH |
| `holt_winters_h24_predictions.parquet` | Day-Ahead ($H=24$) | 6,696 | 2020-03-01 00:00 UTC | 2020-12-04 23:00 UTC | ✅ MATCH |
| `sarima_h24_predictions.parquet` | Day-Ahead ($H=24$) | 6,696 | 2020-03-01 00:00 UTC | 2020-12-04 23:00 UTC | ✅ MATCH |
| `lightgbm_h24_predictions.parquet` | Day-Ahead ($H=24$) | 6,696 | 2020-03-01 00:00 UTC | 2020-12-04 23:00 UTC | ✅ MATCH |
| `chronos_h24_predictions.parquet` | Day-Ahead ($H=24$) | 6,696 | 2020-03-01 00:00 UTC | 2020-12-04 23:00 UTC | ✅ MATCH |
| `baseline_v2_h1_predictions.parquet` | Rolling 1-Step ($H=1$) | 6,700 | 2020-03-01 00:00 UTC | 2020-12-05 03:00 UTC | ✅ MATCH |
| `holt_winters_h1_predictions.parquet` | Rolling 1-Step ($H=1$) | 6,700 | 2020-03-01 00:00 UTC | 2020-12-05 03:00 UTC | ✅ MATCH |
| `sarima_h1_predictions.parquet` | Rolling 1-Step ($H=1$) | 6,700 | 2020-03-01 00:00 UTC | 2020-12-05 03:00 UTC | ✅ MATCH |
| `lightgbm_h1_predictions.parquet` | Rolling 1-Step ($H=1$) | 6,700 | 2020-03-01 00:00 UTC | 2020-12-05 03:00 UTC | ✅ MATCH |
| `chronos_h1_predictions.parquet` | Rolling 1-Step ($H=1$) | 6,700 | 2020-03-01 00:00 UTC | 2020-12-05 03:00 UTC | ✅ MATCH |

Zero timestamps were misaligned, skipped, or duplicated.

---

## 6. Generated Artifacts and Figure Links

### Scripts (Executable & Reproducible)
- Evaluation Harness: [`scripts/phase3_eval_utils.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/phase3_eval_utils.py)
- Baselines (v2): [`scripts/evaluate_v2_baselines.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/evaluate_v2_baselines.py)
- Holt-Winters: [`scripts/train_holt_winters.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/train_holt_winters.py)
- SARIMA: [`scripts/train_sarima.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/train_sarima.py)
- LightGBM: [`scripts/train_lightgbm.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/train_lightgbm.py)
- Chronos: [`scripts/train_chronos.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/train_chronos.py)
- Synthesis & Alignment: [`scripts/evaluate_phase3_models.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/evaluate_phase3_models.py)

### Quantitative Metric Files
- Master Comparison: [`reports/phase3/phase3_models_comparison.csv`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/phase3_models_comparison.csv)
- Seasonal Breakdown Comparison: [`reports/phase3/phase3_seasonal_breakdown_comparison.csv`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/phase3_seasonal_breakdown_comparison.csv)
- Timestamp Alignment Audit: [`reports/phase3/phase3_timestamp_alignment_audit.csv`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/phase3_timestamp_alignment_audit.csv)
- Model Configurations:
  - [`reports/phase3/holt_winters_config.json`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/holt_winters_config.json)
  - [`reports/phase3/sarima_config.json`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/sarima_config.json)
  - [`reports/phase3/lightgbm_config.json`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/lightgbm_config.json)
  - [`reports/phase3/chronos_config.json`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/chronos_config.json)
- Feature Importance: [`reports/phase3/lightgbm_feature_importance.csv`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/lightgbm_feature_importance.csv)
- Training Diagnostics: [`reports/phase3/sarima_diagnostics_summary.csv`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/sarima_diagnostics_summary.csv)

### Visualizations
- Master Day-Ahead Forecasts Overlay: [`reports/phase3/figures/phase3_day_ahead_comparison.png`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/figures/phase3_day_ahead_comparison.png)
- Master Metrics Summary Bar Chart: [`reports/phase3/figures/phase3_metrics_summary_barchart.png`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/figures/phase3_metrics_summary_barchart.png)
- Seasonal Breakdown Bar Chart: [`reports/phase3/figures/phase3_seasonal_breakdown.png`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/figures/phase3_seasonal_breakdown.png)
- Rolling 1-Step Forecast Overlay: [`reports/phase3/figures/phase3_rolling_1step_comparison.png`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/figures/phase3_rolling_1step_comparison.png)
- LightGBM Feature Importance: [`reports/phase3/figures/lightgbm_feature_importance.png`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/figures/lightgbm_feature_importance.png)
- SARIMA Training ACF/PACF Diagnostics: [`reports/phase3/figures/sarima_training_acf_pacf.png`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/phase3/figures/sarima_training_acf_pacf.png)

