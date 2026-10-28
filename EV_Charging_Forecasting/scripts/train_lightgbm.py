"""
Phase 3 - Model 3: LightGBM Gradient Boosted Decision Trees Pipeline
===================================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models

METHODOLOGY AND CAVEAT DOCUMENTATION:
- Target Series: 'avg_power_kw' represents hourly facility energy consumption (kWh/h = kW)
  reconstructed via uniform energy allocation across connection intervals.
  It is NOT measured instantaneous charging power. Intra-session charging dynamics (e.g., CC/CV
  tapering, station-level throttling) are smoothed out by definition.
- Provisional Cap: The 7.7 kW per-session power cap applied during v2 preprocessing remains
  provisional, based on standard Level 2 EVSE maximum delivery (32A @ 240V). While it successfully
  suppressed unphysical meter-carryover spikes (reducing the peak from 375.4 kW to 120.6 kW),
  station-level hardware electrical ratings were not verified against primary engineering spec sheets.
- Chronological Partition:
  * Training:   2018-05-01 00:00 to 2019-12-31 23:00 UTC (14,640 hours)
  * Validation: 2020-01-01 00:00 to 2020-02-29 23:00 UTC (1,440 hours)
  * Test:       2020-03-01 00:00 to 2020-12-05 03:00 UTC (6,700 hours)
- Causal Feature Engineering:
  * Zero Data Leakage: All lag and rolling window features use strictly information available prior
    to the forecast origin.
  * Day-Ahead (H=24): Uses only history <= t-1 (00:00 UTC origin). Features include same-hour lags
    (24h, 48h, 72h, 168h, 336h), origin-level rolling demand statistics (24h, 48h, 168h), lead step (1..24),
    and cyclical/calendar encodings.
  * Rolling 1-Step (H=1): Uses immediate lag feedback (lag 1, 2, 3, 4, 24, 168), short-term rolling
    windows (3h, 6h, 24h, 168h), recent momentum deltas, and calendar encodings.
- Model Selection & Hyperparameter Tuning:
  * Tuned strictly on the Validation set (Jan–Feb 2020).
  * No test set feedback is used during hyperparameter optimization or early stopping.
- Post-processing:
  * Non-negative clipping (demand >= 0 kW) is enforced physically.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from phase3_eval_utils import (
    get_project_root,
    get_phase3_reports_dir,
    load_v2_dataset,
    calculate_metrics,
    get_split_masks,
    get_daily_forecast_origins,
    get_test_subperiods,
)


# =====================================================================
# 1. Feature Engineering: Day-Ahead H=24 Task
# =====================================================================

def build_h24_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs causal day-ahead training/validation/test samples for H=24.
    For every daily origin t (00:00 UTC), generates 24 forecast steps (step 1..24)
    using only information available at origin t (history <= t-1).
    """
    print("[*] Building causal Day-Ahead (H=24) feature dataset...")
    total_len = len(df)
    # Candidate daily origins starting from day 15 (to allow 336h lag)
    daily_origins = df[(df["timestamp"].dt.hour == 0) & (df.index >= 336)].index.tolist()
    daily_origins = [t for t in daily_origins if t + 24 <= total_len]

    y_arr = df["avg_power_kw"].values
    ts_arr = df["timestamp"].values

    samples = []
    for t in daily_origins:
        origin_ts = pd.to_datetime(ts_arr[t], utc=True)
        # Origin-level statistics over history strictly up to t-1
        hist_24 = y_arr[t - 24 : t]
        hist_48 = y_arr[t - 48 : t]
        hist_168 = y_arr[t - 168 : t]

        origin_mean_24 = float(np.mean(hist_24))
        origin_std_24 = float(np.std(hist_24))
        origin_min_24 = float(np.min(hist_24))
        origin_max_24 = float(np.max(hist_24))
        origin_mean_168 = float(np.mean(hist_168))
        origin_std_168 = float(np.std(hist_168))
        origin_last_obs = float(y_arr[t - 1])
        origin_diff_24 = float(y_arr[t - 1] - y_arr[t - 25])

        for step in range(24):
            idx = t + step
            target_ts = pd.to_datetime(ts_arr[idx], utc=True)
            hour = target_ts.hour
            dow = target_ts.dayofweek
            month = target_ts.month
            is_wknd = 1 if dow >= 5 else 0

            # Causal lags relative to target hour (idx - 24 <= t - 1, so all strictly observed before origin t)
            lag_24 = float(y_arr[idx - 24])
            lag_48 = float(y_arr[idx - 48])
            lag_72 = float(y_arr[idx - 72])
            lag_168 = float(y_arr[idx - 168])
            lag_336 = float(y_arr[idx - 336])

            samples.append({
                "origin_timestamp": origin_ts,
                "timestamp": target_ts,
                "step": step + 1,
                "avg_power_kw": float(y_arr[idx]),
                # Calendar features
                "hour": hour,
                "day_of_week": dow,
                "is_weekend": is_wknd,
                "month": month,
                "sin_hour": np.sin(2.0 * np.pi * hour / 24.0),
                "cos_hour": np.cos(2.0 * np.pi * hour / 24.0),
                "sin_dow": np.sin(2.0 * np.pi * dow / 7.0),
                "cos_dow": np.cos(2.0 * np.pi * dow / 7.0),
                # Causal same-hour lags
                "lag_24": lag_24,
                "lag_48": lag_48,
                "lag_72": lag_72,
                "lag_168": lag_168,
                "lag_336": lag_336,
                # Origin summary features
                "origin_last_obs": origin_last_obs,
                "origin_mean_24": origin_mean_24,
                "origin_std_24": origin_std_24,
                "origin_min_24": origin_min_24,
                "origin_max_24": origin_max_24,
                "origin_mean_168": origin_mean_168,
                "origin_std_168": origin_std_168,
                "origin_diff_24": origin_diff_24,
            })

    df_h24 = pd.DataFrame(samples)
    feature_cols = [
        "step", "hour", "day_of_week", "is_weekend", "month",
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        "lag_24", "lag_48", "lag_72", "lag_168", "lag_336",
        "origin_last_obs", "origin_mean_24", "origin_std_24",
        "origin_min_24", "origin_max_24", "origin_mean_168",
        "origin_std_168", "origin_diff_24",
    ]
    print(f"[+] H=24 dataset built: {len(df_h24):,} rows, {len(feature_cols)} features.")
    return df_h24, feature_cols


# =====================================================================
# 2. Feature Engineering: Rolling 1-Step H=1 Task
# =====================================================================

def build_h1_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs causal 1-step-ahead (H=1) training/validation/test samples.
    At each step t, uses immediate lag feedback (y_{t-1}, y_{t-2}, etc.)
    and causal rolling statistics.
    """
    print("[*] Building causal Rolling 1-Step (H=1) feature dataset...")
    df = df.copy()

    # Calendar features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month
    df["sin_hour"] = np.sin(2.0 * np.pi * df["hour"] / 24.0)
    df["cos_hour"] = np.cos(2.0 * np.pi * df["hour"] / 24.0)
    df["sin_dow"] = np.sin(2.0 * np.pi * df["day_of_week"] / 7.0)
    df["cos_dow"] = np.cos(2.0 * np.pi * df["day_of_week"] / 7.0)

    # Causal lags (shift 1..168)
    for lag in [1, 2, 3, 4, 24, 48, 168]:
        df[f"lag_{lag}"] = df["avg_power_kw"].shift(lag)

    # Causal momentum deltas
    df["diff_1"] = df["lag_1"] - df["lag_2"]
    df["diff_24"] = df["lag_1"] - df["lag_24"]

    # Rolling window aggregations over past observations (shifted by 1 to prevent leakage)
    s1 = df["avg_power_kw"].shift(1)
    df["roll_mean_3h"] = s1.rolling(3).mean()
    df["roll_mean_6h"] = s1.rolling(6).mean()
    df["roll_mean_24h"] = s1.rolling(24).mean()
    df["roll_std_24h"] = s1.rolling(24).std()
    df["roll_mean_168h"] = s1.rolling(168).mean()
    df["roll_std_168h"] = s1.rolling(168).std()

    # Drop warm-up rows (first 168 hours)
    df_h1 = df.dropna().reset_index(drop=True)

    feature_cols = [
        "hour", "day_of_week", "is_weekend", "month",
        "sin_hour", "cos_hour", "sin_dow", "cos_dow",
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_24", "lag_48", "lag_168",
        "diff_1", "diff_24",
        "roll_mean_3h", "roll_mean_6h", "roll_mean_24h", "roll_std_24h",
        "roll_mean_168h", "roll_std_168h",
    ]
    print(f"[+] H=1 dataset built: {len(df_h1):,} rows, {len(feature_cols)} features.")
    return df_h1, feature_cols


# =====================================================================
# 3. Validation Tuning & Test Evaluation
# =====================================================================

def tune_and_train_lightgbm_h24(
    df_h24: pd.DataFrame,
    features: List[str],
    reports_dir: Path,
) -> Tuple[lgb.Booster, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Performs grid search on validation data for Day-Ahead H=24 task.
    Evaluates candidates strictly on validation origins (Jan-Feb 2020),
    trains the best model, and outputs predictions on test origins (Mar-Dec 2020).
    """
    print("\n--- Tuning LightGBM for Day-Ahead (H=24) on Validation Data (Jan-Feb 2020) ---")
    train_mask = df_h24["timestamp"] < "2020-01-01"
    val_mask = (df_h24["timestamp"] >= "2020-01-01") & (df_h24["timestamp"] < "2020-03-01")
    test_mask = df_h24["timestamp"] >= "2020-03-01"

    X_train = df_h24.loc[train_mask, features]
    y_train = df_h24.loc[train_mask, "avg_power_kw"].values
    X_val = df_h24.loc[val_mask, features]
    y_val = df_h24.loc[val_mask, "avg_power_kw"].values
    X_test = df_h24.loc[test_mask, features]
    y_test = df_h24.loc[test_mask, "avg_power_kw"].values

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    candidates = [
        {"name": "LGBM_L1_lr05_nl31", "objective": "regression_l1", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"name": "LGBM_L1_lr03_nl31", "objective": "regression_l1", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"name": "LGBM_L1_lr05_nl15", "objective": "regression_l1", "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 30, "colsample_bytree": 0.8},
        {"name": "LGBM_L2_lr05_nl31", "objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"name": "LGBM_L2_lr03_nl63", "objective": "regression", "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 20, "colsample_bytree": 0.8},
    ]

    tuning_records = []
    trained_models = {}

    for cand in candidates:
        params = {
            "objective": cand["objective"],
            "learning_rate": cand["learning_rate"],
            "num_leaves": cand["num_leaves"],
            "min_child_samples": cand["min_child_samples"],
            "colsample_bytree": cand["colsample_bytree"],
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": 4,
        }

        evals_result = {}
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=600,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.record_evaluation(evals_result),
            ],
        )

        val_preds = np.clip(model.predict(X_val), 0, None)
        m_val = calculate_metrics(y_val, val_preds)

        print(
            f"  {cand['name']:<22} | Best Iter: {model.best_iteration:>3} | "
            f"Val H=24 MAE: {m_val['mae_kw']:>6.2f} kW | RMSE: {m_val['rmse_kw']:>6.2f} kW | "
            f"sMAPE: {m_val['smape_pct']:>5.2f}%"
        )

        record = {
            "candidate_name": cand["name"],
            "task": "Day-Ahead (H=24)",
            "best_iteration": model.best_iteration,
            "val_mae_kw": m_val["mae_kw"],
            "val_rmse_kw": m_val["rmse_kw"],
            "val_mape_pct": m_val["mape_filtered_pct"],
            "val_smape_pct": m_val["smape_pct"],
            **cand,
        }
        tuning_records.append(record)
        trained_models[cand["name"]] = (model, cand)

    df_tuning = pd.DataFrame(tuning_records)
    best_row = df_tuning.sort_values("val_mae_kw").iloc[0]
    best_name = best_row["candidate_name"]
    best_model, best_config = trained_models[best_name]

    print(f"\n[+] Selected Best LightGBM (H=24) on Validation: {best_name}")
    print(f"    Validation MAE: {best_row['val_mae_kw']:.2f} kW | RMSE: {best_row['val_rmse_kw']:.2f} kW")

    # Predict on test set
    test_preds = np.clip(best_model.predict(X_test), 0, None)
    df_test_pred = df_h24.loc[test_mask, ["origin_timestamp", "timestamp", "step", "avg_power_kw"]].copy()
    df_test_pred["pred_lightgbm"] = test_preds

    return best_model, best_config, df_tuning, df_test_pred


def tune_and_train_lightgbm_h1(
    df_h1: pd.DataFrame,
    features: List[str],
) -> Tuple[lgb.Booster, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Performs validation tuning for Rolling 1-Step (H=1) task.
    """
    print("\n--- Tuning LightGBM for Rolling 1-Step (H=1) on Validation Data (Jan-Feb 2020) ---")
    train_mask = df_h1["timestamp"] < "2020-01-01"
    val_mask = (df_h1["timestamp"] >= "2020-01-01") & (df_h1["timestamp"] < "2020-03-01")
    test_mask = df_h1["timestamp"] >= "2020-03-01"

    X_train = df_h1.loc[train_mask, features]
    y_train = df_h1.loc[train_mask, "avg_power_kw"].values
    X_val = df_h1.loc[val_mask, features]
    y_val = df_h1.loc[val_mask, "avg_power_kw"].values
    X_test = df_h1.loc[test_mask, features]
    y_test = df_h1.loc[test_mask, "avg_power_kw"].values

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    candidates = [
        {"name": "LGBM_H1_L1_lr05_nl31", "objective": "regression_l1", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"name": "LGBM_H1_L1_lr03_nl31", "objective": "regression_l1", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8},
        {"name": "LGBM_H1_L2_lr05_nl31", "objective": "regression", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20, "colsample_bytree": 0.8},
    ]

    tuning_records = []
    trained_models = {}

    for cand in candidates:
        params = {
            "objective": cand["objective"],
            "learning_rate": cand["learning_rate"],
            "num_leaves": cand["num_leaves"],
            "min_child_samples": cand["min_child_samples"],
            "colsample_bytree": cand["colsample_bytree"],
            "random_state": 42,
            "verbosity": -1,
            "n_jobs": 4,
        }

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=600,
            valid_sets=[dtrain, dval],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        val_preds = np.clip(model.predict(X_val), 0, None)
        m_val = calculate_metrics(y_val, val_preds)

        print(
            f"  {cand['name']:<24} | Best Iter: {model.best_iteration:>3} | "
            f"Val H=1 MAE: {m_val['mae_kw']:>6.2f} kW | RMSE: {m_val['rmse_kw']:>6.2f} kW | "
            f"sMAPE: {m_val['smape_pct']:>5.2f}%"
        )

        tuning_records.append({
            "candidate_name": cand["name"],
            "task": "Rolling 1-Step (H=1)",
            "best_iteration": model.best_iteration,
            "val_mae_kw": m_val["mae_kw"],
            "val_rmse_kw": m_val["rmse_kw"],
            "val_mape_pct": m_val["mape_filtered_pct"],
            "val_smape_pct": m_val["smape_pct"],
            **cand,
        })
        trained_models[cand["name"]] = (model, cand)

    df_tuning = pd.DataFrame(tuning_records)
    best_row = df_tuning.sort_values("val_mae_kw").iloc[0]
    best_name = best_row["candidate_name"]
    best_model, best_config = trained_models[best_name]

    print(f"\n[+] Selected Best LightGBM (H=1) on Validation: {best_name}")
    print(f"    Validation MAE: {best_row['val_mae_kw']:.2f} kW | RMSE: {best_row['val_rmse_kw']:.2f} kW")

    # Predict on test set
    test_preds = np.clip(best_model.predict(X_test), 0, None)
    df_test_pred = df_h1.loc[test_mask, ["timestamp", "avg_power_kw"]].copy()
    df_test_pred["pred_lightgbm"] = test_preds

    return best_model, best_config, df_tuning, df_test_pred


def compute_test_metrics_and_seasonal(
    df_pred: pd.DataFrame,
    task_name: str,
    config_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Computes overall test set metrics and seasonal breakdowns."""
    y_true = df_pred["avg_power_kw"].values
    y_pred = df_pred["pred_lightgbm"].values
    m_all = calculate_metrics(y_true, y_pred)
    m_all["model"] = "LightGBM GBDT"
    m_all["task"] = task_name
    m_all["split"] = "Test"
    m_all["config_name"] = config_name

    seasonal_records = []
    for period_name, start_ts, end_ts in get_test_subperiods():
        mask = (df_pred["timestamp"] >= start_ts) & (df_pred["timestamp"] <= end_ts)
        sub = df_pred[mask]
        if len(sub) > 0:
            m_sub = calculate_metrics(sub["avg_power_kw"].values, sub["pred_lightgbm"].values)
            m_sub["model"] = "LightGBM GBDT"
            m_sub["task"] = task_name
            m_sub["period"] = period_name
            seasonal_records.append(m_sub)

    return m_all, seasonal_records


def generate_plots(
    df_h24: pd.DataFrame,
    df_h1: pd.DataFrame,
    model_h24: lgb.Booster,
    features_h24: List[str],
    figures_dir: Path,
):
    """Generates forecast comparison and feature importance visualizations."""
    sns.set_theme(style="whitegrid")

    # 1. Forecast sample plot (March 1-14, 2020)
    sample_start = "2020-03-01 00:00:00+00:00"
    sample_end = "2020-03-14 23:00:00+00:00"

    sub_h24 = df_h24[(df_h24["timestamp"] >= sample_start) & (df_h24["timestamp"] <= sample_end)]
    sub_h1 = df_h1[(df_h1["timestamp"] >= sample_start) & (df_h1["timestamp"] <= sample_end)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(sub_h24["timestamp"], sub_h24["avg_power_kw"], label="Actual Demand (kW)", color="black", linewidth=2.0)
    ax1.plot(sub_h24["timestamp"], sub_h24["pred_lightgbm"], label="LightGBM Day-Ahead (H=24)", color="#2a9d8f", linestyle="--", linewidth=1.6)
    ax1.set_title("LightGBM Day-Ahead (H=24) Forecast vs. Actual Demand (March 1–14, 2020)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Power Demand (kW)")
    ax1.legend(loc="upper right")

    ax2.plot(sub_h1["timestamp"], sub_h1["avg_power_kw"], label="Actual Demand (kW)", color="black", linewidth=2.0)
    ax2.plot(sub_h1["timestamp"], sub_h1["pred_lightgbm"], label="LightGBM Rolling 1-Step (H=1)", color="#e76f51", linestyle="-.", linewidth=1.6)
    ax2.set_title("LightGBM Rolling 1-Step (H=1) Forecast vs. Actual Demand (March 1–14, 2020)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Power Demand (kW)")
    ax2.set_xlabel("Timestamp (UTC)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig_path = figures_dir / "lightgbm_forecasts.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[+] Saved forecast plot to: {fig_path}")

    # 2. Feature Importance Plot (H=24 model)
    importance_split = model_h24.feature_importance(importance_type="split")
    importance_gain = model_h24.feature_importance(importance_type="gain")
    df_imp = pd.DataFrame({
        "feature": features_h24,
        "importance_split": importance_split,
        "importance_gain": importance_gain,
    }).sort_values("importance_gain", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_imp.head(15), x="importance_gain", y="feature", palette="viridis")
    plt.title("LightGBM Feature Importance (Gain) - Day-Ahead (H=24) Forecasting", fontsize=12, fontweight="bold")
    plt.xlabel("Total Gain")
    plt.ylabel("Feature")
    plt.tight_layout()
    imp_fig_path = figures_dir / "lightgbm_feature_importance.png"
    plt.savefig(imp_fig_path, dpi=200)
    plt.close()
    print(f"[+] Saved feature importance plot to: {imp_fig_path}")

    return df_imp


def main():
    print("=========================================================================")
    print("  Phase 3 - Model 3: LightGBM GBDT Forecasting Pipeline")
    print("=========================================================================")

    reports_dir = get_phase3_reports_dir()
    figures_dir = reports_dir / "figures"
    df = load_v2_dataset()

    # 1. Day-Ahead H=24 Task
    df_h24, features_h24 = build_h24_dataset(df)
    model_h24, cfg_h24, tuning_h24, pred_h24 = tune_and_train_lightgbm_h24(df_h24, features_h24, reports_dir)

    # 2. Rolling 1-Step H=1 Task
    df_h1, features_h1 = build_h1_dataset(df)
    model_h1, cfg_h1, tuning_h1, pred_h1 = tune_and_train_lightgbm_h1(df_h1, features_h1)

    # Combine tuning records
    df_tuning_all = pd.concat([tuning_h24, tuning_h1], ignore_index=True)
    df_tuning_all.to_csv(reports_dir / "lightgbm_validation_tuning.csv", index=False)

    # Compute Test Metrics
    m24_test, s24_test = compute_test_metrics_and_seasonal(pred_h24, "Day-Ahead (H=24)", cfg_h24["name"])
    m1_test, s1_test = compute_test_metrics_and_seasonal(pred_h1, "Rolling 1-Step (H=1)", cfg_h1["name"])

    df_metrics = pd.DataFrame([m24_test, m1_test])
    df_seasonal = pd.DataFrame(s24_test + s1_test)

    # Save Metrics & Predictions
    metrics_path = reports_dir / "lightgbm_metrics.csv"
    seasonal_path = reports_dir / "lightgbm_seasonal_breakdown.csv"
    config_path = reports_dir / "lightgbm_config.json"

    df_metrics.to_csv(metrics_path, index=False)
    df_seasonal.to_csv(seasonal_path, index=False)

    pred_h24.to_parquet(reports_dir / "lightgbm_h24_predictions.parquet", index=False)
    pred_h1.to_parquet(reports_dir / "lightgbm_h1_predictions.parquet", index=False)

    # Configuration summary
    config_info = {
        "model_name": "LightGBM Gradient Boosted Decision Trees",
        "h24_best_configuration": cfg_h24,
        "h1_best_configuration": cfg_h1,
        "h24_features": features_h24,
        "h1_features": features_h1,
        "target_variable": "avg_power_kw (reconstructed hourly energy allocation; not instantaneous physical power)",
        "energy_cap_note": "7.7 kW per-session provisional cap applied during v2 preprocessing",
        "train_range": "2018-05-01 00:00 to 2019-12-31 23:00 UTC",
        "validation_range": "2020-01-01 00:00 to 2020-02-29 23:00 UTC",
        "test_range": "2020-03-01 00:00 to 2020-12-05 03:00 UTC",
    }
    with open(config_path, "w") as f:
        json.dump(config_info, f, indent=2)

    # Generate Plots & Feature Importance
    df_imp = generate_plots(pred_h24, pred_h1, model_h24, features_h24, figures_dir)
    df_imp.to_csv(reports_dir / "lightgbm_feature_importance.csv", index=False)

    print("\n--- Quantitative Test Performance: LightGBM ---")
    print(df_metrics.to_string(index=False))

    print("\n--- Test Set Seasonal Breakdown ---")
    print(df_seasonal.to_string(index=False))

    print(f"\n[+] Saved LightGBM metrics to: {metrics_path}")
    print(f"[+] Saved LightGBM seasonal breakdown to: {seasonal_path}")
    print(f"[+] Saved LightGBM configuration to: {config_path}")


if __name__ == "__main__":
    main()

