"""
Phase 3 - Model 1: Holt-Winters Exponential Smoothing Pipeline
==============================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models

METHODOLOGY AND CAVEAT DOCUMENTATION:
- Target Series: 'avg_power_kw' represents hourly facility energy consumption (kWh/h = kW)
  reconstructed via uniform energy allocation across connection intervals.
  It is NOT measured instantaneous charging power. Intra-session dynamics (e.g., CC/CV taper)
  are smoothed out by definition.
- Provisional Cap: The 7.7 kW per-session power cap applied during v2 preprocessing remains
  provisional, based on standard Level 2 EVSE maximum delivery (32A @ 240V). While it successfully
  suppressed unphysical meter-carryover spikes (reducing the all-time peak from 375.4 kW to 120.6 kW),
  station-level hardware electrical ratings were not verified against primary engineering spec sheets.
- Chronological Partition:
  * Training:   2018-05-01 00:00 to 2019-12-31 23:00 UTC (14,640 hours)
  * Validation: 2020-01-01 00:00 to 2020-02-29 23:00 UTC (1,440 hours)
  * Test:       2020-03-01 00:00 to 2020-12-05 03:00 UTC (6,700 hours)
- Hyperparameter Tuning:
  * Model configurations are selected strictly on the Validation set (Jan–Feb 2020).
  * No test set information is used in selecting trend, damping, or smoothing specifications.
- Tasks Evaluated:
  * Primary: 24-hour Day-Ahead Forecasting (H=24) using daily rolling origin at 00:00 UTC.
  * Secondary: Rolling 1-Step-Ahead Forecasting (H=1) using previous hour feedback.
- Post-processing:
  * Non-negative clipping (demand >= 0 kW) is enforced physically.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from phase3_eval_utils import (
    get_project_root,
    get_phase3_reports_dir,
    load_v2_dataset,
    calculate_metrics,
    get_split_masks,
    get_daily_forecast_origins,
    get_test_subperiods,
)


def tune_holt_winters_on_validation(
    df: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    val_origins: List[int],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluates candidate Holt-Winters specifications on the Validation Set (Jan–Feb 2020).
    Candidates:
      1. No Trend + Additive Diurnal Seasonality (period=24)
      2. Additive Trend + Additive Diurnal Seasonality (period=24)
      3. Damped Additive Trend + Additive Diurnal Seasonality (period=24)
    Returns:
      best_config dict, tuning_results DataFrame
    """
    print("\n--- Tuning Holt-Winters Candidates on Validation Data (Jan-Feb 2020) ---")
    train_series = df.loc[train_mask, "avg_power_kw"].values

    candidates = [
        {
            "name": "HW_NoTrend_AddSeason24",
            "trend": None,
            "seasonal": "add",
            "damped_trend": False,
            "seasonal_periods": 24,
        },
        {
            "name": "HW_AddTrend_AddSeason24",
            "trend": "add",
            "seasonal": "add",
            "damped_trend": False,
            "seasonal_periods": 24,
        },
        {
            "name": "HW_DampedTrend_AddSeason24",
            "trend": "add",
            "seasonal": "add",
            "damped_trend": True,
            "seasonal_periods": 24,
        },
    ]

    tuning_records = []

    for cand in candidates:
        print(f"[*] Fitting {cand['name']} on training split (14,640 hrs)...")
        init_model = ExponentialSmoothing(
            train_series,
            seasonal_periods=cand["seasonal_periods"],
            trend=cand["trend"],
            seasonal=cand["seasonal"],
            damped_trend=cand["damped_trend"],
            initialization_method="estimated",
        )
        fitted = init_model.fit()
        params = fitted.params

        # 1. Day-Ahead H=24 on validation
        preds_h24 = []
        actuals_h24 = []

        for t in val_origins:
            # strictly causal: history up to t-1
            hist_y = df.loc[: t - 1, "avg_power_kw"].values
            fit_kwargs = {
                "smoothing_level": params["smoothing_level"],
                "smoothing_seasonal": params["smoothing_seasonal"],
                "optimized": False,
            }
            if cand["trend"] == "add":
                fit_kwargs["smoothing_trend"] = params["smoothing_trend"]
                if cand["damped_trend"]:
                    fit_kwargs["damping_trend"] = params["damping_trend"]

            m_origin = ExponentialSmoothing(
                hist_y,
                seasonal_periods=cand["seasonal_periods"],
                trend=cand["trend"],
                seasonal=cand["seasonal"],
                damped_trend=cand["damped_trend"],
            ).fit(**fit_kwargs)

            fc = np.clip(m_origin.forecast(24), 0, None)
            preds_h24.extend(fc)
            actuals_h24.extend(df.loc[t : t + 23, "avg_power_kw"].values)

        m24 = calculate_metrics(np.array(actuals_h24), np.array(preds_h24))

        # 2. Rolling 1-Step H=1 on validation
        # Filter whole validation sequence causally
        val_start_idx = df[val_mask].index[0]
        val_end_idx = df[val_mask].index[-1]
        hist_val = df.loc[:val_end_idx, "avg_power_kw"].values
        m_full_val = ExponentialSmoothing(
            hist_val,
            seasonal_periods=cand["seasonal_periods"],
            trend=cand["trend"],
            seasonal=cand["seasonal"],
            damped_trend=cand["damped_trend"],
        ).fit(**fit_kwargs)

        val_preds_h1 = np.clip(m_full_val.fittedvalues[val_start_idx : val_end_idx + 1], 0, None)
        val_actuals_h1 = df.loc[val_mask, "avg_power_kw"].values
        m1 = calculate_metrics(val_actuals_h1, val_preds_h1)

        print(
            f"    -> Val H=24: MAE = {m24['mae_kw']:.2f} kW | RMSE = {m24['rmse_kw']:.2f} kW | "
            f"sMAPE = {m24['smape_pct']:.2f}%"
        )
        print(
            f"    -> Val H=1:  MAE = {m1['mae_kw']:.2f} kW | RMSE = {m1['rmse_kw']:.2f} kW | "
            f"sMAPE = {m1['smape_pct']:.2f}%"
        )

        record = {
            "candidate_name": cand["name"],
            "trend": str(cand["trend"]),
            "damped_trend": cand["damped_trend"],
            "seasonal": cand["seasonal"],
            "seasonal_periods": cand["seasonal_periods"],
            "val_h24_mae_kw": m24["mae_kw"],
            "val_h24_rmse_kw": m24["rmse_kw"],
            "val_h24_mape_pct": m24["mape_filtered_pct"],
            "val_h24_smape_pct": m24["smape_pct"],
            "val_h1_mae_kw": m1["mae_kw"],
            "val_h1_rmse_kw": m1["rmse_kw"],
            "val_h1_mape_pct": m1["mape_filtered_pct"],
            "val_h1_smape_pct": m1["smape_pct"],
            "aic": round(float(fitted.aic), 2),
            "bic": round(float(fitted.bic), 2),
        }
        tuning_records.append(record)

    df_tuning = pd.DataFrame(tuning_records)
    # Select candidate with lowest Validation Day-Ahead (H=24) MAE (primary task)
    best_row = df_tuning.sort_values("val_h24_mae_kw").iloc[0]
    best_cand_name = best_row["candidate_name"]
    best_config = [c for c in candidates if c["name"] == best_cand_name][0]

    print(f"\n[+] Selected Best Model on Validation Data: {best_cand_name}")
    print(f"    Validation H=24 MAE: {best_row['val_h24_mae_kw']:.2f} kW (RMSE: {best_row['val_h24_rmse_kw']:.2f} kW)")

    return best_config, df_tuning


def evaluate_holt_winters_test(
    df: pd.DataFrame,
    best_config: Dict[str, Any],
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
    reports_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates the selected Holt-Winters model on the Test set:
    - Day-Ahead (H=24) rolling daily origins (6,696 hours across 279 days)
    - Rolling 1-Step (H=1) (6,700 hours)
    - Seasonal breakdowns across March–May, June–August, September–December 2020.
    """
    print("\n--- Evaluating Selected Holt-Winters Model on Test Data (Mar-Dec 2020) ---")
    train_series = df.loc[train_mask, "avg_power_kw"].values

    fitted_train = ExponentialSmoothing(
        train_series,
        seasonal_periods=best_config["seasonal_periods"],
        trend=best_config["trend"],
        seasonal=best_config["seasonal"],
        damped_trend=best_config["damped_trend"],
        initialization_method="estimated",
    ).fit()
    params = fitted_train.params

    fit_kwargs = {
        "smoothing_level": params["smoothing_level"],
        "smoothing_seasonal": params["smoothing_seasonal"],
        "optimized": False,
    }
    if best_config["trend"] == "add":
        fit_kwargs["smoothing_trend"] = params["smoothing_trend"]
        if best_config["damped_trend"]:
            fit_kwargs["damping_trend"] = params["damping_trend"]

    # 1. Day-Ahead H=24 on Test Set
    test_origins = get_daily_forecast_origins(df, split="test")
    h24_rows = []

    print(f"[*] Running Day-Ahead H=24 rolling forecasts across {len(test_origins)} origins...")
    for t in test_origins:
        origin_ts = df.loc[t, "timestamp"]
        ts_window = df.loc[t : t + 23, "timestamp"].tolist()
        y_true_window = df.loc[t : t + 23, "avg_power_kw"].values

        # strictly causal history up to t-1
        hist_y = df.loc[: t - 1, "avg_power_kw"].values
        m_origin = ExponentialSmoothing(
            hist_y,
            seasonal_periods=best_config["seasonal_periods"],
            trend=best_config["trend"],
            seasonal=best_config["seasonal"],
            damped_trend=best_config["damped_trend"],
        ).fit(**fit_kwargs)

        fc = np.clip(m_origin.forecast(24), 0, None)

        for step in range(24):
            h24_rows.append({
                "origin_timestamp": origin_ts,
                "timestamp": ts_window[step],
                "step": step + 1,
                "avg_power_kw": y_true_window[step],
                "pred_holt_winters": fc[step],
            })

    df_h24_pred = pd.DataFrame(h24_rows)
    df_h24_pred["timestamp"] = pd.to_datetime(df_h24_pred["timestamp"], utc=True)
    df_h24_pred["origin_timestamp"] = pd.to_datetime(df_h24_pred["origin_timestamp"], utc=True)

    # 2. Rolling 1-Step H=1 on Test Set
    test_start_idx = df[test_mask].index[0]
    test_end_idx = df[test_mask].index[-1]
    hist_test = df.loc[:test_end_idx, "avg_power_kw"].values

    m_full_test = ExponentialSmoothing(
        hist_test,
        seasonal_periods=best_config["seasonal_periods"],
        trend=best_config["trend"],
        seasonal=best_config["seasonal"],
        damped_trend=best_config["damped_trend"],
    ).fit(**fit_kwargs)

    test_preds_h1 = np.clip(m_full_test.fittedvalues[test_start_idx : test_end_idx + 1], 0, None)
    test_actuals_h1 = df.loc[test_mask, "avg_power_kw"].values
    test_timestamps_h1 = df.loc[test_mask, "timestamp"].values

    df_h1_pred = pd.DataFrame({
        "timestamp": pd.to_datetime(test_timestamps_h1, utc=True),
        "avg_power_kw": test_actuals_h1,
        "pred_holt_winters": test_preds_h1,
    })

    # Metrics Summary
    metrics_records = []
    seasonal_records = []

    # Overall Test Metrics
    m_h24 = calculate_metrics(df_h24_pred["avg_power_kw"].values, df_h24_pred["pred_holt_winters"].values)
    m_h1 = calculate_metrics(df_h1_pred["avg_power_kw"].values, df_h1_pred["pred_holt_winters"].values)

    metrics_records.append({
        "model": "Holt-Winters Exponential Smoothing",
        "task": "Day-Ahead (H=24)",
        "split": "Test",
        "n_samples": len(df_h24_pred),
        "mae_kw": m_h24["mae_kw"],
        "rmse_kw": m_h24["rmse_kw"],
        "mape_filtered_pct": m_h24["mape_filtered_pct"],
        "smape_pct": m_h24["smape_pct"],
        "mape_evaluated_samples": m_h24["mape_evaluated_samples"],
        "config_name": best_config["name"],
    })

    metrics_records.append({
        "model": "Holt-Winters Exponential Smoothing",
        "task": "Rolling 1-Step (H=1)",
        "split": "Test",
        "n_samples": len(df_h1_pred),
        "mae_kw": m_h1["mae_kw"],
        "rmse_kw": m_h1["rmse_kw"],
        "mape_filtered_pct": m_h1["mape_filtered_pct"],
        "smape_pct": m_h1["smape_pct"],
        "mape_evaluated_samples": m_h1["mape_evaluated_samples"],
        "config_name": best_config["name"],
    })

    # Seasonal Breakdowns
    for task_name, pred_df in [("Day-Ahead (H=24)", df_h24_pred), ("Rolling 1-Step (H=1)", df_h1_pred)]:
        for period_name, start_ts, end_ts in get_test_subperiods():
            p_mask = (pred_df["timestamp"] >= start_ts) & (pred_df["timestamp"] <= end_ts)
            sub = pred_df[p_mask]
            if len(sub) > 0:
                m_sub = calculate_metrics(sub["avg_power_kw"].values, sub["pred_holt_winters"].values)
                seasonal_records.append({
                    "model": "Holt-Winters Exponential Smoothing",
                    "task": task_name,
                    "period": period_name,
                    "n_samples": len(sub),
                    "mae_kw": m_sub["mae_kw"],
                    "rmse_kw": m_sub["rmse_kw"],
                    "mape_filtered_pct": m_sub["mape_filtered_pct"],
                    "smape_pct": m_sub["smape_pct"],
                    "mape_evaluated_samples": m_sub["mape_evaluated_samples"],
                })

    df_metrics = pd.DataFrame(metrics_records)
    df_seasonal = pd.DataFrame(seasonal_records)

    config_info = {
        "model_name": "Holt-Winters Exponential Smoothing",
        "selected_configuration": best_config,
        "fitted_parameters": {k: float(v) for k, v in params.items() if isinstance(v, (int, float, np.floating))},
        "target_variable": "avg_power_kw (reconstructed hourly energy allocation; not instantaneous physical power)",
        "energy_cap_note": "7.7 kW per-session provisional cap applied during v2 preprocessing",
        "train_range": "2018-05-01 00:00 to 2019-12-31 23:00 UTC",
        "validation_range": "2020-01-01 00:00 to 2020-02-29 23:00 UTC",
        "test_range": "2020-03-01 00:00 to 2020-12-05 03:00 UTC",
    }

    return df_metrics, df_seasonal, df_h24_pred, df_h1_pred, config_info


def generate_plots(df_h24: pd.DataFrame, df_h1: pd.DataFrame, figures_dir: Path):
    """
    Generates comparison visual of Holt-Winters forecasts vs Actuals on the test set.
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # 1. 14-day sample in March 2020 (Shock Onset)
    sample_start = "2020-03-01 00:00:00+00:00"
    sample_end = "2020-03-14 23:00:00+00:00"

    sub_h24 = df_h24[(df_h24["timestamp"] >= sample_start) & (df_h24["timestamp"] <= sample_end)]
    sub_h1 = df_h1[(df_h1["timestamp"] >= sample_start) & (df_h1["timestamp"] <= sample_end)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(sub_h24["timestamp"], sub_h24["avg_power_kw"], label="Actual Demand (kW)", color="black", linewidth=2.0)
    ax1.plot(sub_h24["timestamp"], sub_h24["pred_holt_winters"], label="HW Day-Ahead (H=24)", color="#2b5c8f", linestyle="--", linewidth=1.6)
    ax1.set_title("Holt-Winters Day-Ahead (H=24) Forecast vs. Actual Demand (March 1–14, 2020)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Power Demand (kW)")
    ax1.legend(loc="upper right")

    ax2.plot(sub_h1["timestamp"], sub_h1["avg_power_kw"], label="Actual Demand (kW)", color="black", linewidth=2.0)
    ax2.plot(sub_h1["timestamp"], sub_h1["pred_holt_winters"], label="HW Rolling 1-Step (H=1)", color="#e07a5f", linestyle="-.", linewidth=1.6)
    ax2.set_title("Holt-Winters Rolling 1-Step (H=1) Forecast vs. Actual Demand (March 1–14, 2020)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Power Demand (kW)")
    ax2.set_xlabel("Timestamp (UTC)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig_path = figures_dir / "holt_winters_forecasts.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[+] Saved forecast plot to: {fig_path}")


def main():
    print("=========================================================================")
    print("  Phase 3 - Model 1: Holt-Winters Exponential Smoothing Pipeline")
    print("=========================================================================")

    reports_dir = get_phase3_reports_dir()
    figures_dir = reports_dir / "figures"
    df = load_v2_dataset()

    train_mask, val_mask, test_mask = get_split_masks(df)
    val_origins = get_daily_forecast_origins(df, split="val")

    # Step 1: Tune on validation set
    best_config, df_tuning = tune_holt_winters_on_validation(df, train_mask, val_mask, val_origins)
    df_tuning.to_csv(reports_dir / "holt_winters_validation_tuning.csv", index=False)

    # Step 2: Evaluate on test set
    df_metrics, df_seasonal, df_h24_pred, df_h1_pred, config_info = evaluate_holt_winters_test(
        df, best_config, train_mask, val_mask, test_mask, reports_dir
    )

    # Step 3: Save results
    metrics_path = reports_dir / "holt_winters_metrics.csv"
    seasonal_path = reports_dir / "holt_winters_seasonal_breakdown.csv"
    config_path = reports_dir / "holt_winters_config.json"

    df_metrics.to_csv(metrics_path, index=False)
    df_seasonal.to_csv(seasonal_path, index=False)
    with open(config_path, "w") as f:
        json.dump(config_info, f, indent=2)

    df_h24_pred.to_parquet(reports_dir / "holt_winters_h24_predictions.parquet", index=False)
    df_h1_pred.to_parquet(reports_dir / "holt_winters_h1_predictions.parquet", index=False)

    print("\n--- Quantitative Test Performance: Holt-Winters ---")
    print(df_metrics.to_string(index=False))

    print("\n--- Test Set Seasonal Breakdown ---")
    print(df_seasonal.to_string(index=False))

    # Step 4: Generate plots
    generate_plots(df_h24_pred, df_h1_pred, figures_dir)

    print(f"\n[+] Saved metrics to: {metrics_path}")
    print(f"[+] Saved seasonal breakdown to: {seasonal_path}")
    print(f"[+] Saved configuration to: {config_path}")


if __name__ == "__main__":
    main()

