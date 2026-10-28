"""
Phase 3: Corrected Dataset (v2) Baselines Evaluation
====================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models

This script evaluates classical benchmark baselines on the corrected v2 dataset
(`data/processed_v2/caltech_hourly_demand_v2.parquet`).

Key Documentation:
- The target series 'avg_power_kw' represents hourly facility energy consumption (kW = kWh/h)
  reconstructed via uniform energy allocation across connection intervals.
  It is NOT measured instantaneous charging power.
- The 7.7 kW per-session cap applied during v2 preprocessing remains provisional (based on Level 2
  AC rating assumptions, not verified hardware spec sheets).
- Baselines evaluated:
  * 1-Hour Ahead (Rolling H=1): Persistence (t-1), Seasonal Naive (t-24), Seasonal Naive (t-168)
  * Day-Ahead 24-Hour (Rolling Daily Origin H=24): Persistence (Flat), Seasonal Naive 24h (Diurnal), Seasonal Naive 168h (Weekly)
- Results are saved to:
  * `reports/phase3/baseline_v2_metrics.csv`
  * `reports/phase3/baseline_v2_seasonal_breakdown.csv`
  * `reports/phase3/baseline_v2_predictions.parquet`
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from phase3_eval_utils import (
    get_project_root,
    get_phase3_reports_dir,
    load_v2_dataset,
    calculate_metrics,
    get_split_masks,
    get_daily_forecast_origins,
    get_test_subperiods,
)


def evaluate_h1_baselines(df: pd.DataFrame, reports_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes 1-step-ahead (H=1) baseline forecasts:
    - Persistence (t-1)
    - Seasonal Naive 24h (t-24)
    - Seasonal Naive 168h (t-168)
    Evaluates on both Validation and Test splits, plus seasonal sub-periods.
    """
    df = df.copy()
    df["pred_persistence_h1"] = df["avg_power_kw"].shift(1)
    df["pred_snaive24_h1"] = df["avg_power_kw"].shift(24)
    df["pred_snaive168_h1"] = df["avg_power_kw"].shift(168)

    train_mask, val_mask, test_mask = get_split_masks(df)

    models = [
        ("Persistence (t-1)", "pred_persistence_h1"),
        ("Seasonal Naive (t-24)", "pred_snaive24_h1"),
        ("Seasonal Naive (t-168)", "pred_snaive168_h1"),
    ]

    metrics_records = []
    seasonal_records = []

    for split_name, mask in [("Validation", val_mask), ("Test", test_mask)]:
        sub_df = df[mask].copy()
        y_true = sub_df["avg_power_kw"].values
        for model_name, col in models:
            y_pred = sub_df[col].values
            m = calculate_metrics(y_true, y_pred)
            metrics_records.append({
                "model": model_name,
                "task": "Rolling 1-Step (H=1)",
                "split": split_name,
                "n_samples": len(y_true),
                "mae_kw": m["mae_kw"],
                "rmse_kw": m["rmse_kw"],
                "mape_filtered_pct": m["mape_filtered_pct"],
                "smape_pct": m["smape_pct"],
                "mape_evaluated_samples": m["mape_evaluated_samples"],
            })

    # Test Seasonal breakdown for H=1
    test_df = df[test_mask].copy()
    for period_name, start_ts, end_ts in get_test_subperiods():
        period_df = test_df[(test_df["timestamp"] >= start_ts) & (test_df["timestamp"] <= end_ts)]
        y_true_p = period_df["avg_power_kw"].values
        for model_name, col in models:
            y_pred_p = period_df[col].values
            m = calculate_metrics(y_true_p, y_pred_p)
            seasonal_records.append({
                "model": model_name,
                "task": "Rolling 1-Step (H=1)",
                "period": period_name,
                "n_samples": len(y_true_p),
                "mae_kw": m["mae_kw"],
                "rmse_kw": m["rmse_kw"],
                "mape_filtered_pct": m["mape_filtered_pct"],
                "smape_pct": m["smape_pct"],
                "mape_evaluated_samples": m["mape_evaluated_samples"],
            })

    pred_h1 = test_df[["timestamp", "avg_power_kw", "pred_persistence_h1", "pred_snaive24_h1", "pred_snaive168_h1"]].copy()
    return pd.DataFrame(metrics_records), pd.DataFrame(seasonal_records), pred_h1


def evaluate_h24_baselines(df: pd.DataFrame, reports_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes 24-Hour Day-Ahead (H=24) forecasts using daily rolling origins at 00:00 UTC.
    Evaluates on both Validation (Jan-Feb 2020) and Test (Mar-Dec 2020) sets.
    """
    total_len = len(df)
    train_mask, val_mask, test_mask = get_split_masks(df)

    metrics_records = []
    seasonal_records = []

    h24_test_dfs = []

    for split_name in ["val", "test"]:
        origins = get_daily_forecast_origins(df, split=split_name)
        display_name = "Validation" if split_name == "val" else "Test"

        rows = []
        for t in origins:
            t_ts = df.loc[t, "timestamp"]
            y_true_window = df.loc[t : t + 23, "avg_power_kw"].values
            ts_window = df.loc[t : t + 23, "timestamp"].tolist()

            # Causal forecasts using only info up to t-1:
            # 1. Flat Persistence: last observed value at origin t-1
            pred_persist = np.full(24, df.loc[t - 1, "avg_power_kw"])
            # 2. Seasonal Naive 24h: diurnal cycle of previous day [t-24 : t-1]
            pred_s24 = df.loc[t - 24 : t - 1, "avg_power_kw"].values
            # 3. Seasonal Naive 168h: same 24 hours from 1 week prior [t-168 : t-145]
            pred_s168 = df.loc[t - 168 : t - 145, "avg_power_kw"].values

            for step in range(24):
                rows.append({
                    "origin_timestamp": t_ts,
                    "timestamp": ts_window[step],
                    "step": step + 1,
                    "avg_power_kw": y_true_window[step],
                    "pred_persistence_h24": pred_persist[step],
                    "pred_snaive24_h24": pred_s24[step],
                    "pred_snaive168_h24": pred_s168[step],
                })

        split_h24_df = pd.DataFrame(rows)
        split_h24_df["timestamp"] = pd.to_datetime(split_h24_df["timestamp"], utc=True)
        split_h24_df["origin_timestamp"] = pd.to_datetime(split_h24_df["origin_timestamp"], utc=True)
        if split_name == "test":
            h24_test_dfs.append(split_h24_df)

        y_true = split_h24_df["avg_power_kw"].values
        models = [
            ("Persistence (Flat H=24)", "pred_persistence_h24"),
            ("Seasonal Naive (Diurnal H=24)", "pred_snaive24_h24"),
            ("Seasonal Naive (Weekly H=24)", "pred_snaive168_h24"),
        ]

        for model_name, col in models:
            y_pred = split_h24_df[col].values
            m = calculate_metrics(y_true, y_pred)
            metrics_records.append({
                "model": model_name,
                "task": "Day-Ahead (H=24)",
                "split": display_name,
                "n_samples": len(y_true),
                "mae_kw": m["mae_kw"],
                "rmse_kw": m["rmse_kw"],
                "mape_filtered_pct": m["mape_filtered_pct"],
                "smape_pct": m["smape_pct"],
                "mape_evaluated_samples": m["mape_evaluated_samples"],
            })

        # If test split, calculate seasonal breakdown
        if split_name == "test":
            for period_name, start_ts, end_ts in get_test_subperiods():
                p_mask = (split_h24_df["timestamp"] >= start_ts) & (split_h24_df["timestamp"] <= end_ts)
                p_df = split_h24_df[p_mask]
                y_true_p = p_df["avg_power_kw"].values
                for model_name, col in models:
                    y_pred_p = p_df[col].values
                    m = calculate_metrics(y_true_p, y_pred_p)
                    seasonal_records.append({
                        "model": model_name,
                        "task": "Day-Ahead (H=24)",
                        "period": period_name,
                        "n_samples": len(y_true_p),
                        "mae_kw": m["mae_kw"],
                        "rmse_kw": m["rmse_kw"],
                        "mape_filtered_pct": m["mape_filtered_pct"],
                        "smape_pct": m["smape_pct"],
                        "mape_evaluated_samples": m["mape_evaluated_samples"],
                    })

    test_pred_df = h24_test_dfs[0] if h24_test_dfs else pd.DataFrame()
    return pd.DataFrame(metrics_records), pd.DataFrame(seasonal_records), test_pred_df


def main():
    print("=========================================================================")
    print("  Phase 3: Benchmarking Corrected Dataset (v2) Baselines")
    print("=========================================================================")
    reports_dir = get_phase3_reports_dir()
    df = load_v2_dataset()
    print(f"[+] Loaded v2 dataset: {len(df):,} hours from {df['timestamp'].min()} to {df['timestamp'].max()}")

    # 1. Evaluate H=1 baselines
    m_h1, s_h1, p_h1 = evaluate_h1_baselines(df, reports_dir)

    # 2. Evaluate H=24 Day-Ahead baselines
    m_h24, s_h24, p_h24 = evaluate_h24_baselines(df, reports_dir)

    # Combine metrics and seasonal breakdowns
    all_metrics = pd.concat([m_h24, m_h1], ignore_index=True)
    all_seasonal = pd.concat([s_h24, s_h1], ignore_index=True)

    metrics_csv = reports_dir / "baseline_v2_metrics.csv"
    seasonal_csv = reports_dir / "baseline_v2_seasonal_breakdown.csv"
    all_metrics.to_csv(metrics_csv, index=False)
    all_seasonal.to_csv(seasonal_csv, index=False)

    # Save predictions
    p_h24.to_parquet(reports_dir / "baseline_v2_h24_predictions.parquet", index=False)
    p_h1.to_parquet(reports_dir / "baseline_v2_h1_predictions.parquet", index=False)

    print("\n--- Summary of v2 Baseline Performance (Day-Ahead H=24 Primary Task) ---")
    print(all_metrics[all_metrics["task"] == "Day-Ahead (H=24)"].to_string(index=False))

    print("\n--- Summary of v2 Baseline Performance (Rolling 1-Step H=1) ---")
    print(all_metrics[all_metrics["task"] == "Rolling 1-Step (H=1)"].to_string(index=False))

    print(f"\n[+] Saved baseline metrics to: {metrics_csv}")
    print(f"[+] Saved baseline seasonal breakdown to: {seasonal_csv}")


if __name__ == "__main__":
    main()
