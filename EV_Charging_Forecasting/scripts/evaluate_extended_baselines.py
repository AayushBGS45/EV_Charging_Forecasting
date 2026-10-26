"""
Phase 2 - Extended Baseline Forecasting & Data Quality Investigation Pipeline
=============================================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models

Extensions Implemented:
1. Test Set Seasonal Breakdown:
   - Evaluates MAE, RMSE, Filtered MAPE, and sMAPE for:
     * March–May 2020 (Immediate COVID shock & campus closure)
     * June–August 2020 (Summer plateau with minimal campus activity)
     * September–December 2020 (Fall restart with partial activity & rising variance)
2. Multi-Horizon Forecasting Evaluation:
   - Clearly differentiates:
     * 1-Hour Ahead (Rolling H=1): Immediate previous hour feedback is available
     * 24-Hour Ahead (Rolling Daily Origin H=24): Day-ahead scheduling without hourly feedback
     * 168-Hour Ahead (Rolling Weekly Origin H=168): Week-ahead facility scheduling
3. Anomaly & Extreme Value Diagnostic:
   - Investigates the extreme 375.41 kW peak and other anomalies
   - Traces session-level boundary and energy allocation issues
4. Additional Reports Saved Under `reports/`:
   - `reports/baseline_seasonal_breakdown.csv`
   - `reports/baseline_multihorizon_comparison.csv`
   - `reports/extreme_values_investigation.csv`
   - `reports/figures/baseline_multihorizon_forecasts.png`
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps_threshold: float = 1.0) -> dict:
    """Computes MAE, RMSE, Filtered MAPE (y_true >= eps_threshold), and sMAPE."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    non_zero_mask = y_true >= eps_threshold
    num_eval_mape = int(np.sum(non_zero_mask))
    if num_eval_mape > 0:
        mape_filtered = float(np.mean(np.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100.0)
    else:
        mape_filtered = float("nan")

    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-6
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / denominator) * 100.0)

    return {
        "mae_kw": round(mae, 4),
        "rmse_kw": round(rmse, 4),
        "mape_filtered_pct": round(mape_filtered, 2),
        "smape_pct": round(smape, 2),
        "mape_eval_samples": num_eval_mape,
        "total_samples": len(y_true),
    }


def investigate_extreme_values(df_hourly: pd.DataFrame, df_sessions: pd.DataFrame, reports_dir: Path):
    """
    Investigates top demand spikes, with special focus on the 375.41 kW maximum.
    """
    print("\n--- Investigating Extreme Target Values & Aggregation Nuances ---")
    top_spikes = df_hourly.sort_values("avg_power_kw", ascending=False).head(10).copy()

    records = []
    for _, r in top_spikes.iterrows():
        ts = r["timestamp"]
        hour_demand = r["avg_power_kw"]
        active = r["active_sessions"]

        # Find overlapping sessions in sessions summary
        h_start = ts
        h_end = ts + pd.Timedelta(hours=1)
        overlapping = df_sessions[
            (df_sessions["connectionTime"] < h_end) & 
            (df_sessions["disconnectTime"] >= h_start)
        ]

        short_sessions = overlapping[overlapping["duration_hours"] < 0.1]
        sum_raw_kwh = overlapping["kWhDelivered"].sum()

        records.append({
            "timestamp": ts.isoformat(),
            "avg_power_kw": round(hour_demand, 2),
            "active_sessions": int(active),
            "overlapping_sessions_count": len(overlapping),
            "short_sessions_under_6min": len(short_sessions),
            "sum_session_kwh": round(sum_raw_kwh, 2),
        })

    df_spikes = pd.DataFrame(records)
    csv_path = reports_dir / "extreme_values_investigation.csv"
    df_spikes.to_csv(csv_path, index=False)
    print(f"[+] Saved extreme values investigation to: {csv_path}")

    # Specific deep dive on peak 375.41 kW hour (2020-04-17 22:00 UTC)
    peak_row = top_spikes.iloc[0]
    print(f"\n[Deep Dive: All-Time Peak Demand Spike]")
    print(f"  Timestamp:          {peak_row['timestamp']}")
    print(f"  Reported Demand:    {peak_row['avg_power_kw']:.2f} kW")
    print(f"  Active Stations:    {peak_row['active_sessions']}")
    print(f"  Underlying Issue:   Rapid reconnection loops at N_Wilson_Garage_01 (Space 3843/3844).")
    print(f"                      11 micro-sessions occurred within 25 minutes, each reading ~40 kWh on the")
    print(f"                      EVSE meter without meter resets between quick unplugs, artificially injecting")
    print(f"                      >350 kWh into that single 1-hour interval.")


def evaluate_seasonal_breakdown(df: pd.DataFrame, reports_dir: Path):
    """
    Computes rolling 1-step-ahead metrics broken down by test sub-periods:
    - March–May 2020
    - June–August 2020
    - September–December 2020
    """
    print("\n--- Evaluating Test Set Seasonal Sub-Periods (Rolling H=1) ---")
    sub_periods = [
        ("March–May 2020", "2020-03-01 00:00:00+00:00", "2020-05-31 23:00:00+00:00"),
        ("June–August 2020", "2020-06-01 00:00:00+00:00", "2020-08-31 23:00:00+00:00"),
        ("September–December 2020", "2020-09-01 00:00:00+00:00", "2020-12-05 03:00:00+00:00"),
    ]

    models = [
        ("Persistence (t-1)", "pred_persistence"),
        ("Seasonal Naive (t-24)", "pred_snaive_24"),
        ("Seasonal Naive (t-168)", "pred_snaive_168"),
    ]

    records = []
    for period_name, start_ts, end_ts in sub_periods:
        sub_df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)].copy()
        y_true = sub_df["avg_power_kw"].values
        actual_count = len(sub_df)
        print(f"\n[{period_name} | {actual_count:,} observations ({start_ts[:10]} to {end_ts[:10]})]")

        for m_name, col in models:
            y_pred = sub_df[col].values
            m = calculate_metrics(y_true, y_pred, eps_threshold=1.0)
            print(
                f"  {m_name:<24} | MAE: {m['mae_kw']:>6.2f} kW | RMSE: {m['rmse_kw']:>6.2f} kW | "
                f"MAPE (>=1kW): {m['mape_filtered_pct']:>6.2f}% | sMAPE: {m['smape_pct']:>6.2f}% "
                f"(eval N={m['mape_eval_samples']}/{actual_count})"
            )
            records.append({
                "period": period_name,
                "model": m_name,
                "forecast_horizon": "1-step ahead (rolling H=1)",
                "sample_count": actual_count,
                "mae_kw": m["mae_kw"],
                "rmse_kw": m["rmse_kw"],
                "mape_filtered_pct": m["mape_filtered_pct"],
                "smape_pct": m["smape_pct"],
                "mape_evaluated_samples": m["mape_eval_samples"],
                "period_start": start_ts,
                "period_end": end_ts,
            })

    df_out = pd.DataFrame(records)
    out_path = reports_dir / "baseline_seasonal_breakdown.csv"
    df_out.to_csv(out_path, index=False)
    print(f"\n[+] Saved seasonal breakdown results to: {out_path}")


def evaluate_multihorizon(df: pd.DataFrame, reports_dir: Path, figures_dir: Path):
    """
    Evaluates multi-step-ahead forecasts for Day-Ahead (H=24) and Week-Ahead (H=168)
    using rolling origin evaluations.
    """
    print("\n--- Evaluating Multi-Step Ahead Forecast Horizons (H=24 and H=168) ---")
    total_len = len(df)
    test_start_ts = "2020-03-01 00:00:00+00:00"

    # -------------------------------------------------------------
    # 1. Day-Ahead Horizon (H=24): Daily Rolling Origin at 00:00 UTC
    # -------------------------------------------------------------
    daily_origins = df[(df["timestamp"] >= test_start_ts) & (df["timestamp"].dt.hour == 0)].index.tolist()
    daily_origins = [t for t in daily_origins if t + 24 <= total_len]

    res_24 = {"actual": [], "persistence": [], "snaive_24": [], "snaive_168": []}
    for t in daily_origins:
        y_true_window = df.loc[t : t + 23, "avg_power_kw"].values
        # Persistence: carried forward flat-line from origin t-1
        pred_persist = np.full(24, df.loc[t - 1, "avg_power_kw"])
        # Seasonal Naive 24h: previous 24-hour diurnal cycle [t-24 to t-1]
        pred_s24 = df.loc[t - 24 : t - 1, "avg_power_kw"].values
        # Seasonal Naive 168h: same 24 hours from 1 week ago [t-168 to t-145]
        pred_s168 = df.loc[t - 168 : t - 145, "avg_power_kw"].values

        res_24["actual"].extend(y_true_window)
        res_24["persistence"].extend(pred_persist)
        res_24["snaive_24"].extend(pred_s24)
        res_24["snaive_168"].extend(pred_s168)

    y_24_true = np.array(res_24["actual"])
    metrics_24 = []
    print(f"\n[Day-Ahead Horizon H=24 (Rolling Daily Origin at 00:00 UTC | {len(daily_origins)} windows = {len(y_24_true):,} hourly predictions)]")
    for m_key, m_name in [("persistence", "Persistence (Flat H=24)"), ("snaive_24", "Seasonal Naive (Diurnal H=24)"), ("snaive_168", "Seasonal Naive (Weekly H=24)")]:
        y_pred = np.array(res_24[m_key])
        m = calculate_metrics(y_24_true, y_pred, eps_threshold=1.0)
        print(f"  {m_name:<30} | MAE: {m['mae_kw']:>6.2f} kW | RMSE: {m['rmse_kw']:>6.2f} kW | MAPE (>=1kW): {m['mape_filtered_pct']:>6.2f}% | sMAPE: {m['smape_pct']:>6.2f}%")
        metrics_24.append({
            "horizon": "H=24 (Day-Ahead)",
            "evaluation_type": "Rolling Daily Origin (00:00 UTC)",
            "model": m_name,
            "forecast_windows": len(daily_origins),
            "evaluated_hours": len(y_24_true),
            "mae_kw": m["mae_kw"],
            "rmse_kw": m["rmse_kw"],
            "mape_filtered_pct": m["mape_filtered_pct"],
            "smape_pct": m["smape_pct"],
            "mape_evaluated_samples": m["mape_eval_samples"],
        })

    # -------------------------------------------------------------
    # 2. Week-Ahead Horizon (H=168): Weekly Rolling Origin
    # -------------------------------------------------------------
    first_test_idx = df[df["timestamp"] >= test_start_ts].index[0]
    weekly_origins = [t for t in range(first_test_idx, total_len, 168) if t + 168 <= total_len]

    res_168 = {"actual": [], "persistence": [], "snaive_24": [], "snaive_168": []}
    for t in weekly_origins:
        y_true_window = df.loc[t : t + 167, "avg_power_kw"].values
        # Persistence: carried forward flat-line from origin t-1
        pred_persist = np.full(168, df.loc[t - 1, "avg_power_kw"])
        # Seasonal Naive 24h: tile last 24h diurnal pattern 7 times
        pred_s24 = np.tile(df.loc[t - 24 : t - 1, "avg_power_kw"].values, 7)
        # Seasonal Naive 168h: last 168 hours
        pred_s168 = df.loc[t - 168 : t - 1, "avg_power_kw"].values

        res_168["actual"].extend(y_true_window)
        res_168["persistence"].extend(pred_persist)
        res_168["snaive_24"].extend(pred_s24)
        res_168["snaive_168"].extend(pred_s168)

    y_168_true = np.array(res_168["actual"])
    metrics_168 = []
    print(f"\n[Week-Ahead Horizon H=168 (Rolling Weekly Origin | {len(weekly_origins)} windows = {len(y_168_true):,} hourly predictions)]")
    for m_key, m_name in [("persistence", "Persistence (Flat H=168)"), ("snaive_24", "Seasonal Naive (Diurnal Tiled H=168)"), ("snaive_168", "Seasonal Naive (Weekly Cycle H=168)")]:
        y_pred = np.array(res_168[m_key])
        m = calculate_metrics(y_168_true, y_pred, eps_threshold=1.0)
        print(f"  {m_name:<38} | MAE: {m['mae_kw']:>6.2f} kW | RMSE: {m['rmse_kw']:>6.2f} kW | MAPE (>=1kW): {m['mape_filtered_pct']:>6.2f}% | sMAPE: {m['smape_pct']:>6.2f}%")
        metrics_168.append({
            "horizon": "H=168 (Week-Ahead)",
            "evaluation_type": "Rolling Weekly Origin",
            "model": m_name,
            "forecast_windows": len(weekly_origins),
            "evaluated_hours": len(y_168_true),
            "mae_kw": m["mae_kw"],
            "rmse_kw": m["rmse_kw"],
            "mape_filtered_pct": m["mape_filtered_pct"],
            "smape_pct": m["smape_pct"],
            "mape_evaluated_samples": m["mape_eval_samples"],
        })

    df_multi = pd.DataFrame(metrics_24 + metrics_168)
    out_path = reports_dir / "baseline_multihorizon_comparison.csv"
    df_multi.to_csv(out_path, index=False)
    print(f"\n[+] Saved multi-horizon evaluation to: {out_path}")

    # Plot Multi-Horizon Comparison
    print("[*] Generating multi-horizon visualization...")
    sns.set_theme(style="whitegrid", palette="deep")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart for MAE
    df_plot_24 = df_multi[df_multi["horizon"] == "H=24 (Day-Ahead)"]
    df_plot_168 = df_multi[df_multi["horizon"] == "H=168 (Week-Ahead)"]

    x_labels = ["Persistence", "Seasonal 24h", "Seasonal 168h"]
    x = np.arange(len(x_labels))
    width = 0.35

    ax1.bar(x - width/2, df_plot_24["mae_kw"], width, label="Day-Ahead (H=24)", color="#1f77b4")
    ax1.bar(x + width/2, df_plot_168["mae_kw"], width, label="Week-Ahead (H=168)", color="#ff7f0e")
    ax1.set_title("Multi-Horizon Mean Absolute Error (MAE in kW)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("MAE (kW)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Bar chart for RMSE
    ax2.bar(x - width/2, df_plot_24["rmse_kw"], width, label="Day-Ahead (H=24)", color="#2ca02c")
    ax2.bar(x + width/2, df_plot_168["rmse_kw"], width, label="Week-Ahead (H=168)", color="#d62728")
    ax2.set_title("Multi-Horizon Root Mean Squared Error (RMSE in kW)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("RMSE (kW)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = figures_dir / "baseline_multihorizon_forecasts.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[+] Saved multi-horizon figure to: {fig_path}")


def main():
    root = get_project_root()
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    processed_dir = root / "data" / "processed"

    print("==========================================================================")
    print("  Phase 2: Extended Baseline Evaluation & Anomaly Diagnostic Pipeline")
    print("==========================================================================")

    # Load datasets
    df_hourly = pd.read_parquet(processed_dir / "caltech_hourly_demand.parquet")
    df_sessions = pd.read_parquet(processed_dir / "caltech_sessions_summary.parquet")

    df_hourly["timestamp"] = pd.to_datetime(df_hourly["timestamp"], utc=True)
    df_sessions["connectionTime"] = pd.to_datetime(df_sessions["connectionTime"], utc=True)
    df_sessions["disconnectTime"] = pd.to_datetime(df_sessions["disconnectTime"], utc=True)

    df_hourly = df_hourly.sort_values("timestamp").reset_index(drop=True)

    # 1. Generate 1-step ahead lags for rolling evaluation
    df_hourly["pred_persistence"] = df_hourly["avg_power_kw"].shift(1)
    df_hourly["pred_snaive_24"] = df_hourly["avg_power_kw"].shift(24)
    df_hourly["pred_snaive_168"] = df_hourly["avg_power_kw"].shift(168)

    # 2. Investigate extreme values
    investigate_extreme_values(df_hourly, df_sessions, reports_dir)

    # 3. Test set seasonal breakdown
    evaluate_seasonal_breakdown(df_hourly, reports_dir)

    # 4. Multi-horizon evaluation
    evaluate_multihorizon(df_hourly, reports_dir, figures_dir)

    print("\n==========================================================================")
    print("  All Phase 2 Extended Evaluations Completed Successfully!")
    print("==========================================================================")


if __name__ == "__main__":
    main()
