"""
Phase 3: Unified Cross-Model Synthesis & Comparative Evaluation
===============================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models

This script consolidates predictions, metrics, and seasonal breakdowns across all Phase 3 models:
1. Corrected v2 Baselines (Persistence, Seasonal Naive 24h, Seasonal Naive 168h)
2. Holt-Winters Exponential Smoothing
3. SARIMA (Box-Jenkins identified on train, selected on val)
4. LightGBM GBDT (Causal lags, rolling statistics, calendar features)
5. Chronos Foundation Time Series Model (chronos-t5-tiny)

Outputs:
- `reports/phase3/phase3_models_comparison.csv`
- `reports/phase3/phase3_seasonal_breakdown_comparison.csv`
- `reports/phase3/phase3_timestamp_alignment_audit.csv`
- `reports/phase3/figures/phase3_day_ahead_comparison.png`
- `reports/phase3/figures/phase3_metrics_summary_barchart.png`
- `reports/phase3/figures/phase3_seasonal_breakdown.png`
- `reports/phase3/figures/phase3_rolling_1step_comparison.png`
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from phase3_eval_utils import (
    get_project_root,
    get_phase3_reports_dir,
    calculate_metrics,
)


def load_all_metrics(reports_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and concatenates metric tables and seasonal breakdown tables from all models."""
    metric_files = [
        reports_dir / "baseline_v2_metrics.csv",
        reports_dir / "holt_winters_metrics.csv",
        reports_dir / "sarima_metrics.csv",
        reports_dir / "lightgbm_metrics.csv",
        reports_dir / "chronos_metrics.csv",
    ]

    seasonal_files = [
        reports_dir / "baseline_v2_seasonal_breakdown.csv",
        reports_dir / "holt_winters_seasonal_breakdown.csv",
        reports_dir / "sarima_seasonal_breakdown.csv",
        reports_dir / "lightgbm_seasonal_breakdown.csv",
        reports_dir / "chronos_seasonal_breakdown.csv",
    ]

    metrics_list = []
    for f in metric_files:
        if f.exists():
            df_m = pd.read_csv(f)
            if "total_samples" in df_m.columns and "n_samples" not in df_m.columns:
                df_m = df_m.rename(columns={"total_samples": "n_samples"})
            metrics_list.append(df_m)
        else:
            print(f"[!] Warning: Missing metric file: {f}")

    seasonal_list = []
    for f in seasonal_files:
        if f.exists():
            df_s = pd.read_csv(f)
            if "total_samples" in df_s.columns and "n_samples" not in df_s.columns:
                df_s = df_s.rename(columns={"total_samples": "n_samples"})
            seasonal_list.append(df_s)
        else:
            print(f"[!] Warning: Missing seasonal file: {f}")

    df_all_metrics = pd.concat(metrics_list, ignore_index=True)
    df_all_seasonal = pd.concat(seasonal_list, ignore_index=True)

    # Filter to Test split for comparison
    df_test_metrics = df_all_metrics[df_all_metrics["split"] == "Test"].copy()
    return df_test_metrics, df_all_seasonal


def verify_timestamp_alignments(reports_dir: Path) -> pd.DataFrame:
    """
    Verifies that all Day-Ahead (H=24) and 1-Step (H=1) prediction files have
    identical timestamps and row counts across all evaluated models.
    """
    print("\n--- Auditing Timestamp Alignments across All Phase 3 Models ---")
    h24_files = {
        "Baseline_v2": reports_dir / "baseline_v2_h24_predictions.parquet",
        "Holt_Winters": reports_dir / "holt_winters_h24_predictions.parquet",
        "SARIMA": reports_dir / "sarima_h24_predictions.parquet",
        "LightGBM": reports_dir / "lightgbm_h24_predictions.parquet",
        "Chronos": reports_dir / "chronos_h24_predictions.parquet",
    }

    h1_files = {
        "Baseline_v2": reports_dir / "baseline_v2_h1_predictions.parquet",
        "Holt_Winters": reports_dir / "holt_winters_h1_predictions.parquet",
        "SARIMA": reports_dir / "sarima_h1_predictions.parquet",
        "LightGBM": reports_dir / "lightgbm_h1_predictions.parquet",
        "Chronos": reports_dir / "chronos_h1_predictions.parquet",
    }

    audit_records = []
    ref_h24_ts = None
    ref_h1_ts = None

    for m_name, path in h24_files.items():
        if path.exists():
            df_pred = pd.read_parquet(path)
            ts = pd.to_datetime(df_pred["timestamp"], utc=True)
            if ref_h24_ts is None:
                ref_h24_ts = ts
                match = True
            else:
                match = (ref_h24_ts.equals(ts))

            audit_records.append({
                "task": "Day-Ahead (H=24)",
                "model": m_name,
                "n_timestamps": len(ts),
                "min_timestamp": str(ts.min()),
                "max_timestamp": str(ts.max()),
                "matches_reference_timestamps": match,
            })

    for m_name, path in h1_files.items():
        if path.exists():
            df_pred = pd.read_parquet(path)
            ts = pd.to_datetime(df_pred["timestamp"], utc=True)
            if ref_h1_ts is None:
                ref_h1_ts = ts
                match = True
            else:
                match = (ref_h1_ts.equals(ts))

            audit_records.append({
                "task": "Rolling 1-Step (H=1)",
                "model": m_name,
                "n_timestamps": len(ts),
                "min_timestamp": str(ts.min()),
                "max_timestamp": str(ts.max()),
                "matches_reference_timestamps": match,
            })

    df_audit = pd.DataFrame(audit_records)
    print(df_audit.to_string(index=False))
    return df_audit


def generate_synthesis_plots(reports_dir: Path, figures_dir: Path, df_metrics: pd.DataFrame, df_seasonal: pd.DataFrame):
    """Generates comparative visualizations across all Phase 3 models."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # 1. Bar Chart: Day-Ahead (H=24) MAE and RMSE Comparison
    h24_metrics = df_metrics[df_metrics["task"] == "Day-Ahead (H=24)"].sort_values("mae_kw")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    palette = sns.color_palette("mako", len(h24_metrics))
    bars1 = ax1.barh(h24_metrics["model"], h24_metrics["mae_kw"], color=palette)
    ax1.set_title("Day-Ahead (H=24) Test MAE (kW) - Lower is Better", fontsize=12, fontweight="bold")
    ax1.set_xlabel("MAE (kW)")
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 0.15, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=10, fontweight="bold")

    bars2 = ax2.barh(h24_metrics["model"], h24_metrics["rmse_kw"], color=palette)
    ax2.set_title("Day-Ahead (H=24) Test RMSE (kW) - Lower is Better", fontsize=12, fontweight="bold")
    ax2.set_xlabel("RMSE (kW)")
    for bar in bars2:
        w = bar.get_width()
        ax2.text(w + 0.2, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p1 = figures_dir / "phase3_metrics_summary_barchart.png"
    plt.savefig(p1, dpi=200)
    plt.close()
    print(f"[+] Saved summary bar chart to: {p1}")

    # 2. Detailed Forecasts Overlay (March 1–14, 2020 Shock Onset)
    p_hw = pd.read_parquet(reports_dir / "holt_winters_h24_predictions.parquet")
    p_sar = pd.read_parquet(reports_dir / "sarima_h24_predictions.parquet")
    p_lgb = pd.read_parquet(reports_dir / "lightgbm_h24_predictions.parquet")
    p_chr = pd.read_parquet(reports_dir / "chronos_h24_predictions.parquet")

    # Filter to March 1-14 window
    start_w = "2020-03-01 00:00:00+00:00"
    end_w = "2020-03-14 23:00:00+00:00"

    sub_hw = p_hw[(p_hw["timestamp"] >= start_w) & (p_hw["timestamp"] <= end_w)]
    sub_sar = p_sar[(p_sar["timestamp"] >= start_w) & (p_sar["timestamp"] <= end_w)]
    sub_lgb = p_lgb[(p_lgb["timestamp"] >= start_w) & (p_lgb["timestamp"] <= end_w)]
    sub_chr = p_chr[(p_chr["timestamp"] >= start_w) & (p_chr["timestamp"] <= end_w)]

    plt.figure(figsize=(15, 6))
    plt.plot(sub_hw["timestamp"], sub_hw["avg_power_kw"], label="Actual Demand", color="black", linewidth=2.5, zorder=5)
    plt.plot(sub_sar["timestamp"], sub_sar["pred_sarima"], label="SARIMA (2,0,1)x(0,1,1,24)", color="#2a9d8f", linestyle="--", linewidth=1.5)
    plt.plot(sub_lgb["timestamp"], sub_lgb["pred_lightgbm"], label="LightGBM GBDT", color="#e76f51", linestyle="-.", linewidth=1.5)
    plt.plot(sub_chr["timestamp"], sub_chr["pred_chronos"], label="Chronos (chronos-t5-tiny)", color="#7209b7", linestyle=":", linewidth=1.8)
    plt.plot(sub_hw["timestamp"], sub_hw["pred_holt_winters"], label="Holt-Winters", color="#457b9d", linestyle="--", alpha=0.6, linewidth=1.2)

    plt.title("Day-Ahead (H=24) Forecast Comparison across Models (March 1–14, 2020)", fontsize=13, fontweight="bold")
    plt.ylabel("Power Demand (kW)")
    plt.xlabel("Timestamp (UTC)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    p2 = figures_dir / "phase3_day_ahead_comparison.png"
    plt.savefig(p2, dpi=200)
    plt.close()
    print(f"[+] Saved forecast comparison overlay to: {p2}")

    # 3. Seasonal Breakdown Heatmap/Bar chart
    h24_seasonal = df_seasonal[df_seasonal["task"] == "Day-Ahead (H=24)"].copy()
    # Order periods chronologically
    order_periods = ["March–May 2020", "June–August 2020", "September–December 2020"]
    h24_seasonal["period"] = pd.Categorical(h24_seasonal["period"], categories=order_periods, ordered=True)
    h24_seasonal = h24_seasonal.sort_values(["period", "mae_kw"])

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=h24_seasonal,
        x="period",
        y="mae_kw",
        hue="model",
        palette="tab10",
    )
    plt.title("Day-Ahead (H=24) Test MAE (kW) Across Seasonal Sub-Periods (2020 COVID Trajectory)", fontsize=12, fontweight="bold")
    plt.ylabel("MAE (kW)")
    plt.xlabel("Sub-Period")
    plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    p3 = figures_dir / "phase3_seasonal_breakdown.png"
    plt.savefig(p3, dpi=200)
    plt.close()
    print(f"[+] Saved seasonal breakdown plot to: {p3}")

    # 4. Rolling 1-Step (H=1) Forecast Overlay
    p1_hw = pd.read_parquet(reports_dir / "holt_winters_h1_predictions.parquet")
    p1_sar = pd.read_parquet(reports_dir / "sarima_h1_predictions.parquet")
    p1_lgb = pd.read_parquet(reports_dir / "lightgbm_h1_predictions.parquet")
    p1_chr = pd.read_parquet(reports_dir / "chronos_h1_predictions.parquet")

    sub1_hw = p1_hw[(p1_hw["timestamp"] >= start_w) & (p1_hw["timestamp"] <= end_w)]
    sub1_sar = p1_sar[(p1_sar["timestamp"] >= start_w) & (p1_sar["timestamp"] <= end_w)]
    sub1_lgb = p1_lgb[(p1_lgb["timestamp"] >= start_w) & (p1_lgb["timestamp"] <= end_w)]
    sub1_chr = p1_chr[(p1_chr["timestamp"] >= start_w) & (p1_chr["timestamp"] <= end_w)]

    plt.figure(figsize=(15, 6))
    plt.plot(sub1_hw["timestamp"], sub1_hw["avg_power_kw"], label="Actual Demand", color="black", linewidth=2.5, zorder=5)
    plt.plot(sub1_sar["timestamp"], sub1_sar["pred_sarima"], label="SARIMA (2,0,1)x(0,1,1,24)", color="#2a9d8f", linestyle="--", linewidth=1.5)
    plt.plot(sub1_lgb["timestamp"], sub1_lgb["pred_lightgbm"], label="LightGBM GBDT", color="#e76f51", linestyle="-.", linewidth=1.5)
    plt.plot(sub1_chr["timestamp"], sub1_chr["pred_chronos"], label="Chronos (chronos-t5-tiny)", color="#7209b7", linestyle=":", linewidth=1.8)
    plt.plot(sub1_hw["timestamp"], sub1_hw["pred_holt_winters"], label="Holt-Winters", color="#457b9d", linestyle="--", alpha=0.6, linewidth=1.2)

    plt.title("Rolling 1-Step (H=1) Forecast Comparison across Models (March 1–14, 2020)", fontsize=13, fontweight="bold")
    plt.ylabel("Power Demand (kW)")
    plt.xlabel("Timestamp (UTC)")
    plt.legend(loc="upper right")
    plt.tight_layout()
    p4 = figures_dir / "phase3_rolling_1step_comparison.png"
    plt.savefig(p4, dpi=200)
    plt.close()
    print(f"[+] Saved rolling 1-step forecast comparison overlay to: {p4}")


def main():
    print("=========================================================================")
    print("  Phase 3: Unified Cross-Model Synthesis & Comparative Evaluation")
    print("=========================================================================")

    reports_dir = get_phase3_reports_dir()
    figures_dir = reports_dir / "figures"

    # Step 1: Verify Timestamp Alignments
    df_audit = verify_timestamp_alignments(reports_dir)
    df_audit.to_csv(reports_dir / "phase3_timestamp_alignment_audit.csv", index=False)

    # Step 2: Load and Combine All Metrics
    df_test_metrics, df_all_seasonal = load_all_metrics(reports_dir)

    # Sort Day-Ahead by MAE
    df_test_metrics = df_test_metrics.sort_values(["task", "mae_kw"], ascending=[True, True])
    df_test_metrics.to_csv(reports_dir / "phase3_models_comparison.csv", index=False)
    df_all_seasonal.to_csv(reports_dir / "phase3_seasonal_breakdown_comparison.csv", index=False)

    print("\n--- Master Comparison: Primary Task - Day-Ahead (H=24) ---")
    h24_table = df_test_metrics[df_test_metrics["task"] == "Day-Ahead (H=24)"][
        ["model", "n_samples", "mae_kw", "rmse_kw", "mape_filtered_pct", "smape_pct"]
    ]
    print(h24_table.to_string(index=False))

    print("\n--- Master Comparison: Secondary Task - Rolling 1-Step (H=1) ---")
    h1_table = df_test_metrics[df_test_metrics["task"] == "Rolling 1-Step (H=1)"][
        ["model", "n_samples", "mae_kw", "rmse_kw", "mape_filtered_pct", "smape_pct"]
    ]
    print(h1_table.to_string(index=False))

    # Step 3: Generate Visualizations
    generate_synthesis_plots(reports_dir, figures_dir, df_test_metrics, df_all_seasonal)

    print(f"\n[+] Master model comparison saved to: {reports_dir / 'phase3_models_comparison.csv'}")
    print(f"[+] Seasonal breakdown comparison saved to: {reports_dir / 'phase3_seasonal_breakdown_comparison.csv'}")


if __name__ == "__main__":
    main()
