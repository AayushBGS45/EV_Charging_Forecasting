"""
Phase 2 - Task 1: Baseline Forecasting Pipeline
================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models

This script implements, executes, and evaluates baseline time series forecasting models:
1. Chronological Train / Validation / Test Splitting:
   - Training:   < 2020-01-01 (14,640 hours)
   - Validation: 2020-01-01 to 2020-02-29 (1,440 hours)
   - Test:       2020-03-01 to 2020-12-05 (6,700 hours)
2. Forecasting Baselines:
   - Persistence / Last-Value (t-1): 1-step-ahead rolling forecast using immediately preceding observation
   - Seasonal Naive 24h (t-24): 1-step-ahead rolling forecast using the same hour yesterday
   - Seasonal Naive 168h (t-168): 1-step-ahead rolling forecast using the same hour of the same day last week
3. Performance Metrics:
   - MAE (Mean Absolute Error, kW)
   - RMSE (Root Mean Squared Error, kW)
   - Filtered MAPE (Mean Absolute Percentage Error on y_true >= 1.0 kW to avoid division-by-zero)
   - sMAPE (Symmetric Mean Absolute Percentage Error, %)
4. Output Artifacts:
   - `reports/baseline_model_comparison.csv`
   - `reports/figures/baseline_forecasts.png`

Limitation Note:
The target series `avg_power_kw` is constructed via session-energy allocation across connection intervals.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent


def load_dataset(root: Path) -> pd.DataFrame:
    """Loads the processed continuous hourly power demand dataset."""
    parquet_path = root / "data" / "processed" / "caltech_hourly_demand.parquet"
    csv_path = root / "data" / "processed" / "caltech_hourly_demand.csv"

    if parquet_path.exists():
        print(f"[*] Loading data from: {parquet_path}")
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"[*] Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"[!] Processed dataset not found in {root / 'data' / 'processed'}. "
            "Please run `python scripts/process_time_series.py` first."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps_threshold: float = 1.0) -> dict:
    """
    Computes MAE, RMSE, sMAPE, and Filtered MAPE (handling zero/near-zero actual values).
    """
    # 1. MAE
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # 2. RMSE
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # 3. Filtered MAPE (avoiding division by zero or near-zero demand)
    non_zero_mask = y_true >= eps_threshold
    num_eval_mape = int(np.sum(non_zero_mask))
    if num_eval_mape > 0:
        mape_filtered = float(np.mean(np.abs(y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]) * 100.0)
    else:
        mape_filtered = float("nan")

    # 4. sMAPE (Symmetric MAPE across all points)
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-6
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / denominator) * 100.0)

    return {
        "mae_kw": round(mae, 4),
        "rmse_kw": round(rmse, 4),
        "mape_filtered_pct": round(mape_filtered, 2),
        "smape_pct": round(smape, 2),
        "mape_eval_samples": num_eval_mape,
    }


def main():
    root = get_project_root()
    reports_dir = root / "reports"
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=====================================================================")
    print("  Phase 2 - Task 1: EV Charging Demand Baseline Forecasting Pipeline")
    print("=====================================================================")

    # Step 1: Load and inspect
    df = load_dataset(root)
    print(f"[+] Total observations: {len(df):,} hourly intervals")
    print(f"[+] Timeline range:     {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"[+] Target column:      'avg_power_kw' (Mean: {df['avg_power_kw'].mean():.2f} kW, Max: {df['avg_power_kw'].max():.2f} kW)")

    # Step 2: Chronological Train / Validation / Test Split
    train_mask = df["timestamp"] < "2020-01-01"
    val_mask = (df["timestamp"] >= "2020-01-01") & (df["timestamp"] < "2020-03-01")
    test_mask = df["timestamp"] >= "2020-03-01"

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()
    df_test = df[test_mask].copy()

    print("\n--- Step 2: Chronological Dataset Partition ---")
    print(f"  Training Set:   {len(df_train):>6,} hours ({df_train['timestamp'].min().strftime('%Y-%m-%d')} to {df_train['timestamp'].max().strftime('%Y-%m-%d')})")
    print(f"  Validation Set: {len(df_val):>6,} hours ({df_val['timestamp'].min().strftime('%Y-%m-%d')} to {df_val['timestamp'].max().strftime('%Y-%m-%d')})")
    print(f"  Test Set:       {len(df_test):>6,} hours ({df_test['timestamp'].min().strftime('%Y-%m-%d')} to {df_test['timestamp'].max().strftime('%Y-%m-%d')})")
    assert len(df_train) + len(df_val) + len(df_test) == len(df), "Data split row count mismatch!"
    print("[+] Chronological partition verified: No overlapping timestamps, strictly time-ordered.")

    # Step 3: Implement Baselines
    # All baselines are strictly causal (using only observations available before the target timestamp).
    # Horizon H=1 (1-step-ahead rolling forecast)
    print("\n--- Step 3: Generating Baseline Forecasts ---")
    df["pred_persistence"] = df["avg_power_kw"].shift(1)
    df["pred_snaive_24"] = df["avg_power_kw"].shift(24)
    df["pred_snaive_168"] = df["avg_power_kw"].shift(168)

    models = [
        ("Persistence (t-1)", "pred_persistence", "Last observed value from 1 hour prior (H=1)"),
        ("Seasonal Naive (t-24)", "pred_snaive_24", "Value from exactly 24 hours prior (diurnal lag, H=1)"),
        ("Seasonal Naive (t-168)", "pred_snaive_168", "Value from exactly 168 hours prior (weekly lag, H=1)"),
    ]

    # Step 4: Evaluate Models on Test and Validation sets
    print("\n--- Step 4: Quantitative Evaluation ---")
    eval_records = []

    for split_name, split_df in [("Validation", df[val_mask]), ("Test", df[test_mask])]:
        y_true = split_df["avg_power_kw"].values
        start_ts = split_df["timestamp"].min().isoformat()
        end_ts = split_df["timestamp"].max().isoformat()
        sample_count = len(split_df)

        print(f"\n[Evaluating on {split_name} Set ({sample_count:,} observations: {start_ts} to {end_ts})]")
        for model_name, col_name, description in models:
            y_pred = split_df[col_name].values
            metrics = calculate_metrics(y_true, y_pred, eps_threshold=1.0)

            print(
                f"  {model_name:<24} | MAE: {metrics['mae_kw']:>6.2f} kW | RMSE: {metrics['rmse_kw']:>6.2f} kW | "
                f"MAPE (>=1kW): {metrics['mape_filtered_pct']:>6.2f}% | sMAPE: {metrics['smape_pct']:>6.2f}%"
            )

            eval_records.append({
                "model": model_name,
                "split": split_name,
                "forecast_horizon": "1-hour ahead (rolling H=1)",
                "sample_count": sample_count,
                "mae_kw": metrics["mae_kw"],
                "rmse_kw": metrics["rmse_kw"],
                "mape_filtered_pct": metrics["mape_filtered_pct"],
                "smape_pct": metrics["smape_pct"],
                "mape_evaluated_samples": metrics["mape_eval_samples"],
                "evaluation_start": start_ts,
                "evaluation_end": end_ts,
                "description": description,
            })

    # Step 5: Save Output Comparison Table
    df_eval = pd.DataFrame(eval_records)
    csv_out_path = reports_dir / "baseline_model_comparison.csv"
    df_eval.to_csv(csv_out_path, index=False)
    print(f"\n[+] Saved baseline comparison metrics to: {csv_out_path}")

    # Step 6: Generate Forecast Visualization
    print("[*] Generating illustrative forecast comparison figure...")
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    # Select an illustrative 10-day window in early March 2020 (at the onset of the test set)
    plot_start = "2020-03-01 00:00:00+00:00"
    plot_end = "2020-03-14 23:00:00+00:00"
    window = df[(df["timestamp"] >= plot_start) & (df["timestamp"] <= plot_end)].copy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})

    # Top plot: Actuals vs Baseline predictions
    ax1.plot(window["timestamp"], window["avg_power_kw"], label="Actual Demand (kW)", color="black", linewidth=2.0)
    ax1.plot(window["timestamp"], window["pred_persistence"], label="Persistence (t-1)", color="#1f77b4", linestyle="--", alpha=0.8, linewidth=1.4)
    ax1.plot(window["timestamp"], window["pred_snaive_24"], label="Seasonal Naive (t-24)", color="#2ca02c", linestyle=":", alpha=0.8, linewidth=1.4)
    ax1.plot(window["timestamp"], window["pred_snaive_168"], label="Seasonal Naive (t-168)", color="#d62728", linestyle="-.", alpha=0.8, linewidth=1.4)

    ax1.set_title("EV Facility Charging Demand vs. Baseline Forecasts (Test Set: March 1–14, 2020)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Power Demand (kW)", fontsize=11)
    ax1.legend(loc="upper right", frameon=True, facecolor="white")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Bottom plot: Residual errors (Actual - Predicted)
    res_persistence = window["avg_power_kw"] - window["pred_persistence"]
    res_snaive24 = window["avg_power_kw"] - window["pred_snaive_24"]
    ax2.plot(window["timestamp"], res_persistence, label="Error: Persistence", color="#1f77b4", linewidth=1.0)
    ax2.plot(window["timestamp"], res_snaive24, label="Error: Seasonal Naive 24h", color="#2ca02c", linewidth=1.0, alpha=0.7)
    ax2.axhline(0, color="grey", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Error (kW)", fontsize=11)
    ax2.set_xlabel("Timestamp (UTC)", fontsize=11)
    ax2.legend(loc="lower right", frameon=True, facecolor="white")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = figures_dir / "baseline_forecasts.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[+] Saved forecast visualization to: {fig_path}")

    print("\n=====================================================================")
    print("  Phase 2 - Task 1 Successfully Completed!")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
