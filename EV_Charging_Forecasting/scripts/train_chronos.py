"""
Phase 3 - Model 4: Chronos Foundation Time Series Model Pipeline
================================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models

METHODOLOGY AND CAVEAT DOCUMENTATION:
- Target Series: 'avg_power_kw' represents hourly facility energy consumption (kWh/h = kW)
  reconstructed via uniform energy allocation across connection intervals.
  It is NOT measured instantaneous charging power. Intra-session dynamics (e.g., CC/CV
  tapering, hardware throttling) are smoothed out by definition.
- Provisional Cap: The 7.7 kW per-session power cap applied during v2 preprocessing remains
  provisional, based on standard Level 2 EVSE maximum delivery (32A @ 240V). While it successfully
  suppressed unphysical meter-carryover spikes (reducing the peak from 375.4 kW to 120.6 kW),
  station-level hardware electrical ratings were not verified against primary engineering spec sheets.
- Chronological Partition:
  * Training:   2018-05-01 00:00 to 2019-12-31 23:00 UTC (14,640 hours)
  * Validation: 2020-01-01 00:00 to 2020-02-29 23:00 UTC (1,440 hours)
  * Test:       2020-03-01 00:00 to 2020-12-05 03:00 UTC (6,700 hours)
- Foundation Model Architecture:
  * Chronos (amazon/chronos-t5-tiny) zero-shot pretrained probabilistic time series model.
  * Point forecasts derived via sample median quantile (q=0.50).
- Model Selection & Hyperparameter Tuning:
  * Context window lengths (168h, 336h, 504h) evaluated strictly on the Validation set (Jan–Feb 2020).
  * No test set feedback is used during configuration selection.
- Tasks Evaluated:
  * Primary: 24-hour Day-Ahead Forecasting (H=24) using daily rolling origin at 00:00 UTC.
  * Secondary: Rolling 1-Step-Ahead Forecasting (H=1) using previous hour feedback.
- Post-processing:
  * Non-negative clipping (demand >= 0 kW) is enforced physically.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from chronos import ChronosPipeline

from phase3_eval_utils import (
    get_project_root,
    get_phase3_reports_dir,
    load_v2_dataset,
    calculate_metrics,
    get_split_masks,
    get_daily_forecast_origins,
    get_test_subperiods,
)


def tune_chronos_on_validation(
    df: pd.DataFrame,
    pipeline: ChronosPipeline,
    val_origins: List[int],
) -> Tuple[int, pd.DataFrame]:
    """
    Evaluates candidate context window lengths (168h, 336h, 504h) on the Validation set.
    Selects the context window minimizing Validation Day-Ahead H=24 MAE.
    """
    print("\n--- Tuning Chronos Context Length on Validation Data (Jan-Feb 2020) ---")
    y_arr = df["avg_power_kw"].values

    candidates = [168, 336, 504]
    tuning_records = []

    actuals_h24 = np.array([y_arr[t : t + 24] for t in val_origins])

    for ctx_len in candidates:
        print(f"[*] Evaluating context length = {ctx_len} hours...")
        contexts = [torch.tensor(y_arr[t - ctx_len : t], dtype=torch.float32) for t in val_origins]

        # Predict batch of 60 validation days
        forecasts = pipeline.predict(contexts, prediction_length=24, num_samples=20)
        # Median quantile point forecast
        median_preds = np.clip(torch.quantile(forecasts, 0.5, dim=1).numpy(), 0, None)

        m = calculate_metrics(actuals_h24.flatten(), median_preds.flatten())
        print(
            f"  Context {ctx_len:>3}h | Val H=24 MAE: {m['mae_kw']:>6.2f} kW | "
            f"RMSE: {m['rmse_kw']:>6.2f} kW | sMAPE: {m['smape_pct']:>5.2f}%"
        )

        tuning_records.append({
            "model": "Chronos (chronos-t5-tiny)",
            "context_length_hours": ctx_len,
            "val_mae_kw": m["mae_kw"],
            "val_rmse_kw": m["rmse_kw"],
            "val_mape_pct": m["mape_filtered_pct"],
            "val_smape_pct": m["smape_pct"],
        })

    df_tuning = pd.DataFrame(tuning_records)
    best_row = df_tuning.sort_values("val_mae_kw").iloc[0]
    best_ctx_len = int(best_row["context_length_hours"])

    print(f"\n[+] Selected Best Context Length on Validation: {best_ctx_len} hours")
    print(f"    Validation MAE: {best_row['val_mae_kw']:.2f} kW | RMSE: {best_row['val_rmse_kw']:.2f} kW")

    return best_ctx_len, df_tuning


def evaluate_chronos_test(
    df: pd.DataFrame,
    pipeline: ChronosPipeline,
    best_ctx_len: int,
    test_mask: pd.Series,
    reports_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates Chronos with optimal context length on the Test set:
    - Day-Ahead (H=24) rolling daily origins (6,696 hours across 279 days)
    - Rolling 1-Step (H=1) across all 6,700 test hours (batched in chunks)
    - Seasonal breakdowns across March–May, June–August, September–December 2020.
    """
    print(f"\n--- Evaluating Chronos on Test Data (Mar-Dec 2020) with Context={best_ctx_len}h ---")
    y_arr = df["avg_power_kw"].values
    ts_arr = df["timestamp"].values

    # 1. Day-Ahead H=24 Task
    test_origins = get_daily_forecast_origins(df, split="test")
    print(f"[*] Running Day-Ahead H=24 rolling forecasts across {len(test_origins)} origins...")

    contexts_h24 = [torch.tensor(y_arr[t - best_ctx_len : t], dtype=torch.float32) for t in test_origins]
    # Predict in chunks of 50 to avoid memory pressure
    chunk_size = 50
    preds_h24_list = []
    for i in range(0, len(contexts_h24), chunk_size):
        chunk = contexts_h24[i : i + chunk_size]
        fc = pipeline.predict(chunk, prediction_length=24, num_samples=20)
        med = np.clip(torch.quantile(fc, 0.5, dim=1).numpy(), 0, None)
        preds_h24_list.append(med)

    preds_h24_all = np.vstack(preds_h24_list)

    h24_rows = []
    for idx_o, t in enumerate(test_origins):
        origin_ts = pd.to_datetime(ts_arr[t], utc=True)
        for step in range(24):
            h24_rows.append({
                "origin_timestamp": origin_ts,
                "timestamp": pd.to_datetime(ts_arr[t + step], utc=True),
                "step": step + 1,
                "avg_power_kw": float(y_arr[t + step]),
                "pred_chronos": float(preds_h24_all[idx_o, step]),
            })

    df_h24_pred = pd.DataFrame(h24_rows)

    # 2. Rolling 1-Step H=1 Task
    test_indices = df[test_mask].index.tolist()
    print(f"[*] Running Rolling 1-Step (H=1) forecasts across {len(test_indices)} test hours...")

    # We batch the 1-step predictions in chunks of 250
    chunk_size_h1 = 250
    h1_preds_all = []
    for i in range(0, len(test_indices), chunk_size_h1):
        chunk_indices = test_indices[i : i + chunk_size_h1]
        contexts_h1 = [torch.tensor(y_arr[t - best_ctx_len : t], dtype=torch.float32) for t in chunk_indices]
        fc1 = pipeline.predict(contexts_h1, prediction_length=1, num_samples=10)
        med1 = np.clip(torch.quantile(fc1, 0.5, dim=1).numpy().flatten(), 0, None)
        h1_preds_all.extend(med1.tolist())

    df_h1_pred = pd.DataFrame({
        "timestamp": pd.to_datetime(ts_arr[test_indices], utc=True),
        "avg_power_kw": y_arr[test_indices],
        "pred_chronos": np.array(h1_preds_all),
    })

    # Metrics Summary
    metrics_records = []
    seasonal_records = []

    m_h24 = calculate_metrics(df_h24_pred["avg_power_kw"].values, df_h24_pred["pred_chronos"].values)
    m_h1 = calculate_metrics(df_h1_pred["avg_power_kw"].values, df_h1_pred["pred_chronos"].values)

    metrics_records.append({
        "model": "Chronos (chronos-t5-tiny)",
        "task": "Day-Ahead (H=24)",
        "split": "Test",
        "n_samples": len(df_h24_pred),
        "mae_kw": m_h24["mae_kw"],
        "rmse_kw": m_h24["rmse_kw"],
        "mape_filtered_pct": m_h24["mape_filtered_pct"],
        "smape_pct": m_h24["smape_pct"],
        "mape_evaluated_samples": m_h24["mape_evaluated_samples"],
        "config_name": f"chronos_tiny_ctx{best_ctx_len}",
    })

    metrics_records.append({
        "model": "Chronos (chronos-t5-tiny)",
        "task": "Rolling 1-Step (H=1)",
        "split": "Test",
        "n_samples": len(df_h1_pred),
        "mae_kw": m_h1["mae_kw"],
        "rmse_kw": m_h1["rmse_kw"],
        "mape_filtered_pct": m_h1["mape_filtered_pct"],
        "smape_pct": m_h1["smape_pct"],
        "mape_evaluated_samples": m_h1["mape_evaluated_samples"],
        "config_name": f"chronos_tiny_ctx{best_ctx_len}",
    })

    # Seasonal Breakdowns
    for task_name, pred_df in [("Day-Ahead (H=24)", df_h24_pred), ("Rolling 1-Step (H=1)", df_h1_pred)]:
        for period_name, start_ts, end_ts in get_test_subperiods():
            p_mask = (pred_df["timestamp"] >= start_ts) & (pred_df["timestamp"] <= end_ts)
            sub = pred_df[p_mask]
            if len(sub) > 0:
                m_sub = calculate_metrics(sub["avg_power_kw"].values, sub["pred_chronos"].values)
                seasonal_records.append({
                    "model": "Chronos (chronos-t5-tiny)",
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
        "model_name": "Chronos Foundation Time Series Model",
        "hf_model_id": "amazon/chronos-t5-tiny",
        "selected_context_length_hours": best_ctx_len,
        "target_variable": "avg_power_kw (reconstructed hourly energy allocation; not instantaneous physical power)",
        "energy_cap_note": "7.7 kW per-session provisional cap applied during v2 preprocessing",
        "train_range": "2018-05-01 00:00 to 2019-12-31 23:00 UTC",
        "validation_range": "2020-01-01 00:00 to 2020-02-29 23:00 UTC",
        "test_range": "2020-03-01 00:00 to 2020-12-05 03:00 UTC",
    }

    return df_metrics, df_seasonal, df_h24_pred, df_h1_pred, config_info


def generate_plots(df_h24: pd.DataFrame, df_h1: pd.DataFrame, figures_dir: Path):
    """Generates comparison visual of Chronos forecasts vs Actuals on the test set."""
    sns.set_theme(style="whitegrid")
    sample_start = "2020-03-01 00:00:00+00:00"
    sample_end = "2020-03-14 23:00:00+00:00"

    sub_h24 = df_h24[(df_h24["timestamp"] >= sample_start) & (df_h24["timestamp"] <= sample_end)]
    sub_h1 = df_h1[(df_h1["timestamp"] >= sample_start) & (df_h1["timestamp"] <= sample_end)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(sub_h24["timestamp"], sub_h24["avg_power_kw"], label="Actual Demand (kW)", color="black", linewidth=2.0)
    ax1.plot(sub_h24["timestamp"], sub_h24["pred_chronos"], label="Chronos Day-Ahead (H=24)", color="#7209b7", linestyle="--", linewidth=1.6)
    ax1.set_title("Chronos Day-Ahead (H=24) Forecast vs. Actual Demand (March 1–14, 2020)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Power Demand (kW)")
    ax1.legend(loc="upper right")

    ax2.plot(sub_h1["timestamp"], sub_h1["avg_power_kw"], label="Actual Demand (kW)", color="black", linewidth=2.0)
    ax2.plot(sub_h1["timestamp"], sub_h1["pred_chronos"], label="Chronos Rolling 1-Step (H=1)", color="#4361ee", linestyle="-.", linewidth=1.6)
    ax2.set_title("Chronos Rolling 1-Step (H=1) Forecast vs. Actual Demand (March 1–14, 2020)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Power Demand (kW)")
    ax2.set_xlabel("Timestamp (UTC)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig_path = figures_dir / "chronos_forecasts.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[+] Saved forecast plot to: {fig_path}")


def main():
    print("=========================================================================")
    print("  Phase 3 - Model 4: Chronos Foundation Time Series Pipeline")
    print("=========================================================================")

    reports_dir = get_phase3_reports_dir()
    figures_dir = reports_dir / "figures"
    df = load_v2_dataset()

    train_mask, val_mask, test_mask = get_split_masks(df)
    val_origins = get_daily_forecast_origins(df, split="val")

    print("[*] Initializing Chronos Pipeline (amazon/chronos-t5-tiny)...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-tiny",
        device_map="cpu",
    )

    # Step 1: Tune context length on validation set
    best_ctx_len, df_tuning = tune_chronos_on_validation(df, pipeline, val_origins)
    df_tuning.to_csv(reports_dir / "chronos_validation_tuning.csv", index=False)

    # Step 2: Evaluate on test set
    df_metrics, df_seasonal, df_h24_pred, df_h1_pred, config_info = evaluate_chronos_test(
        df, pipeline, best_ctx_len, test_mask, reports_dir
    )

    # Step 3: Save results
    metrics_path = reports_dir / "chronos_metrics.csv"
    seasonal_path = reports_dir / "chronos_seasonal_breakdown.csv"
    config_path = reports_dir / "chronos_config.json"

    df_metrics.to_csv(metrics_path, index=False)
    df_seasonal.to_csv(seasonal_path, index=False)
    with open(config_path, "w") as f:
        json.dump(config_info, f, indent=2)

    df_h24_pred.to_parquet(reports_dir / "chronos_h24_predictions.parquet", index=False)
    df_h1_pred.to_parquet(reports_dir / "chronos_h1_predictions.parquet", index=False)

    print("\n--- Quantitative Test Performance: Chronos ---")
    print(df_metrics.to_string(index=False))

    print("\n--- Test Set Seasonal Breakdown ---")
    print(df_seasonal.to_string(index=False))

    # Step 4: Generate plots
    generate_plots(df_h24_pred, df_h1_pred, figures_dir)

    print(f"\n[+] Saved Chronos metrics to: {metrics_path}")
    print(f"[+] Saved Chronos seasonal breakdown to: {seasonal_path}")
    print(f"[+] Saved Chronos configuration to: {config_path}")


if __name__ == "__main__":
    main()

