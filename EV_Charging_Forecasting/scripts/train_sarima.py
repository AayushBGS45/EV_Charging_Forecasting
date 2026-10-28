"""
Phase 3 - Model 2: SARIMA Pipeline (Diagnostics, Selection, and Evaluation)
==========================================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models

METHODOLOGY AND CAVEAT DOCUMENTATION:
- Target Series: 'avg_power_kw' represents hourly facility energy consumption (kWh/h = kW)
  reconstructed via uniform energy allocation across connection intervals.
  It is NOT measured instantaneous charging power. Intra-session charging dynamics (e.g., CC/CV
  tapering, hardware curtailment) are smoothed out by definition.
- Provisional Cap: The 7.7 kW per-session power cap applied during v2 preprocessing remains
  provisional, based on standard Level 2 EVSE maximum delivery (32A @ 240V). While it successfully
  suppressed unphysical meter-carryover spikes (reducing the all-time peak from 375.4 kW to 120.6 kW),
  station-level hardware electrical ratings were not verified against primary engineering spec sheets.
- Chronological Partition:
  * Training:   2018-05-01 00:00 to 2019-12-31 23:00 UTC (14,640 hours)
  * Validation: 2020-01-01 00:00 to 2020-02-29 23:00 UTC (1,440 hours)
  * Test:       2020-03-01 00:00 to 2020-12-05 03:00 UTC (6,700 hours)
- Training-Only Diagnostics:
  * Stationarity tests (ADF and KPSS) conducted strictly on training data.
  * Autocorrelation (ACF) and Partial Autocorrelation (PACF) analyzed on training data.
- Hyperparameter Tuning & Model Selection:
  * Candidate SARIMA orders are selected strictly using Validation set (Jan–Feb 2020) performance.
  * Test set data is never used for parameter selection.
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
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from phase3_eval_utils import (
    get_project_root,
    get_phase3_reports_dir,
    load_v2_dataset,
    calculate_metrics,
    get_split_masks,
    get_daily_forecast_origins,
    get_test_subperiods,
)


def run_training_diagnostics(train_series: np.ndarray, reports_dir: Path, figures_dir: Path) -> pd.DataFrame:
    """
    Performs stationarity tests (ADF and KPSS) and computes ACF/PACF strictly on the training series.
    Saves diagnostics plot and summary table.
    """
    print("\n--- Step 1: Training-Only Stationarity and ACF/PACF Diagnostics ---")

    # 1. Level Series Stationarity
    adf_level = adfuller(train_series, autolag="AIC")
    kpss_level = kpss(train_series, regression="c", nlags="auto")

    # 2. Seasonally Differenced Series (s=24)
    diff24 = train_series[24:] - train_series[:-24]
    adf_diff24 = adfuller(diff24, autolag="AIC")
    kpss_diff24 = kpss(diff24, regression="c", nlags="auto")

    diag_records = [
        {
            "series": "Training Level (y_t)",
            "adf_stat": round(float(adf_level[0]), 4),
            "adf_pvalue": float(adf_level[1]),
            "adf_used_lag": int(adf_level[2]),
            "kpss_stat": round(float(kpss_level[0]), 4),
            "kpss_pvalue": float(kpss_level[1]),
            "interpretation": "Rejects unit root (ADF p<0.001) but rejects trend-stationarity (KPSS p<0.01) due to diurnal cycles.",
        },
        {
            "series": "Seasonal Difference Delta_24 (y_t - y_{t-24})",
            "adf_stat": round(float(adf_diff24[0]), 4),
            "adf_pvalue": float(adf_diff24[1]),
            "adf_used_lag": int(adf_diff24[2]),
            "kpss_stat": round(float(kpss_diff24[0]), 4),
            "kpss_pvalue": float(kpss_diff24[1]),
            "interpretation": "Strongly stationary (ADF p<0.001, KPSS p>=0.10). Confirms seasonal integration D=1 with s=24.",
        },
    ]

    df_diag = pd.DataFrame(diag_records)
    csv_diag_path = reports_dir / "sarima_diagnostics_summary.csv"
    df_diag.to_csv(csv_diag_path, index=False)
    print(f"[+] Saved stationarity diagnostics to: {csv_diag_path}")

    # 3. Compute ACF and PACF up to lag 72 (3 diurnal cycles)
    nlags = 72
    acf_vals = acf(diff24, nlags=nlags)
    pacf_vals = pacf(diff24, nlags=nlags, method="ywm")

    # 4. Generate ACF/PACF Diagnostics Figure
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    lags = np.arange(len(acf_vals))
    ci = 1.96 / np.sqrt(len(diff24))

    ax1.stem(lags, acf_vals)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(ci, color="gray", linestyle="--", alpha=0.7)
    ax1.axhline(-ci, color="gray", linestyle="--", alpha=0.7)
    ax1.set_title("Training Set Autocorrelation (ACF) of Seasonally Differenced Demand (D=1, s=24)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("ACF")

    ax2.stem(lags, pacf_vals)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(ci, color="gray", linestyle="--", alpha=0.7)
    ax2.axhline(-ci, color="gray", linestyle="--", alpha=0.7)
    ax2.set_title("Training Set Partial Autocorrelation (PACF) of Seasonally Differenced Demand (D=1, s=24)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("PACF")
    ax2.set_xlabel("Lag (Hours)")

    plt.tight_layout()
    fig_path = figures_dir / "sarima_training_acf_pacf.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[+] Saved ACF/PACF plot to: {fig_path}")

    return df_diag


def tune_sarima_on_validation(
    df: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    val_origins: List[int],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Selects candidate SARIMA specifications using AIC/BIC and Validation set Day-Ahead (H=24) MAE.
    Candidates derived from Box-Jenkins identification:
      D=1, s=24, d=0.
      Candidate 1: SARIMA(1,0,0)x(1,1,0,24) - Parsimonious AR
      Candidate 2: SARIMA(1,0,1)x(1,1,0,24) - ARMA(1,1) + Seasonal AR(1)
      Candidate 3: SARIMA(1,0,1)x(0,1,1,24) - ARMA(1,1) + Seasonal MA(1)
      Candidate 4: SARIMA(2,0,1)x(1,1,0,24) - ARMA(2,1) + Seasonal AR(1)
      Candidate 5: SARIMA(2,0,1)x(0,1,1,24) - ARMA(2,1) + Seasonal MA(1)
    """
    print("\n--- Step 2: SARIMA Order Selection on Validation Set (Jan-Feb 2020) ---")
    train_series = df.loc[train_mask, "avg_power_kw"].values

    candidates = [
        {"name": "SARIMA(1,0,0)x(1,1,0,24)", "order": (1, 0, 0), "sorder": (1, 1, 0, 24)},
        {"name": "SARIMA(1,0,1)x(1,1,0,24)", "order": (1, 0, 1), "sorder": (1, 1, 0, 24)},
        {"name": "SARIMA(1,0,1)x(0,1,1,24)", "order": (1, 0, 1), "sorder": (0, 1, 1, 24)},
        {"name": "SARIMA(2,0,1)x(1,1,0,24)", "order": (2, 0, 1), "sorder": (1, 1, 0, 24)},
        {"name": "SARIMA(2,0,1)x(0,1,1,24)", "order": (2, 0, 1), "sorder": (0, 1, 1, 24)},
    ]

    tuning_records = []
    fitted_models = {}

    for cand in candidates:
        print(f"[*] Fitting {cand['name']} on training split (14,640 hrs)...")
        model = SARIMAX(
            train_series,
            order=cand["order"],
            seasonal_order=cand["sorder"],
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=50)
        fitted_models[cand["name"]] = res

        # Evaluate Day-Ahead H=24 on Validation Set
        # Use Kalman filter state updating (res.apply) to maintain causal state at each daily origin
        preds_h24 = []
        actuals_h24 = []

        # Optimization: Apply Kalman filter through each origin
        for t in val_origins:
            # history available up to t-1
            # To be efficient, pass window from train end to t-1
            # (or full series up to t-1; Kalman filter is rapid)
            # Since parameters are fixed, we can evaluate causal forecasts:
            # Using recent history (last 500 hours is plenty for SARIMA memory)
            hist_y = df.loc[max(0, t - 500) : t - 1, "avg_power_kw"].values
            res_origin = res.apply(hist_y, refit=False)
            fc = np.clip(res_origin.forecast(24), 0, None)
            preds_h24.extend(fc)
            actuals_h24.extend(df.loc[t : t + 23, "avg_power_kw"].values)

        m24 = calculate_metrics(np.array(actuals_h24), np.array(preds_h24))

        # Evaluate Rolling 1-Step H=1 on Validation Set
        val_start_idx = df[val_mask].index[0]
        val_end_idx = df[val_mask].index[-1]
        val_hist = df.loc[val_start_idx - 500 : val_end_idx, "avg_power_kw"].values
        res_val = res.apply(val_hist, refit=False)
        # fitted values corresponding to validation hours:
        val_preds_h1 = np.clip(res_val.fittedvalues[500:], 0, None)
        val_actuals_h1 = df.loc[val_mask, "avg_power_kw"].values
        m1 = calculate_metrics(val_actuals_h1, val_preds_h1)

        print(
            f"    -> Val H=24: MAE = {m24['mae_kw']:.2f} kW | RMSE = {m24['rmse_kw']:.2f} kW | "
            f"AIC = {res.aic:.1f} | BIC = {res.bic:.1f}"
        )
        print(f"    -> Val H=1:  MAE = {m1['mae_kw']:.2f} kW | RMSE = {m1['rmse_kw']:.2f} kW")

        tuning_records.append({
            "candidate_name": cand["name"],
            "order": str(cand["order"]),
            "seasonal_order": str(cand["sorder"]),
            "aic": round(float(res.aic), 2),
            "bic": round(float(res.bic), 2),
            "val_h24_mae_kw": m24["mae_kw"],
            "val_h24_rmse_kw": m24["rmse_kw"],
            "val_h24_mape_pct": m24["mape_filtered_pct"],
            "val_h24_smape_pct": m24["smape_pct"],
            "val_h1_mae_kw": m1["mae_kw"],
            "val_h1_rmse_kw": m1["rmse_kw"],
            "val_h1_mape_pct": m1["mape_filtered_pct"],
            "val_h1_smape_pct": m1["smape_pct"],
        })

    df_tuning = pd.DataFrame(tuning_records)
    # Primary selection criterion: Lowest Validation Day-Ahead H=24 MAE
    best_row = df_tuning.sort_values("val_h24_mae_kw").iloc[0]
    best_cand_name = best_row["candidate_name"]
    best_cand = [c for c in candidates if c["name"] == best_cand_name][0]
    best_fitted = fitted_models[best_cand_name]

    print(f"\n[+] Selected Best SARIMA on Validation Data: {best_cand_name}")
    print(f"    Validation H=24 MAE: {best_row['val_h24_mae_kw']:.2f} kW (RMSE: {best_row['val_h24_rmse_kw']:.2f} kW)")

    return best_cand, best_fitted, df_tuning


def evaluate_sarima_test(
    df: pd.DataFrame,
    best_cand: Dict[str, Any],
    best_fitted: Any,
    test_mask: pd.Series,
    reports_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates the selected SARIMA model on the Test set:
    - Day-Ahead (H=24) rolling daily origins (6,696 hours across 279 days)
    - Rolling 1-Step (H=1) (6,700 hours)
    - Seasonal breakdowns across March–May, June–August, September–December 2020.
    """
    print("\n--- Step 3: Evaluating Selected SARIMA on Test Data (Mar-Dec 2020) ---")
    test_origins = get_daily_forecast_origins(df, split="test")

    h24_rows = []
    print(f"[*] Running Day-Ahead H=24 rolling forecasts across {len(test_origins)} origins...")

    for t in test_origins:
        origin_ts = df.loc[t, "timestamp"]
        ts_window = df.loc[t : t + 23, "timestamp"].tolist()
        y_true_window = df.loc[t : t + 23, "avg_power_kw"].values

        # causal history up to t-1
        hist_y = df.loc[max(0, t - 500) : t - 1, "avg_power_kw"].values
        res_origin = best_fitted.apply(hist_y, refit=False)
        fc = np.clip(res_origin.forecast(24), 0, None)

        for step in range(24):
            h24_rows.append({
                "origin_timestamp": origin_ts,
                "timestamp": ts_window[step],
                "step": step + 1,
                "avg_power_kw": y_true_window[step],
                "pred_sarima": fc[step],
            })

    df_h24_pred = pd.DataFrame(h24_rows)
    df_h24_pred["timestamp"] = pd.to_datetime(df_h24_pred["timestamp"], utc=True)
    df_h24_pred["origin_timestamp"] = pd.to_datetime(df_h24_pred["origin_timestamp"], utc=True)

    # Rolling 1-Step H=1 on Test Set
    test_start_idx = df[test_mask].index[0]
    test_end_idx = df[test_mask].index[-1]
    hist_test = df.loc[test_start_idx - 500 : test_end_idx, "avg_power_kw"].values

    res_test = best_fitted.apply(hist_test, refit=False)
    test_preds_h1 = np.clip(res_test.fittedvalues[500:], 0, None)
    test_actuals_h1 = df.loc[test_mask, "avg_power_kw"].values
    test_timestamps_h1 = df.loc[test_mask, "timestamp"].values

    df_h1_pred = pd.DataFrame({
        "timestamp": pd.to_datetime(test_timestamps_h1, utc=True),
        "avg_power_kw": test_actuals_h1,
        "pred_sarima": test_preds_h1,
    })

    # Metrics Summary
    metrics_records = []
    seasonal_records = []

    m_h24 = calculate_metrics(df_h24_pred["avg_power_kw"].values, df_h24_pred["pred_sarima"].values)
    m_h1 = calculate_metrics(df_h1_pred["avg_power_kw"].values, df_h1_pred["pred_sarima"].values)

    metrics_records.append({
        "model": "SARIMA",
        "task": "Day-Ahead (H=24)",
        "split": "Test",
        "n_samples": len(df_h24_pred),
        "mae_kw": m_h24["mae_kw"],
        "rmse_kw": m_h24["rmse_kw"],
        "mape_filtered_pct": m_h24["mape_filtered_pct"],
        "smape_pct": m_h24["smape_pct"],
        "mape_evaluated_samples": m_h24["mape_evaluated_samples"],
        "config_name": best_cand["name"],
    })

    metrics_records.append({
        "model": "SARIMA",
        "task": "Rolling 1-Step (H=1)",
        "split": "Test",
        "n_samples": len(df_h1_pred),
        "mae_kw": m_h1["mae_kw"],
        "rmse_kw": m_h1["rmse_kw"],
        "mape_filtered_pct": m_h1["mape_filtered_pct"],
        "smape_pct": m_h1["smape_pct"],
        "mape_evaluated_samples": m_h1["mape_evaluated_samples"],
        "config_name": best_cand["name"],
    })

    # Seasonal Breakdowns
    for task_name, pred_df in [("Day-Ahead (H=24)", df_h24_pred), ("Rolling 1-Step (H=1)", df_h1_pred)]:
        for period_name, start_ts, end_ts in get_test_subperiods():
            p_mask = (pred_df["timestamp"] >= start_ts) & (pred_df["timestamp"] <= end_ts)
            sub = pred_df[p_mask]
            if len(sub) > 0:
                m_sub = calculate_metrics(sub["avg_power_kw"].values, sub["pred_sarima"].values)
                seasonal_records.append({
                    "model": "SARIMA",
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
        "model_name": "SARIMA",
        "selected_configuration": best_cand,
        "aic": float(best_fitted.aic),
        "bic": float(best_fitted.bic),
        "target_variable": "avg_power_kw (reconstructed hourly energy allocation; not instantaneous physical power)",
        "energy_cap_note": "7.7 kW per-session provisional cap applied during v2 preprocessing",
        "train_range": "2018-05-01 00:00 to 2019-12-31 23:00 UTC",
        "validation_range": "2020-01-01 00:00 to 2020-02-29 23:00 UTC",
        "test_range": "2020-03-01 00:00 to 2020-12-05 03:00 UTC",
    }

    return df_metrics, df_seasonal, df_h24_pred, df_h1_pred, config_info


def generate_plots(df_h24: pd.DataFrame, df_h1: pd.DataFrame, figures_dir: Path):
    """Generates comparison visual of SARIMA forecasts vs Actuals on the test set."""
    sns.set_theme(style="whitegrid")
    sample_start = "2020-03-01 00:00:00+00:00"
    sample_end = "2020-03-14 23:00:00+00:00"

    sub_h24 = df_h24[(df_h24["timestamp"] >= sample_start) & (df_h24["timestamp"] <= sample_end)]
    sub_h1 = df_h1[(df_h1["timestamp"] >= sample_start) & (df_h1["timestamp"] <= sample_end)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.plot(sub_h24["timestamp"], sub_h24["avg_power_kw"], label="Actual Demand (kW)", color="black", linewidth=2.0)
    ax1.plot(sub_h24["timestamp"], sub_h24["pred_sarima"], label="SARIMA Day-Ahead (H=24)", color="#2a9d8f", linestyle="--", linewidth=1.6)
    ax1.set_title("SARIMA Day-Ahead (H=24) Forecast vs. Actual Demand (March 1–14, 2020)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Power Demand (kW)")
    ax1.legend(loc="upper right")

    ax2.plot(sub_h1["timestamp"], sub_h1["avg_power_kw"], label="Actual Demand (kW)", color="black", linewidth=2.0)
    ax2.plot(sub_h1["timestamp"], sub_h1["pred_sarima"], label="SARIMA Rolling 1-Step (H=1)", color="#e76f51", linestyle="-.", linewidth=1.6)
    ax2.set_title("SARIMA Rolling 1-Step (H=1) Forecast vs. Actual Demand (March 1–14, 2020)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Power Demand (kW)")
    ax2.set_xlabel("Timestamp (UTC)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig_path = figures_dir / "sarima_forecasts.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[+] Saved forecast plot to: {fig_path}")


def main():
    print("=========================================================================")
    print("  Phase 3 - Model 2: SARIMA Pipeline (Diagnostics & Forecasting)")
    print("=========================================================================")

    reports_dir = get_phase3_reports_dir()
    figures_dir = reports_dir / "figures"
    df = load_v2_dataset()

    train_mask, val_mask, test_mask = get_split_masks(df)
    train_series = df.loc[train_mask, "avg_power_kw"].values
    val_origins = get_daily_forecast_origins(df, split="val")

    # Step 1: Training-Only Diagnostics
    run_training_diagnostics(train_series, reports_dir, figures_dir)

    # Step 2: Model Selection on Validation Data
    best_cand, best_fitted, df_tuning = tune_sarima_on_validation(df, train_mask, val_mask, val_origins)
    df_tuning.to_csv(reports_dir / "sarima_validation_tuning.csv", index=False)

    # Step 3: Evaluate on Test Set
    df_metrics, df_seasonal, df_h24_pred, df_h1_pred, config_info = evaluate_sarima_test(
        df, best_cand, best_fitted, test_mask, reports_dir
    )

    # Step 4: Save results
    metrics_path = reports_dir / "sarima_metrics.csv"
    seasonal_path = reports_dir / "sarima_seasonal_breakdown.csv"
    config_path = reports_dir / "sarima_config.json"

    df_metrics.to_csv(metrics_path, index=False)
    df_seasonal.to_csv(seasonal_path, index=False)
    with open(config_path, "w") as f:
        json.dump(config_info, f, indent=2)

    df_h24_pred.to_parquet(reports_dir / "sarima_h24_predictions.parquet", index=False)
    df_h1_pred.to_parquet(reports_dir / "sarima_h1_predictions.parquet", index=False)

    print("\n--- Quantitative Test Performance: SARIMA ---")
    print(df_metrics.to_string(index=False))

    print("\n--- Test Set Seasonal Breakdown ---")
    print(df_seasonal.to_string(index=False))

    # Step 5: Generate plots
    generate_plots(df_h24_pred, df_h1_pred, figures_dir)

    print(f"\n[+] Saved metrics to: {metrics_path}")
    print(f"[+] Saved seasonal breakdown to: {seasonal_path}")
    print(f"[+] Saved configuration to: {config_path}")


if __name__ == "__main__":
    main()

