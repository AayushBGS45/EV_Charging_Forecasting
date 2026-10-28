"""
Phase 3 Evaluation Utilities and Shared Harness
================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models

IMPORTANT METHODOLOGICAL NOTES:
1. Target series 'avg_power_kw' represents hourly facility energy consumption (kWh/h = kW)
   reconstructed via uniform energy allocation across each session's connection duration.
   It does NOT represent instantaneous measured charging power, and intra-session charging
   rate fluctuations (e.g. CC/CV tapering, station curtailment) are smoothed out.
2. The 7.7 kW session power cap implemented in Phase 2.1 is provisional, based on standard
   Level 2 EVSE maximum delivery (32A @ 240V). While it successfully suppresses unphysical
   meter-carryover spikes (reducing the peak from 375.4 kW to 120.6 kW), station-specific
   hardware electrical ratings were not verified against primary engineering spec sheets.
3. Chronological partitioning is strictly enforced:
   - Training:   2018-05-01 00:00 to 2019-12-31 23:00 UTC (14,640 hours)
   - Validation: 2020-01-01 00:00 to 2020-02-29 23:00 UTC (1,440 hours)
   - Test:       2020-03-01 00:00 to 2020-12-05 03:00 UTC (6,700 hours)
4. Primary Task:
   - 24-hour Day-Ahead Forecasting: Daily rolling origin at 00:00 UTC (H=24).
     Forecasts t ... t+23 using only information available prior to origin t (t-1 and older).
5. Secondary Task:
   - 1-Step-Ahead Rolling Forecasting (H=1): Hourly rolling forecast using feedback from t-1.
6. Validation data is used strictly for model selection / hyperparameter tuning; test data
   is evaluated only once for the final unbiased comparison.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_phase3_reports_dir() -> Path:
    """Returns and ensures the reports/phase3 directory exists."""
    reports_dir = get_project_root() / "reports" / "phase3"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def load_v2_dataset() -> pd.DataFrame:
    """
    Loads the corrected continuous hourly demand dataset (v2).
    Enforces UTC timestamp datetime parsing and chronological ordering.
    """
    root = get_project_root()
    parquet_path = root / "data" / "processed_v2" / "caltech_hourly_demand_v2.parquet"
    csv_path = root / "data" / "processed_v2" / "caltech_hourly_demand_v2.csv"

    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"[!] Processed v2 dataset not found at {parquet_path}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, eps_threshold: float = 1.0) -> Dict[str, Any]:
    """
    Calculates standard evaluation metrics:
    - MAE: Mean Absolute Error (kW)
    - RMSE: Root Mean Squared Error (kW)
    - Filtered MAPE (%): Evaluated only where y_true >= eps_threshold (default 1.0 kW)
    - sMAPE (%): Symmetric Mean Absolute Percentage Error
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # MAE & RMSE
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Filtered MAPE
    mask = y_true >= eps_threshold
    num_eval_mape = int(np.sum(mask))
    if num_eval_mape > 0:
        mape_filtered = float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]) * 100.0)
    else:
        mape_filtered = float("nan")

    # sMAPE
    denominator = np.abs(y_true) + np.abs(y_pred) + 1e-6
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / denominator) * 100.0)

    return {
        "mae_kw": round(mae, 4),
        "rmse_kw": round(rmse, 4),
        "mape_filtered_pct": round(mape_filtered, 2) if not np.isnan(mape_filtered) else None,
        "smape_pct": round(smape, 2),
        "mape_evaluated_samples": num_eval_mape,
        "total_samples": len(y_true),
    }


def get_split_masks(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns boolean masks for train, val, test splits.
    Train:      < 2020-01-01 (14,640 hrs)
    Validation: 2020-01-01 to 2020-02-29 (1,440 hrs)
    Test:       >= 2020-03-01 to 2020-12-05 03:00 (6,700 hrs)
    """
    train_mask = df["timestamp"] < "2020-01-01"
    val_mask = (df["timestamp"] >= "2020-01-01") & (df["timestamp"] < "2020-03-01")
    test_mask = df["timestamp"] >= "2020-03-01"
    return train_mask, val_mask, test_mask


def get_test_subperiods() -> List[Tuple[str, str, str]]:
    """
    Defines test set sub-periods for seasonal breakdown:
    - March–May 2020: Lockdown onset & shock
    - June–August 2020: Summer low-demand plateau
    - September–December 2020: Fall partial resumption & variance increase
    """
    return [
        ("March–May 2020", "2020-03-01 00:00:00+00:00", "2020-05-31 23:00:00+00:00"),
        ("June–August 2020", "2020-06-01 00:00:00+00:00", "2020-08-31 23:00:00+00:00"),
        ("September–December 2020", "2020-09-01 00:00:00+00:00", "2020-12-05 03:00:00+00:00"),
    ]


def get_daily_forecast_origins(df: pd.DataFrame, split: str = "test") -> List[int]:
    """
    Returns integer row indices for daily rolling forecast origins (at 00:00 UTC)
    where a full 24-hour horizon [t, t+23] is available.
    """
    total_len = len(df)
    if split == "test":
        condition = (df["timestamp"] >= "2020-03-01") & (df["timestamp"].dt.hour == 0)
    elif split == "val":
        condition = (df["timestamp"] >= "2020-01-01") & (df["timestamp"] < "2020-03-01") & (df["timestamp"].dt.hour == 0)
    elif split == "all":
        condition = df["timestamp"].dt.hour == 0
    else:
        raise ValueError(f"Unknown split: {split}")

    candidate_indices = df[condition].index.tolist()
    valid_origins = [t for t in candidate_indices if t + 24 <= total_len]
    return valid_origins


def evaluate_seasonal_breakdown(
    y_true_series: pd.Series,
    y_pred_series: pd.Series,
    timestamps: pd.Series,
) -> List[Dict[str, Any]]:
    """
    Evaluates predictions broken down across the 3 test sub-periods.
    """
    records = []
    for period_name, start_ts, end_ts in get_test_subperiods():
        mask = (timestamps >= start_ts) & (timestamps <= end_ts)
        y_t = y_true_series[mask].values
        y_p = y_pred_series[mask].values
        if len(y_t) > 0:
            m = calculate_metrics(y_t, y_p)
            m["period"] = period_name
            m["period_start"] = start_ts
            m["period_end"] = end_ts
            records.append(m)
    return records

