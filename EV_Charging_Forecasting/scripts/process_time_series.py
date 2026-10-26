"""
Time Series Preprocessing & Feature Aggregation Pipeline
=========================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data

This script processes raw static session time-series files from ACN-Data:
1. Extracts session-level metadata (connectionTime, disconnectTime, duration_hours, kWhDelivered, maxCurrent, garage).
2. Computes facility-level hourly aggregated demand (active_sessions, total_energy_kwh, avg_power_kw, peak_power_kw).
3. Saves processed datasets into `data/processed/` for downstream modeling, EDA, and backend APIs.

Usage:
    python scripts/process_time_series.py --site caltech --max-files 5000
    python scripts/process_time_series.py --site caltech --all
"""

import os
import sys
import glob
import gzip
import argparse
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np


def get_project_root() -> Path:
    """Returns the root directory of the EV_Charging_Forecasting project."""
    return Path(__file__).resolve().parent.parent


def parse_single_session_file(filepath: str) -> dict:
    """
    Parses a single gzip-compressed CSV file and extracts key session metrics.
    Fast streaming line reader without loading entire DataFrames.
    """
    path_obj = Path(filepath)
    filename = path_obj.name
    garage = path_obj.parent.name
    site = path_obj.parent.parent.name

    try:
        with gzip.open(filepath, "rt", encoding="utf-8", errors="ignore") as gz:
            lines = [line.strip() for line in gz if line.strip()]

        if len(lines) < 2:
            return None

        # Row 0 is header, Row 1 is start of session, Row -1 is disconnect
        first_row = lines[1].split(",")
        last_row = lines[-1].split(",")

        start_time_str = first_row[0]
        end_time_str = last_row[0]

        start_dt = pd.to_datetime(start_time_str, utc=True)
        end_dt = pd.to_datetime(end_time_str, utc=True)

        duration_sec = (end_dt - start_dt).total_seconds()
        duration_hrs = round(max(duration_sec / 3600.0, 0.0), 4)

        max_kwh = 0.0
        max_amp = 0.0

        for row_str in lines[1:]:
            parts = row_str.split(",")
            try:
                if len(parts) > 1 and parts[1]:
                    amp = float(parts[1])
                    if amp > max_amp:
                        max_amp = amp
                if len(parts) > 5 and parts[5]:
                    kwh = float(parts[5])
                    if kwh > max_kwh:
                        max_kwh = kwh
            except (ValueError, IndexError):
                continue

        station_id = "-".join(filename.split("-")[:4]) if "-" in filename else "unknown"

        return {
            "session_id": filename.replace(".csv.gz", ""),
            "site": site,
            "garage": garage,
            "station_id": station_id,
            "connectionTime": start_dt,
            "disconnectTime": end_dt,
            "duration_hours": duration_hrs,
            "kWhDelivered": round(max_kwh, 4),
            "maxCurrent_A": round(max_amp, 2),
            "rows_recorded": len(lines) - 1,
        }
    except Exception as e:
        return None


def generate_hourly_demand_series(df_sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs continuous hourly aggregated demand time series from session intervals.
    Calculates active sessions, energy delivered per hour, and estimated average power (kW).
    """
    if df_sessions.empty:
        return pd.DataFrame()

    df = df_sessions.copy()
    df = df.dropna(subset=["connectionTime", "disconnectTime"])
    df = df[df["disconnectTime"] >= df["connectionTime"]]

    min_time = df["connectionTime"].min().floor("h")
    max_time = df["disconnectTime"].max().ceil("h")

    hourly_index = pd.date_range(start=min_time, end=max_time, freq="1h", tz="UTC")
    print(f"[*] Aggregating hourly demand from {min_time} to {max_time} ({len(hourly_index)} hours)...")

    # Filter out spurious zero-duration sessions
    valid_sessions = df[df["duration_hours"] > 0.01].copy()
    valid_sessions["avg_rate_kw"] = valid_sessions["kWhDelivered"] / valid_sessions["duration_hours"]

    # Pre-allocate hourly arrays
    active_sessions_count = np.zeros(len(hourly_index), dtype=int)
    hourly_energy_kwh = np.zeros(len(hourly_index), dtype=float)

    for _, row in valid_sessions.iterrows():
        s_idx = max(0, int((row["connectionTime"] - min_time).total_seconds() // 3600))
        e_idx = min(len(hourly_index) - 1, int((row["disconnectTime"] - min_time).total_seconds() // 3600))
        if s_idx <= e_idx:
            active_sessions_count[s_idx : e_idx + 1] += 1
            num_hours = (e_idx - s_idx + 1)
            alloc_kwh = row["kWhDelivered"] / num_hours
            hourly_energy_kwh[s_idx : e_idx + 1] += alloc_kwh

    df_hourly = pd.DataFrame(
        {
            "timestamp": hourly_index,
            "active_sessions": active_sessions_count,
            "energy_delivered_kwh": np.round(hourly_energy_kwh, 4),
            "avg_power_kw": np.round(hourly_energy_kwh, 4),  # Since 1 hour interval, kWh = avg kW
        }
    )

    # Feature engineering for non-stationary patterns
    df_hourly["hour_of_day"] = df_hourly["timestamp"].dt.hour
    df_hourly["day_of_week"] = df_hourly["timestamp"].dt.day_of_week
    df_hourly["is_weekend"] = df_hourly["day_of_week"].isin([5, 6]).astype(int)
    df_hourly["month"] = df_hourly["timestamp"].dt.month
    df_hourly["year"] = df_hourly["timestamp"].dt.year

    return df_hourly


def run_pipeline(site: str = "caltech", max_files: int = None, workers: int = None):
    """Executes the data processing workflow."""
    root = get_project_root()
    raw_dir = root / "data" / "raw" / "ACN-Data-Static-main" / "time series data" / site
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"[!] Directory not found: {raw_dir}")
        sys.exit(1)

    all_files = sorted(glob.glob(str(raw_dir / "**" / "*.csv.gz"), recursive=True))
    total_found = len(all_files)
    print(f"[*] Found {total_found:,} files in {raw_dir}")

    if max_files and max_files < total_found:
        files_to_process = all_files[:max_files]
        print(f"[*] Limiting run to first {max_files:,} files as requested.")
    else:
        files_to_process = all_files

    print(f"[*] Extracting sessions using multiprocessing...")
    records = []
    num_workers = workers or max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(parse_single_session_file, f): f for f in files_to_process}
        count = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                records.append(res)
            count += 1
            if count % 2000 == 0 or count == len(files_to_process):
                print(f"  -> Processed {count:,}/{len(files_to_process):,} files...")

    df_sessions = pd.DataFrame(records)
    print(f"[+] Successfully extracted {len(df_sessions):,} session records.")

    # Sort chronologically
    df_sessions = df_sessions.sort_values("connectionTime").reset_index(drop=True)

    # Save sessions summary
    sessions_csv = processed_dir / f"{site}_sessions_summary.csv"
    sessions_parquet = processed_dir / f"{site}_sessions_summary.parquet"
    df_sessions.to_csv(sessions_csv, index=False)
    df_sessions.to_parquet(sessions_parquet, index=False)
    print(f"[+] Saved sessions to: {sessions_csv} ({sessions_csv.stat().st_size / 1e6:.2f} MB)")
    print(f"[+] Saved parquet to: {sessions_parquet} ({sessions_parquet.stat().st_size / 1e6:.2f} MB)")

    # Generate hourly demand time series
    df_hourly = generate_hourly_demand_series(df_sessions)
    hourly_csv = processed_dir / f"{site}_hourly_demand.csv"
    hourly_parquet = processed_dir / f"{site}_hourly_demand.parquet"
    df_hourly.to_csv(hourly_csv, index=False)
    df_hourly.to_parquet(hourly_parquet, index=False)
    print(f"[+] Saved hourly demand to: {hourly_csv} ({hourly_csv.stat().st_size / 1e6:.2f} MB)")
    print(f"[+] Saved parquet to: {hourly_parquet} ({hourly_parquet.stat().st_size / 1e6:.2f} MB)")

    print("\n=== Dataset Summary Statistics ===")
    print(f"Total sessions: {len(df_sessions):,}")
    print(f"Total energy delivered: {df_sessions['kWhDelivered'].sum():,.2f} kWh")
    print(f"Average session duration: {df_sessions['duration_hours'].mean():.2f} hours")
    print(f"Date range: {df_sessions['connectionTime'].min()} to {df_sessions['connectionTime'].max()}")
    print("==================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ACN-Data static time-series into structured datasets.")
    parser.add_argument("--site", type=str, default="caltech", help="Site name (default: caltech)")
    parser.add_argument("--max-files", type=int, default=None, help="Maximum files to process (default: all)")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    args = parser.parse_args()

    run_pipeline(site=args.site, max_files=args.max_files, workers=args.workers)
