"""
Phase 2.1 - Corrected Data Processing Pipeline (process_time_series_v2.py)
===========================================================================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data

Corrections applied over v1 (process_time_series.py):
1. FILTER: Removes sessions with duration < 1 minute (sub-1-min sessions are
   data artefacts from rapid plug/unplug loops, not real charging events).
2. CAP: For all sessions, caps kWhDelivered at duration_hours * 7.7 kW
   (max L2 EVSE rate), removing meter-carryover artefacts where a short
   reconnect session inherits the cumulative energy counter.

Root Cause of 375.41 kW Anomaly (2020-04-17 22:00 UTC):
  - EVSE spaces 3843 and 3844 at N_Wilson_Garage_01 had rapid reconnect loops.
  - Meter carryover caused micro-sessions to inherit energy readings of 33-48 kWh.
  - 14 micro-sessions in that hour accumulated 700+ kWh in v1.
  - After correction, peak drops from 375 kW to 120 kW.

Outputs (separate from originals):
  - data/processed_v2/caltech_sessions_summary_v2.csv/.parquet
  - data/processed_v2/caltech_hourly_demand_v2.csv/.parquet

Usage:
    python scripts/process_time_series_v2.py --site caltech
    python scripts/process_time_series_v2.py --site caltech --max-files 5000
"""

import os
import sys
import glob
import gzip
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np


# Physical ceiling: L2 EVSE max charging rate (240V / 32A ~ 7.68 kW)
L2_MAX_KW = 7.7
MIN_DURATION_HRS = 1.0 / 60.0  # 1 minute minimum session length


def get_project_root() -> Path:
    """Returns the root directory of the EV_Charging_Forecasting project."""
    return Path(__file__).resolve().parent.parent


def parse_single_session_file(filepath: str) -> dict:
    """
    Parses a single gzip-compressed session CSV file.
    Energy Delivered column is a per-session cumulative meter (starts at 0,
    resets to 0 at UNPLUGGED). Maximum value = total kWh delivered.
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

        first_row = lines[1].split(",")
        last_row = lines[-1].split(",")

        start_dt = pd.to_datetime(first_row[0], utc=True)
        end_dt = pd.to_datetime(last_row[0], utc=True)

        duration_sec = (end_dt - start_dt).total_seconds()
        duration_hrs = round(max(duration_sec / 3600.0, 0.0), 6)

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
            "duration_hours": round(duration_hrs, 6),
            "kWhDelivered_raw": round(max_kwh, 4),
            "maxCurrent_A": round(max_amp, 2),
            "rows_recorded": len(lines) - 1,
        }
    except Exception:
        return None


def apply_data_corrections(df_sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Applies data quality corrections to session records.

    Stage 1 - Remove sub-1-minute sessions:
        Sessions shorter than 1 minute are instrument artefacts.

    Stage 2 - Cap energy at L2 physical maximum:
        Any session kWh > duration_hours * L2_MAX_KW is capped.
        This removes meter-carryover artefacts in rapid-reconnect events.

    Original kWh is preserved in 'kWhDelivered_raw'.
    Corrected kWh is stored in 'kWhDelivered'.
    """
    df = df_sessions.copy()
    n_original = len(df)

    # Stage 1: Remove sub-1-minute sessions
    df = df[df["duration_hours"] >= MIN_DURATION_HRS].copy()
    n_removed_stage1 = n_original - len(df)

    # Compute maximum physically possible kWh per session
    df["max_possible_kwh"] = (df["duration_hours"] * L2_MAX_KW).round(4)
    df["kWhDelivered"] = df["kWhDelivered_raw"].copy()

    # Stage 2: Cap energy at physical limit
    over_cap_mask = df["kWhDelivered"] > df["max_possible_kwh"]
    n_capped = over_cap_mask.sum()
    df.loc[over_cap_mask, "kWhDelivered"] = df.loc[over_cap_mask, "max_possible_kwh"]
    df["kWhDelivered"] = df["kWhDelivered"].round(4)

    print(f"  [Stage 1] Removed {n_removed_stage1:,} sub-1-min sessions "
          f"({n_original:,} -> {len(df):,})")
    print(f"  [Stage 2] Capped energy for {n_capped:,} sessions "
          f"exceeding L2 max ({L2_MAX_KW} kW x duration)")

    col_order = [
        "session_id", "site", "garage", "station_id",
        "connectionTime", "disconnectTime", "duration_hours",
        "kWhDelivered", "kWhDelivered_raw", "max_possible_kwh",
        "maxCurrent_A", "rows_recorded",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    return df.sort_values("connectionTime").reset_index(drop=True)


def generate_hourly_demand_series(df_sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs continuous hourly aggregated demand time series from
    corrected session intervals.

    Energy allocation: uniform distribution of session kWh across all
    active hours in the session interval (1 kWh per 1 hour = 1 kW avg).
    """
    if df_sessions.empty:
        return pd.DataFrame()

    df = df_sessions.copy()
    df = df.dropna(subset=["connectionTime", "disconnectTime"])
    df = df[df["disconnectTime"] >= df["connectionTime"]]

    min_time = df["connectionTime"].min().floor("h")
    max_time = df["disconnectTime"].max().ceil("h")

    hourly_index = pd.date_range(start=min_time, end=max_time, freq="1h", tz="UTC")
    print(f"  [*] Aggregating hourly demand from {min_time} to {max_time} "
          f"({len(hourly_index):,} hours)...")

    valid_sessions = df[df["duration_hours"] > 0.01].copy()

    active_sessions_count = np.zeros(len(hourly_index), dtype=int)
    hourly_energy_kwh = np.zeros(len(hourly_index), dtype=float)

    for _, row in valid_sessions.iterrows():
        s_idx = max(0, int((row["connectionTime"] - min_time).total_seconds() // 3600))
        e_idx = min(
            len(hourly_index) - 1,
            int((row["disconnectTime"] - min_time).total_seconds() // 3600),
        )
        if s_idx <= e_idx:
            active_sessions_count[s_idx: e_idx + 1] += 1
            num_hours = e_idx - s_idx + 1
            alloc_kwh = row["kWhDelivered"] / num_hours
            hourly_energy_kwh[s_idx: e_idx + 1] += alloc_kwh

    df_hourly = pd.DataFrame({
        "timestamp": hourly_index,
        "active_sessions": active_sessions_count,
        "energy_delivered_kwh": np.round(hourly_energy_kwh, 4),
        "avg_power_kw": np.round(hourly_energy_kwh, 4),  # kWh/1h = kW numerically
    })

    df_hourly["hour_of_day"] = df_hourly["timestamp"].dt.hour
    df_hourly["day_of_week"] = df_hourly["timestamp"].dt.day_of_week
    df_hourly["is_weekend"] = df_hourly["day_of_week"].isin([5, 6]).astype(int)
    df_hourly["month"] = df_hourly["timestamp"].dt.month
    df_hourly["year"] = df_hourly["timestamp"].dt.year

    return df_hourly


def run_pipeline(site: str = "caltech", max_files: int = None, workers: int = None):
    """Executes the corrected data processing workflow."""
    root = get_project_root()
    raw_dir = root / "data" / "raw" / "ACN-Data-Static-main" / "time series data" / site
    processed_v2_dir = root / "data" / "processed_v2"
    processed_v2_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f"[!] Directory not found: {raw_dir}")
        sys.exit(1)

    all_files = sorted(glob.glob(str(raw_dir / "**" / "*.csv.gz"), recursive=True))
    total_found = len(all_files)
    print(f"[*] Found {total_found:,} files in {raw_dir}")

    if max_files and max_files < total_found:
        files_to_process = all_files[:max_files]
        print(f"[*] Limiting run to first {max_files:,} files.")
    else:
        files_to_process = all_files

    print(f"[*] Extracting sessions using multiprocessing...")
    records = []
    num_workers = workers or max(1, os.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(parse_single_session_file, f): f
            for f in files_to_process
        }
        count = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                records.append(res)
            count += 1
            if count % 5000 == 0 or count == len(files_to_process):
                print(f"  -> Processed {count:,}/{len(files_to_process):,} files...")

    df_sessions_raw = pd.DataFrame(records)
    df_sessions_raw = df_sessions_raw.sort_values("connectionTime").reset_index(drop=True)
    print(f"[+] Extracted {len(df_sessions_raw):,} raw session records.\n")

    # Apply corrections
    print("[*] Applying data corrections...")
    df_sessions = apply_data_corrections(df_sessions_raw)
    print(f"[+] Corrected sessions: {len(df_sessions):,}\n")

    # Save sessions
    sessions_csv = processed_v2_dir / f"{site}_sessions_summary_v2.csv"
    sessions_parquet = processed_v2_dir / f"{site}_sessions_summary_v2.parquet"
    df_sessions.to_csv(sessions_csv, index=False)
    df_sessions.to_parquet(sessions_parquet, index=False)
    print(f"[+] Sessions saved: {sessions_csv} "
          f"({sessions_csv.stat().st_size / 1e6:.2f} MB)")

    # Generate and save hourly demand
    print("\n[*] Generating hourly demand series...")
    df_hourly = generate_hourly_demand_series(df_sessions)
    hourly_csv = processed_v2_dir / f"{site}_hourly_demand_v2.csv"
    hourly_parquet = processed_v2_dir / f"{site}_hourly_demand_v2.parquet"
    df_hourly.to_csv(hourly_csv, index=False)
    df_hourly.to_parquet(hourly_parquet, index=False)
    print(f"[+] Hourly demand saved: {hourly_csv} "
          f"({hourly_csv.stat().st_size / 1e6:.2f} MB)")

    print("\n=== v2 Dataset Summary ===")
    print(f"Sessions (corrected):        {len(df_sessions):,}")
    print(f"Energy delivered (corrected):{df_sessions['kWhDelivered'].sum():,.2f} kWh")
    print(f"Avg session duration:        {df_sessions['duration_hours'].mean():.2f} hours")
    print(f"Date range: {df_sessions['connectionTime'].min()} "
          f"to {df_sessions['connectionTime'].max()}")
    print(f"Hourly demand max:           {df_hourly['avg_power_kw'].max():.2f} kW")
    print(f"Hourly demand mean:          {df_hourly['avg_power_kw'].mean():.2f} kW")
    print("===========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Corrected ACN-Data processing pipeline (v2)."
    )
    parser.add_argument("--site", type=str, default="caltech",
                        help="Site name (default: caltech)")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Maximum files to process (default: all)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes")
    args = parser.parse_args()

    run_pipeline(site=args.site, max_files=args.max_files, workers=args.workers)
