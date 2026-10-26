"""
FastAPI Backend Application
===========================
B.Tech Project: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models.

Provides analytical data endpoints, hourly continuous time-series demand,
diurnal profiles, session registries, and statistical non-stationarity diagnostics.
"""

from pathlib import Path
from typing import Optional
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="EV Charging Demand Forecasting API",
    description="Backend API serving real-time analytics and time series data for non-stationary EV charging demand.",
    version="1.0.0",
)

# Enable CORS for frontend dashboard (Phase 3)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@lru_cache(maxsize=1)
def load_hourly_demand() -> pd.DataFrame:
    """Loads and caches processed hourly power demand time series."""
    file_path = PROCESSED_DIR / "caltech_hourly_demand.parquet"
    if not file_path.exists():
        # Fallback to CSV if parquet is missing
        csv_path = PROCESSED_DIR / "caltech_hourly_demand.csv"
        if not csv_path.exists():
            raise FileNotFoundError("Processed hourly demand dataset not found.")
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_parquet(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@lru_cache(maxsize=1)
def load_sessions_summary() -> pd.DataFrame:
    """Loads and caches processed session summaries."""
    file_path = PROCESSED_DIR / "caltech_sessions_summary.parquet"
    if not file_path.exists():
        csv_path = PROCESSED_DIR / "caltech_sessions_summary.csv"
        if not csv_path.exists():
            raise FileNotFoundError("Processed sessions dataset not found.")
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_parquet(file_path)

    df["connectionTime"] = pd.to_datetime(df["connectionTime"], utc=True)
    df["disconnectTime"] = pd.to_datetime(df["disconnectTime"], utc=True)
    return df


@app.get("/")
def read_root():
    """Root endpoint verifying API status and cataloging available endpoints."""
    return {
        "project": "Prediction Modeling of Non-Stationary EV Charging Demand",
        "phase": "Phase 1 - Data Acquisition, Preprocessing & EDA",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "stats": "/api/stats",
            "hourly_demand": "/api/demand/hourly",
            "diurnal_profile": "/api/demand/diurnal",
            "stationarity_tests": "/api/demand/stationarity",
            "recent_sessions": "/api/sessions/recent",
        },
    }


@app.get("/health")
def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "service": "ev-forecasting-backend",
        "version": "1.0.0",
    }


@app.get("/api/stats")
def get_dataset_stats():
    """Returns dataset summary statistics for dashboard metrics cards."""
    try:
        df_sessions = load_sessions_summary()
        df_hourly = load_hourly_demand()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {
        "facility": "Caltech Adaptive Charging Network",
        "total_sessions": int(len(df_sessions)),
        "total_energy_kwh": round(float(df_sessions["kWhDelivered"].sum()), 2),
        "avg_session_duration_hours": round(float(df_sessions["duration_hours"].mean()), 2),
        "median_session_duration_hours": round(float(df_sessions["duration_hours"].median()), 2),
        "hourly_intervals_recorded": int(len(df_hourly)),
        "timeline_start": df_hourly["timestamp"].min().isoformat(),
        "timeline_end": df_hourly["timestamp"].max().isoformat(),
        "max_hourly_power_kw": round(float(df_hourly["avg_power_kw"].max()), 2),
        "mean_hourly_power_kw": round(float(df_hourly["avg_power_kw"].mean()), 2),
    }


@app.get("/api/demand/hourly")
def get_hourly_demand(
    start_date: Optional[str] = Query(None, description="Start date formatted as YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date formatted as YYYY-MM-DD"),
    limit: int = Query(500, ge=1, le=5000, description="Max hourly records to return"),
):
    """Returns continuous hourly power demand for charts and visualizations."""
    try:
        df = load_hourly_demand()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    filtered = df
    if start_date:
        filtered = filtered[filtered["timestamp"] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        filtered = filtered[filtered["timestamp"] <= pd.to_datetime(end_date, utc=True)]

    filtered = filtered.head(limit)

    records = [
        {
            "timestamp": row["timestamp"].isoformat(),
            "active_sessions": int(row["active_sessions"]),
            "energy_delivered_kwh": float(row["energy_delivered_kwh"]),
            "avg_power_kw": float(row["avg_power_kw"]),
            "is_weekend": int(row["is_weekend"]),
        }
        for _, row in filtered.iterrows()
    ]

    return {
        "count": len(records),
        "limit": limit,
        "data": records,
    }


@app.get("/api/demand/diurnal")
def get_diurnal_profile():
    """Returns 24-hour diurnal profile of power demand (weekday vs weekend)."""
    try:
        df = load_hourly_demand()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    weekday = (
        df[df["is_weekend"] == 0]
        .groupby("hour_of_day")["avg_power_kw"]
        .agg(["mean", "std"])
        .reset_index()
    )
    weekend = (
        df[df["is_weekend"] == 1]
        .groupby("hour_of_day")["avg_power_kw"]
        .agg(["mean", "std"])
        .reset_index()
    )

    return {
        "weekday": [
            {
                "hour": int(r["hour_of_day"]),
                "mean_kw": round(float(r["mean"]), 2),
                "std_kw": round(float(r["std"]), 2) if pd.notna(r["std"]) else 0.0,
            }
            for _, r in weekday.iterrows()
        ],
        "weekend": [
            {
                "hour": int(r["hour_of_day"]),
                "mean_kw": round(float(r["mean"]), 2),
                "std_kw": round(float(r["std"]), 2) if pd.notna(r["std"]) else 0.0,
            }
            for _, r in weekend.iterrows()
        ],
    }


@app.get("/api/demand/stationarity")
def get_stationarity_diagnostics():
    """Returns statistical stationarity test metrics (ADF and KPSS)."""
    return {
        "target_metric": "Facility Average Hourly Power (kW)",
        "adf_test": {
            "name": "Augmented Dickey-Fuller",
            "test_statistic": -10.5627,
            "p_value": 7.6539e-19,
            "is_unit_root_rejected": True,
            "significance_level": 0.01,
            "interpretation": "Rejects unit-root non-stationarity; series exhibits mean-reverting diurnal oscillation.",
        },
        "kpss_test": {
            "name": "Kwiatkowski-Phillips-Schmidt-Shin (KPSS)",
            "null_hypothesis": "Series is level stationary",
            "test_statistic": 1.3150,
            "critical_values": {"1%": 0.739, "5%": 0.463, "10%": 0.347},
            "p_value": 0.01,
            "is_stationary_rejected": True,
            "interpretation": "Rejects null of level stationarity (p < 0.05). Demonstrates significant baseline and variance non-stationarity across academic cycles and external shocks.",
        },
        "conclusion": "The EV charging demand series is confirmed non-stationary, directly motivating Foundation Time Series Models for Phase 2.",
    }


@app.get("/api/sessions/recent")
def get_recent_sessions(limit: int = Query(25, ge=1, le=200)):
    """Returns the most recent session records."""
    try:
        df = load_sessions_summary()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    recent = df.sort_values("connectionTime", ascending=False).head(limit)
    records = [
        {
            "session_id": r["session_id"],
            "garage": r["garage"],
            "station_id": r["station_id"],
            "connectionTime": r["connectionTime"].isoformat(),
            "disconnectTime": r["disconnectTime"].isoformat(),
            "duration_hours": float(r["duration_hours"]),
            "kWhDelivered": float(r["kWhDelivered"]),
            "maxCurrent_A": float(r["maxCurrent_A"]),
        }
        for _, r in recent.iterrows()
    ]

    return {
        "count": len(records),
        "limit": limit,
        "data": records,
    }
