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

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
PROCESSED_V2_DIR = PROJECT_ROOT / "data" / "processed_v2"


@lru_cache(maxsize=1)
def load_hourly_demand_v2() -> pd.DataFrame:
    """Loads and caches corrected hourly power demand time series (v2)."""
    file_path = PROCESSED_V2_DIR / "caltech_hourly_demand_v2.parquet"
    if not file_path.exists():
        csv_path = PROCESSED_V2_DIR / "caltech_hourly_demand_v2.csv"
        if not csv_path.exists():
            raise FileNotFoundError("Corrected v2 hourly demand dataset not found.")
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_parquet(file_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@lru_cache(maxsize=1)
def get_sarima_forecaster():
    """
    Fits and caches in memory the validated SARIMA (2,0,1)x(0,1,1,24) forecasting model.
    To avoid fitting repeatedly across API calls:
      1. Fits the model strictly on the permitted historical training series (pre-2020).
      2. Updates the Kalman state over the historical series up to the forecast origin
         using .apply(..., refit=False) without altering trained parameters.
      3. Caches the fitted forecaster instance in memory via @lru_cache for instant inference (~2ms).
    """
    df_v2 = load_hourly_demand_v2()
    df_v2 = df_v2.sort_values("timestamp").reset_index(drop=True)
    forecast_origin = df_v2["timestamp"].iloc[-1]

    # Strictly training split: pre-2020
    train_mask = df_v2["timestamp"] < "2020-01-01"
    train_series = df_v2.loc[train_mask, "avg_power_kw"].values

    # Fit SARIMAX using validated Phase 3 configuration
    model = SARIMAX(
        train_series,
        order=(2, 0, 1),
        seasonal_order=(0, 1, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)

    # Update Kalman filter state up to latest observation origin without refitting
    full_series = df_v2["avg_power_kw"].values
    forecaster = res.apply(full_series, refit=False)

    return forecaster, forecast_origin


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
            "forecast": "/api/forecast?horizon=24",
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
    start_date: Optional[str] = Query(None, description="Start date formatted as YYYY-MM-DD or ISO timestamp"),
    end_date: Optional[str] = Query(None, description="End date formatted as YYYY-MM-DD or ISO timestamp"),
    limit: int = Query(500, ge=1, le=5000, description="Max hourly records to return"),
    recent: bool = Query(False, description="If True and start_date is omitted, returns the latest limit records up to end_date (or dataset end)"),
    version: Optional[str] = Query("v2", description="Dataset version: 'v1' or 'v2' (defaults to v2 corrected series)"),
):
    """Returns continuous hourly power demand for charts and visualizations."""
    try:
        if version == "v1":
            df = load_hourly_demand()
        else:
            try:
                df = load_hourly_demand_v2()
            except Exception:
                df = load_hourly_demand()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    filtered = df.sort_values("timestamp").reset_index(drop=True)
    has_start = isinstance(start_date, str) and bool(start_date.strip())
    has_end = isinstance(end_date, str) and bool(end_date.strip())

    if has_start:
        filtered = filtered[filtered["timestamp"] >= pd.to_datetime(start_date, utc=True)]
    if has_end:
        filtered = filtered[filtered["timestamp"] <= pd.to_datetime(end_date, utc=True)]

    if bool(recent) and not has_start:
        filtered = filtered.tail(limit)
    else:
        filtered = filtered.head(limit)

    records = [
        {
            "timestamp": row["timestamp"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "active_sessions": int(row["active_sessions"]) if "active_sessions" in row else 0,
            "energy_delivered_kwh": float(row["energy_delivered_kwh"]) if "energy_delivered_kwh" in row else 0.0,
            "avg_power_kw": float(row["avg_power_kw"]),
            "is_weekend": int(row["is_weekend"]) if "is_weekend" in row else 0,
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


@app.get("/api/forecast")
def get_forecast(
    horizon: int = Query(24, ge=1, le=168, description="Forecast horizon in hours (1-168)"),
):
    """
    Generates causal day-ahead or multi-step hourly EV charging demand forecasts
    using the validated SARIMA (2,0,1)x(0,1,1,24) pipeline on corrected v2 demand data.

    Returns predicted demand in kW for the requested horizon (1-168 hours) starting
    immediately after the latest available observation timestamp (forecast origin).
    """
    if horizon < 1 or horizon > 168:
        raise HTTPException(
            status_code=422,
            detail="Forecast horizon must be an integer between 1 and 168 hours.",
        )

    try:
        forecaster, origin_ts = get_sarima_forecaster()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Data dependency unavailable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecasting engine initialization error: {str(e)}")

    try:
        raw_preds = forecaster.forecast(steps=horizon)
        clipped_preds = np.clip(raw_preds, 0.0, None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast generation failed: {str(e)}")

    predictions = []
    for h, p in enumerate(clipped_preds, start=1):
        target_ts = origin_ts + pd.Timedelta(hours=h)
        val = round(float(p), 2)
        if np.isnan(val) or np.isinf(val):
            raise HTTPException(status_code=500, detail="Forecast calculation produced non-finite values.")
        predictions.append({
            "timestamp": target_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "predicted_demand_kw": val,
        })

    origin_str = origin_ts.strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "model": "SARIMA",
        "horizon": horizon,
        "unit": "kW",
        "forecast_origin": origin_str,
        "predictions": predictions,
    }
