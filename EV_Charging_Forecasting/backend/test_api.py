"""
Focused Test Suite for FastAPI Backend and Forecasting Endpoint
==============================================================
Validates:
1. GET /api/forecast with default horizon (24 hours).
2. GET /api/forecast with boundary horizons: horizon=1 and horizon=168.
3. Validation of invalid horizons (0, -1, 169) returning HTTP 422.
4. Timestamp chronology, hourly progression, and origin alignment.
5. Finite non-negative predictions (no NaN, null, or inf values).
6. Non-regression of existing endpoints (/, /health, stats, hourly, diurnal, stationarity, recent).
"""

import math
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

# Ensure backend module can be imported
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    app,
    get_forecast,
    read_root,
    health_check,
    get_dataset_stats,
    get_hourly_demand,
    get_diurnal_profile,
    get_stationarity_diagnostics,
    get_recent_sessions,
)


def test_root_endpoint():
    """Verify root endpoint status and endpoints catalog."""
    data = read_root()
    assert data["status"] == "operational"
    assert "endpoints" in data
    assert "forecast" in data["endpoints"]
    assert data["endpoints"]["forecast"] == "/api/forecast?horizon=24"
    assert "health" in data["endpoints"]
    assert "stats" in data["endpoints"]


def test_health_endpoint():
    """Verify health check endpoint."""
    data = health_check()
    assert data["status"] == "healthy"
    assert data["service"] == "ev-forecasting-backend"


def test_existing_endpoints_regression():
    """Verify existing endpoints remain functional and return valid schemas."""
    # Stats
    stats = get_dataset_stats()
    assert stats["facility"] == "Caltech Adaptive Charging Network"
    assert stats["total_sessions"] > 0
    assert stats["total_energy_kwh"] > 0

    # Hourly demand
    hourly = get_hourly_demand(start_date=None, end_date=None, limit=5)
    assert hourly["count"] == 5
    assert len(hourly["data"]) == 5
    assert "timestamp" in hourly["data"][0]
    assert "avg_power_kw" in hourly["data"][0]

    # Diurnal profile
    diurnal = get_diurnal_profile()
    assert "weekday" in diurnal
    assert "weekend" in diurnal
    assert len(diurnal["weekday"]) == 24

    # Stationarity
    stationarity = get_stationarity_diagnostics()
    assert "adf_test" in stationarity
    assert "kpss_test" in stationarity

    # Recent sessions
    recent = get_recent_sessions(limit=5)
    assert recent["count"] == 5
    assert len(recent["data"]) == 5


def test_forecast_default_24h():
    """Verify forecast with default horizon (24 hours)."""
    data = get_forecast(horizon=24)
    assert data["model"] == "SARIMA"
    assert data["horizon"] == 24
    assert data["unit"] == "kW"
    assert data["forecast_origin"] == "2020-12-05T03:00:00Z"

    predictions = data["predictions"]
    assert len(predictions) == 24

    origin_dt = pd.to_datetime("2020-12-05T03:00:00Z")
    for i, p in enumerate(predictions, start=1):
        expected_dt = origin_dt + pd.Timedelta(hours=i)
        expected_ts = expected_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        assert p["timestamp"] == expected_ts, f"Mismatch at step {i}: expected {expected_ts}, got {p['timestamp']}"

        val = p["predicted_demand_kw"]
        assert isinstance(val, (int, float))
        assert not math.isnan(val)
        assert not math.isinf(val)
        assert val >= 0.0


def test_forecast_horizon_1():
    """Verify forecast with horizon=1 returns exactly 1 step."""
    data = get_forecast(horizon=1)
    assert data["model"] == "SARIMA"
    assert data["horizon"] == 1
    assert data["forecast_origin"] == "2020-12-05T03:00:00Z"

    predictions = data["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["timestamp"] == "2020-12-05T04:00:00Z"
    assert predictions[0]["predicted_demand_kw"] >= 0.0


def test_forecast_horizon_168():
    """Verify forecast with horizon=168 returns exactly 168 hours (7 days)."""
    data = get_forecast(horizon=168)
    assert data["model"] == "SARIMA"
    assert data["horizon"] == 168
    predictions = data["predictions"]
    assert len(predictions) == 168

    origin_dt = pd.to_datetime("2020-12-05T03:00:00Z")
    final_dt = origin_dt + pd.Timedelta(hours=168)
    assert predictions[-1]["timestamp"] == final_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    for p in predictions:
        val = p["predicted_demand_kw"]
        assert not math.isnan(val) and not math.isinf(val)
        assert val >= 0.0


def test_forecast_invalid_horizons():
    """Verify invalid horizons raise HTTP 422 HTTPException."""
    for invalid_h in [0, -1, 169]:
        try:
            get_forecast(horizon=invalid_h)
            assert False, f"Expected HTTPException(422) for horizon={invalid_h}, but call succeeded."
        except HTTPException as exc:
            assert exc.status_code == 422, f"Expected status_code 422 for horizon={invalid_h}, got {exc.status_code}"


def run_all_tests():
    """Executes all tests with latency measurement and clear reporting."""
    tests = [
        test_root_endpoint,
        test_health_endpoint,
        test_existing_endpoints_regression,
        test_forecast_default_24h,
        test_forecast_horizon_1,
        test_forecast_horizon_168,
        test_forecast_invalid_horizons,
    ]

    print(f"[*] Starting Focused Backend & Forecast Test Suite ({len(tests)} tests)...", flush=True)
    passed = 0
    failed = 0

    for test_fn in tests:
        test_name = test_fn.__name__
        t0 = time.time()
        try:
            test_fn()
            duration = time.time() - t0
            print(f" [PASS] {test_name:<36} ({duration:6.3f}s)", flush=True)
            passed += 1
        except Exception as e:
            duration = time.time() - t0
            print(f" [FAIL] {test_name:<36} ({duration:6.3f}s) -> Error: {e}", flush=True)
            failed += 1

    print("\n" + "=" * 50)
    print(f"Test Execution Summary: {passed} passed, {failed} failed.")
    print("=" * 50)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
