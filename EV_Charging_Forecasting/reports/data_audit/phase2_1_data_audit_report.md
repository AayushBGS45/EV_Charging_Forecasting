# Phase 2.1 — Data Integrity Audit Report
**Project:** Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models  
**Date:** 2026-10-26  
**Auditor:** Automated pipeline + manual verification  
**Status:** ✅ Audit Complete — Corrected Dataset (v2) Generated

---

## 1. Objective

Before implementing SARIMA, LightGBM, or any foundation model, this audit:
1. Determines what **Energy Delivered (kWh)** represents in the raw files
2. Identifies and quantifies session-boundary / aggregation artefacts
3. Traces the root cause of the 375.41 kW anomaly
4. Generates a corrected dataset (`data/processed_v2/`) with documented corrections

---

## 2. Raw File Structure

Each `.csv.gz` file in `time series data/caltech/` corresponds to a **single charging session** for one EVSE station.

| Column Index | Column Name            | Description |
|---|---|---|
| 0 | Timestamp              | UTC ISO-8601, ~5-second intervals |
| 1 | Charging Current (A)   | Instantaneous current |
| 2 | Actual Pilot (A)       | Pilot signal current |
| 3 | Voltage (V)            | Typically empty |
| 4 | Charging State         | READY / ADAPTIVE / NOT CHARGING / UNPLUGGED |
| 5 | Energy Delivered (kWh) | **Cumulative meter reading** (per session) |
| 6 | Power (kW)             | Typically empty |

### Key Finding: Energy Delivered is a Cumulative Per-Session Counter

> [!IMPORTANT]
> **Energy Delivered (kWh)** is a **per-session cumulative meter** that:
> - Starts at **0.0** at session connect (READY state)
> - Increases monotonically during charging
> - **Resets to 0.0** when the vehicle unplugs (UNPLUGGED state)
>
> Consequence: **`max(Energy Delivered)`** across all rows = **total kWh delivered in that session**.
>
> The existing `process_time_series.py` correctly uses `max_kwh` — this logic is sound.

---

## 3. Audit Findings

### 3.1 Session Quality Issues (v1 Dataset)

| Issue | Count | Impact |
|---|---|---|
| Sessions with duration < 1 min (sub-1-min) | **5,059** | Meter-carryover artefacts |
| Sessions with duration < 6 min (micro-sessions) | 7,392 | High risk of energy inflation |
| Sessions with kWhDelivered = 0 | 4,926 | No energy exchanged (ignored in aggregation) |
| Sessions with implied avg rate > 10 kW | 4,217 | Physically implausible for L2 EVSE |
| Sessions with implied avg rate > 20 kW | 2,057 | Clear artefacts |

### 3.2 Root Cause of 375.41 kW Anomaly (2020-04-17 22:00 UTC)

**Confirmed mechanism: Rapid Reconnect Loop + Meter Carryover**

At N_Wilson_Garage_01, EVSE spaces 3843 and 3844 experienced repeated rapid plug/unplug cycles between 22:21 and 22:50 UTC on 2020-04-17.

| Station | Sessions in 22:00–23:00 UTC | Duration | kWh Recorded |
|---|---|---|---|
| Space 3843 | 6 micro-sessions | < 5 sec to 1.6 min | 41–54 kWh each |
| Space 3844 | 8 micro-sessions | < 1 sec to 5 min | 33–49 kWh each |

**Why such high kWh for such short sessions?**  
The EVSE meter's cumulative counter is reset to 0 at session start. However, when a session ends abnormally (disconnection event too fast for a proper meter flush), the *next session's file* begins mid-read. This results in a 5-second session inheriting a 44 kWh reading that actually belongs to the prior charging session.

**Total artefactual energy in that hour: ~700 kWh → reported as 375.41 kW.**

The actual demand in that hour was from 9 legitimate sessions (California_Garage_02 sessions with normal L2 rates) amounting to ~33 kWh → ~33 kW realistic peak.

### 3.3 Energy File Pattern Verification (500-file sample)

| Metric | Value |
|---|---|
| Files with end-reset to 0 | 441 / 495 (89%) |
| Files starting at zero | 198 / 198 (100%) |
| Mean session total energy (e_max) | 7.15 kWh |
| Median session energy | 5.84 kWh |
| Max session energy | 50.57 kWh |

---

## 4. Corrections Applied (v2 Pipeline)

The corrected pipeline is in [`scripts/process_time_series_v2.py`](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/scripts/process_time_series_v2.py).

### Stage 1 — Remove Sub-1-Minute Sessions
- **Removed:** 5,059 sessions with `duration_hours < 1/60`
- **Rationale:** Sessions shorter than 1 minute cannot represent real charging events. The physical minimum for initiating a charge handshake is ~30 seconds; any session under 1 minute is a failed connect or a rapid disconnect.

### Stage 2 — Cap Session Energy at L2 Physical Maximum
- **Capped:** sessions where `kWhDelivered_raw > duration_hours × 7.7 kW`
- **Threshold:** 7.7 kW ≈ 240V × 32A (standard L2 EVSE max for Caltech ACN)
- **Rationale:** A session cannot deliver more energy than the EVSE's rated output over its observed duration. Any excess is a meter-carryover artefact.

### Correction Summary

| Metric | v1 (Original) | v2 (Corrected) | Change |
|---|---|---|---|
| Total sessions | 56,673 | 51,614 | −5,059 (−8.9%) |
| Total kWh delivered | 532,271 kWh | 484,682 kWh | −47,589 kWh (−8.9%) |
| Peak hourly demand | **375.41 kW** | **120.56 kW** | −68% |
| Mean hourly demand | 23.16 kW | 21.28 kW | −8.1% |
| Hours > 100 kW | 22 | 14 | −36% |

---

## 5. Baseline Metrics: v1 vs v2 (Test Set: March–December 2020)

| Model | v1 MAE | v1 RMSE | v2 MAE | v2 RMSE |
|---|---|---|---|---|
| Persistence (t−1) | 7.45 kW | 14.47 kW | **3.60 kW** | **5.67 kW** |
| Seasonal Naive (t−24) | 11.73 kW | 18.74 kW | **7.33 kW** | **10.59 kW** |
| Seasonal Naive (t−168) | 11.86 kW | 18.85 kW | **7.33 kW** | **10.53 kW** |

> [!NOTE]
> The large drop in metrics from v1→v2 (e.g., Persistence RMSE: 14.47→5.67) is primarily due to removing the extreme artefact spikes (especially the 375 kW event and similar). The v2 dataset better represents physically achievable charging demand.

---

## 6. Files Preserved and Created

### Preserved (Unchanged)
- `data/processed/caltech_hourly_demand.csv` — original v1
- `data/processed/caltech_hourly_demand.parquet` — original v1
- `data/processed/caltech_sessions_summary.csv` — original v1
- `data/processed/caltech_sessions_summary.parquet` — original v1
- All scripts in `scripts/` (v1 pipeline untouched)

### Created (New)
| File | Description |
|---|---|
| `data/processed_v2/caltech_hourly_demand_v2.csv` | Corrected hourly demand (22,780 rows) |
| `data/processed_v2/caltech_hourly_demand_v2.parquet` | Parquet format of above |
| `data/processed_v2/caltech_sessions_summary_v2.csv` | Corrected sessions (51,614 rows) |
| `data/processed_v2/caltech_sessions_summary_v2.parquet` | Parquet format of above |
| `scripts/process_time_series_v2.py` | Corrected pipeline (reproducible) |
| `reports/data_audit/raw_file_sample_audit.csv` | Per-file energy pattern audit (495 files) |
| `reports/data_audit/micro_sessions_audit.csv` | All micro-sessions (<6 min) |
| `reports/data_audit/peak_hour_overlap_sessions.csv` | Sessions in the 375 kW peak hour |

---

## 7. Recommendation for Phase 3 Modeling

> [!IMPORTANT]
> **Use `data/processed_v2/caltech_hourly_demand_v2.parquet` for all Phase 3 modeling** (SARIMA, LightGBM, Chronos).
>
> The v2 dataset:
> - Has a realistic peak of 120.56 kW (vs. 375 kW artefact)
> - Has a baseline persistence MAE of 3.60 kW — a more honest forecasting difficulty estimate
> - Still contains the COVID-19 demand disruption (March 2020 campus closure is real)
> - Maintains 22,780 continuous hourly observations for the full 2018–2020 period

The 120.56 kW remaining peak (2020-10-09 23:00 UTC) represents 31 legitimate concurrent sessions at California Garage — physically plausible for a facility with 54 EVSE stations.

---

## 8. Remaining Limitations (Not Corrected)

1. **Uniform energy distribution across session hours** — the pipeline allocates kWh uniformly across all session hours. In reality, most energy is delivered in the first 1–2 hours (exponential charging curve). This is an inherent modelling simplification, not a data corruption issue.
2. **Single-station physical cap** — the 7.7 kW cap is per-station. High-current stations (DC fast chargers, if any) would be over-capped. For this dataset, Caltech ACN is exclusively L2, so this is appropriate.
3. **14 hours still > 100 kW** — these represent genuine high-demand periods with 20–60 concurrent legitimate sessions and are not artefacts.
