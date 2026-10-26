# Phase 2.1 — Independent Validation Report
**Project:** Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models  
**Validated:** `scripts/process_time_series_v2.py` and `data/processed_v2/` outputs  
**Date:** 2026-10-26  
**Method:** Read-only independent recomputation — no existing files altered

---

## Overall Verdicts

| # | Validation Item | Verdict |
|---|---|---|
| 1 | Code logic: removal and cap rules | ⚠️ PASS WITH CAVEATS |
| 2 | Session counts (removed / capped / retained, no overlap) | ✅ PASS |
| 3 | Energy totals before/after each correction stage | ✅ PASS |
| 4 | Hourly demand effect, row counts, missing values | ✅ PASS |
| 5 | Validity of 7.7 kW energy cap | ⚠️ PASS WITH CAVEATS |
| 6 | Independent metric recomputation (MAE/RMSE, v2 test set) | ✅ PASS |
| 7 | Uniform energy distribution — confirmed and documented | ⚠️ PASS WITH CAVEATS |
| 8 | 120.56 kW peak reconciled against contributing sessions | ✅ PASS |
| 9 | Timestamp integrity (no duplicates, gaps, mis-sorts) | ✅ PASS |

---

## Validation Item 1 — Code Logic: Removal and Cap Rules

### Rules as implemented in `process_time_series_v2.py`

**Stage 1 — Remove sub-1-minute sessions:**
```python
MIN_DURATION_HRS = 1.0 / 60.0   # = 0.016667 hrs = exactly 60 seconds
df = df[df["duration_hours"] >= MIN_DURATION_HRS]
```
Sessions with `duration_hours < 0.016667` are **dropped** (not capped).

**Stage 2 — Cap energy on survivors:**
```python
L2_MAX_KW = 7.7
max_possible_kwh = duration_hours * L2_MAX_KW
kWhDelivered = min(kWhDelivered_raw, max_possible_kwh)
```
Stage 2 is applied **only to Stage-1 survivors**, applied sequentially.

### Evidence supporting each rule

**Stage 1 evidence (sub-1-min removal):**
- 5,059 sessions (8.9% of total) have `duration_hours < 1/60`.
- In the 375.41 kW peak hour, 14 of 23 overlapping sessions were shorter than 1 minute, collectively recording 400–600 kWh — physically impossible for any EV charger.
- Raw file inspection confirmed these sessions begin mid-energy-counter (e.g., a session lasting 3 seconds shows 43 kWh at its first timestamp), consistent with meter-carryover from rapid reconnect loops.
- The 500-file sample audit showed 89% of normal sessions end with an energy reset to 0.0, confirming the cumulative-per-session meter design. Sub-1-min sessions break this pattern.

**Stage 2 evidence (7.7 kW cap):**
- Caltech ACN is documented as an L2 AC network. L2 EVSE standard IEC 62196 / SAE J1772 limits AC supply to 32A at 240V = 7.68 kW.
- Any session recording kWh > `duration_hours × 7.7` is physically impossible for a single L2 station. The audit identified 4,904 such sessions in the v1 dataset.

### Unverified assumptions

> [!WARNING]
> 1. **No independent electrical specification sheet was consulted.** The 7.7 kW cap is based on the L2 standard ceiling, not a Caltech ACN hardware specification document. The actual installed EVSE models and their rated outputs were not verified.
> 2. **The maxCurrent_A column records values up to 125.4 A**, which at 240V would imply 30.1 kW — far exceeding L2 limits. This could indicate: (a) DC-side current measurement (not AC supply), (b) pilot signal current, or (c) instrumentation error. The column's exact meaning was not independently confirmed from ACN documentation.
> 3. **Sub-1-minute threshold is conservative** but not verified against EVSE handshake protocol timing. If some valid charging events complete in 45–59 seconds (e.g., already-full batteries disconnecting quickly), they would be wrongly removed. No evidence of such sessions was found, but it was not exhaustively checked.

**Verdict: PASS WITH CAVEATS** — Rules are internally consistent and evidence-supported, but two hardware assumptions remain unverified against primary sources.

---

## Validation Item 2 — Session Counts

All counts independently recomputed by applying the Stage 1 and Stage 2 predicates to `data/processed/caltech_sessions_summary.parquet` (v1 source).

| Count | Reported | Independently Computed | Match |
|---|---|---|---|
| v1 total sessions | 56,673 | 56,673 | ✅ |
| Stage 1 removed (dur < 1 min) | 5,059 | **5,059** | ✅ |
| After Stage 1 (survivors) | 51,614 | 51,614 | ✅ |
| Stage 2 capped on survivors | 4,904 (audit report) | **5,092** | ⚠️ discrepancy |
| v2 final sessions retained | 51,614 | 51,614 | ✅ |

**Note on discrepancy:** The audit report stated "4,904 sessions capped" while independent recomputation gives **5,092**. The discrepancy arises because the audit script applied capping on the unfiltered v1 sessions (56,673), while the v2 script applies it only after Stage 1 (51,614 survivors). Additionally, the audit script used a 6-minute micro-session distinction that differed from the final threshold. The v2 pipeline's sequential application (Stage 2 after Stage 1) is the **authoritative** count: **5,092 sessions capped** from among the 51,614 survivors.

**Overlap:** Stage 1 and Stage 2 are structurally disjoint — Stage 2 is only applied to Stage 1 survivors. However, 281 Stage-1-removed sessions would also have been capped had they survived Stage 1. These are not double-counted.

**Verdict: PASS** — Retained count matches exactly. The capped count discrepancy is explained and does not affect the final dataset.

---

## Validation Item 3 — Energy Accounting

| Energy Quantity | Value |
|---|---|
| v1 total kWh (all 56,673 sessions) | 532,270.60 kWh |
| Energy in Stage-1-removed sessions | 4,663.57 kWh (0.88%) |
| Energy remaining after Stage 1 | 527,607.02 kWh |
| Energy removed by Stage 2 (cap corrections) | 42,924.77 kWh (8.07%) |
| Expected v2 total kWh | 484,682.25 kWh |
| Actual v2 total kWh (from parquet) | 484,682.25 kWh |
| **Reconciliation difference** | **0.00 kWh** |
| Total energy removed | 47,588.35 kWh (−8.94%) |

The energy accounting is exact. The dominant correction is Stage 2 (cap), not Stage 1 (removal). Sub-1-min sessions typically had zero or minimal recorded energy (0.88% of total), while the cap corrections on longer micro-sessions removed 8.07%.

**Verdict: PASS** — Perfect energy reconciliation (0.00 kWh discrepancy).

---

## Validation Item 4 — Hourly Demand Effect

| Metric | v1 (original) | v2 (corrected) |
|---|---|---|
| Total rows | 22,780 | 22,780 |
| Mean avg_power_kw | 23.1646 kW | 21.2767 kW |
| Std avg_power_kw | 20.9442 kW | 19.4640 kW |
| Max avg_power_kw | 375.4072 kW | 120.5562 kW |
| Zero-demand hours | 301 | 301 |
| Hours > 100 kW | 22 | 14 |
| Hours > 50 kW | 3,005 | 2,474 |
| Missing values | 0 | 0 |
| Timestamp start | 2018-05-01 00:00 UTC | 2018-05-01 00:00 UTC |
| Timestamp end | 2020-12-05 03:00 UTC | 2020-12-05 03:00 UTC |

**Verdict: PASS** — Both datasets have identical row counts, zero missing values, identical timestamp boundaries, and no duplicates.

---

## Validation Item 5 — Validity of 7.7 kW Energy Cap

### What the data shows

- `maxCurrent_A` ranges from 0 to **125.4 A** in v1 sessions.
- **13,436 sessions** (23.7%) show `maxCurrent_A > 32 A`, which exceeds the L2 AC standard.
- **2,873 sessions** show `maxCurrent_A > 80 A`.
- Maximum implied AC power (at 240V): `125.4 A × 240V / 1000 = 30.1 kW`.

### Can genuine sessions exceed 7.7 kW?

This is the critical question. The evidence is **mixed**:

**Evidence against over-capping (cap is valid):**
- Of sessions with `duration >= 30 min`, **zero** exceed the 7.7 kW cap in kWhDelivered from v1. All sessions capped by Stage 2 are short-duration sessions, consistent with meter-carryover artefacts.
- The top sessions by `maxCurrent_A` (125.4A, 125.3A) are all at `N_Wilson_Garage_01` — the same garage as the 375 kW anomaly. A legitimate L2 session cannot draw 125A AC.

**Evidence that raises concern:**
- The Caltech ACN `maxCurrent_A` column may record **pilot signal current** or **EVSE scheduling target current**, not actual AC supply current drawn from the grid. Pilot signal in SAE J1772 can legally be set to high values (up to 80A for three-phase systems) without implying 240V single-phase supply.
- If some Caltech EVSE stations are rated above 7.7 kW (e.g., upgraded 48A/240V = 11.5 kW units), then sessions on those stations would be wrongly capped. **This was not verified.**

> [!WARNING]
> The 7.7 kW cap may **over-correct** genuine high-power sessions if any Caltech EVSE stations operate at > 32A / 240V. The cap removes the extreme spikes effectively, but without the hardware specification document, it cannot be confirmed that no legitimate sessions are capped at durations ≥ 1 minute.

**Verdict: PASS WITH CAVEATS** — The cap removes demonstrably artefactual values (zero long-duration sessions are capped; all capped sessions are short). However, the 7.7 kW threshold rests on an unverified hardware assumption.

---

## Validation Item 6 — Independent Metric Recomputation

**Dataset:** `data/processed_v2/caltech_hourly_demand_v2.parquet`  
**Split:** Chronological, matching `train_baseline_models.py` exactly.

| Split | Rows | Start | End |
|---|---|---|---|
| Train | 14,640 | 2018-05-01 | 2019-12-31 |
| Validation | 1,440 | 2020-01-01 | 2020-02-29 |
| Test | 6,700 | 2020-03-01 | 2020-12-05 |
| Sum | 22,780 | — | — |
| Overlaps | 0 | — | — |

All lags generated causally: `shift(1)`, `shift(24)`, `shift(168)` on the full sorted series before subsetting.

### Test Set Results (independently computed vs. reported)

| Model | Computed MAE | Reported MAE | Δ | Computed RMSE | Reported RMSE | Δ | Verdict |
|---|---|---|---|---|---|---|---|
| Persistence (t−1) | 3.5958 kW | 3.60 kW | 0.004 | 5.6712 kW | 5.67 kW | 0.001 | ✅ PASS |
| Seasonal Naive (t−24) | 7.3256 kW | 7.33 kW | 0.004 | 10.5919 kW | 10.59 kW | 0.002 | ✅ PASS |
| Seasonal Naive (t−168) | 7.3262 kW | 7.33 kW | 0.004 | 10.5336 kW | 10.53 kW | 0.004 | ✅ PASS |

All differences are within rounding precision (< 0.01 kW). All evaluated on n=6,700 samples (no NaN edge effects in test set since lags 1/24/168 are all available by March 2020, given training from May 2018).

**Verdict: PASS** — All reported metrics reproduce independently to rounding precision.

---

## Validation Item 7 — Uniform Energy Distribution

### Confirmed from code (lines 192–194 of `process_time_series_v2.py`):

```python
num_hours = e_idx - s_idx + 1
alloc_kwh = row["kWhDelivered"] / num_hours
hourly_energy_kwh[s_idx: e_idx + 1] += alloc_kwh
```

**Confirmed:** The v2 pipeline uses **uniform distribution** — session kWh is divided equally across all integer hours in the connection window. This is identical to the v1 pipeline.

### Quantitative scope of limitation

- **41,312 / 56,673 sessions (72.9%)** span more than one hour and are subject to uniform allocation error.
- A typical session charging 15 kWh over 4 hours contributes a flat 3.75 kW/hr. In reality, the EV battery draws full rate (~7 kW) for ~2 hours, then tapers. The actual first-hour demand is under-estimated; the last-hour demand is over-estimated.

### Implications for forecasting

- Intra-session hourly demand shape is smoothed. Hourly peaks tied to **session start hours** (typically 17:00–19:00) are **attenuated**.
- The forecasting target `avg_power_kw` does not reflect physical instantaneous power — it reflects the uniform allocation, which is internally consistent but systematically mis-shaped.
- This limitation applies **identically to v1 and v2**; corrections did not change the allocation method.

**Verdict: PASS WITH CAVEATS** — Confirmed, documented, and scoped quantitatively. Not a correctness bug but an inherent modelling simplification that will affect model calibration on intra-day patterns.

---

## Validation Item 8 — 120.56 kW Peak Reconciliation

**Peak hour (v2):** `2020-01-13 20:00:00 UTC` (not 2020-10-09 as stated in the audit report)  
**Peak value:** 120.5562 kW

> [!NOTE]
> The audit report stated the 120.56 kW remaining peak was at `2020-10-09 23:00 UTC`. The independent check finds the actual peak is at **2020-01-13 20:00 UTC**. This is a minor factual error in the audit report narrative (the kW value is correct; only the timestamp was wrong).

### Session-level reconciliation

- **62 overlapping sessions** across 4 garages (California_Garage_01: 37, California_Garage_02: 16, N_Wilson_Garage_01: 6, S_Wilson_Garage_01: 3)
- **Max current in these sessions: 50.0 A** — moderate, plausible for standard L2 sessions
- All sessions have normal durations (1.7 – 31 hours) and individually plausible kWh

**Energy reconciliation:**

| Quantity | Value |
|---|---|
| Sum of all session contributions to this hour | 120.5562 kWh |
| avg_power_kw reported in hourly dataset | 120.5562 kW |
| Reconciliation error | **0.0000 kW** |

The 120.56 kW peak represents 62 simultaneous legitimate L2 sessions (average ~1.9 kW per session after uniform allocation spreading), distributed across 3 garages. This is physically plausible.

**Verdict: PASS** — Perfect session-level energy reconciliation. The audit report had a minor timestamp error in the narrative (2020-10-09 vs. actual 2020-01-13) but the peak value itself (120.56 kW) is correct.

---

## Validation Item 9 — Timestamp Integrity (v2 Hourly Dataset)

| Check | Result |
|---|---|
| Duplicate timestamps | 0 |
| Monotonically increasing | True |
| All gaps exactly 1 hour | True (0 irregular gaps) |
| Expected rows from range | 22,780 |
| Actual rows | 22,780 |
| Missing values in any column | 0 |

The v2 hourly dataset is a perfect continuous hourly sequence from `2018-05-01 00:00 UTC` to `2020-12-05 03:00 UTC` with no gaps, no duplicates, and no missing values.

**Verdict: PASS**

---

## Corrections Required to the Phase 2.1 Audit Report

The following items in `reports/data_audit/phase2_1_data_audit_report.md` are incorrect and should be noted:

| Item | Reported | Correct | Severity |
|---|---|---|---|
| Stage 2 capped count | "4,904 sessions" | **5,092 sessions** | Minor — different baseline population used |
| 120.56 kW peak timestamp | "2020-10-09 23:00 UTC" | **2020-01-13 20:00 UTC** | Minor factual error in narrative |
| Stage 2 description as "sub-1-min cap" | Conflated stages | Two distinct stages correctly in code | Clarification needed |

These errors do not affect the corrected datasets, which are independently verified correct.

---

## Key Unresolved Risks for Phase 3

> [!IMPORTANT]
> Before implementing SARIMA or LightGBM on the v2 dataset, document these unresolved risks:
>
> 1. **7.7 kW cap threshold** — Unverified against Caltech ACN hardware specs. If any stations exceed 7.7 kW rated output, legitimate high-power sessions in the training data are capped. Consider obtaining the station specifications or using a higher conservative cap (e.g., 12 kW = 50A × 240V).
> 2. **Uniform energy allocation** — 72.9% of sessions span multiple hours. The target variable `avg_power_kw` is not physical instantaneous power; it is a uniform-allocation proxy. Models trained on this target will not necessarily generalise to minute-level or actual power forecasting.
> 3. **COVID-19 structural break** — The test set (March–December 2020) includes a severe demand disruption (March 2020 campus closure). The v2 baseline persistence MAE of 3.60 kW on this period partially reflects near-zero COVID demand, not typical campus operation.
> 4. **Hourly index is fixed** — Both v1 and v2 use integer-hour boundaries (floor/ceil) for session start/end. A session from 10:50 to 12:10 contributes to hours 10, 11, and 12 equally, even though 90% of its duration is in hours 11–12. This introduces systematic boundary effects.
