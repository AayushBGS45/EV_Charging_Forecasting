# Phase 1: Exploratory Data Analysis & Statistical Inspection Report

**Project Title**: Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models  
**Phase**: 1 - Data Acquisition, Static Time Series Pipeline, EDA & Stationarity Profiling  
**Target Facility**: Caltech Adaptive Charging Network (California Garage, Wilson Garages, LIGO)  
**Time Span**: May 1, 2018 – December 2, 2020 (22,780 continuous hourly intervals)  

---

## 1. Executive Summary & Dataset Profile
We extracted and ingested authentic time-series session records from the static snapshot of the **Caltech Adaptive Charging Network (ACN-Data)**.

| Metric | Empirical Value | Description |
|---|---|---|
| **Total Processed Sessions** | **56,673** | Discrete charging events recorded across Caltech stations |
| **Total Energy Delivered** | **532,270.60 kWh** (~0.53 GWh) | Cumulative electric energy supplied to vehicles |
| **Average Session Duration** | **4.83 hours** | Mean plug-in duration per charging event |
| **Median Session Duration** | **3.85 hours** | Typical commuter dwell time |
| **Target Time Series** | **Hourly Mean Facility Power (kW)** | Aggregated continuous grid load across all stations |
| **Hourly Steps Generated** | **22,780 hours** | May 2018 through December 2020 uninterrupted timeline |

Artifacts generated:
- [caltech_sessions_summary.csv](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/data/processed/caltech_sessions_summary.csv) & [.parquet](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/data/processed/caltech_sessions_summary.parquet)
- [caltech_hourly_demand.csv](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/data/processed/caltech_hourly_demand.csv) & [.parquet](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/data/processed/caltech_hourly_demand.parquet)

---

## 2. Statistical Stationarity Profiling
To formally evaluate the core research hypothesis regarding EV charging demand non-stationarity, we performed standard unit-root and stationarity hypothesis testing on the facility hourly demand series (`avg_power_kw`):

### Augmented Dickey-Fuller (ADF) Test
- **Test Statistic**: `-10.5627`
- **p-value**: `7.6539e-19` (Well below critical threshold $\alpha = 0.01$)
- **Interpretation**: Rejects unit-root non-stationarity, confirming strong mean-reverting diurnal/cyclical oscillations.

### KPSS Test for Trend & Level Stationarity ($H_0$: Stationary)
- **KPSS Statistic**: `1.3150`
- **Critical Threshold (1%)**: `0.739`
- **p-value**: `0.0100` ($p < 0.05$)
- **Interpretation**: **Rejects the null hypothesis of stationarity ($p < 0.05$)**.

> **Conclusion**: The combination of ADF rejection and KPSS rejection confirms that EV charging demand exhibits **structural non-stationarity** (shifting mean baselines, variance heterogeneity, and external regime breaks) combined with periodic diurnal seasonality.

---

## 3. Visual & Empirical Findings

### 1. Diurnal Arrival Patterns
- Commuters predominantly connect during the morning peak between **07:00 and 10:00 UTC (local morning work arrival hours)**.
- New charging connections drop sharply in the late afternoon and remain minimal through the overnight hours.
- *Visualization*: [diurnal_arrival_distribution.png](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/figures/diurnal_arrival_distribution.png)

### 2. Hourly Facility Power Demand (kW) – Weekday vs. Weekend Split
- Weekday facility load reaches an average plateau of **~35–45 kW** across the station network during peak midday charging hours.
- Weekend power demand drops dramatically to under **~10–12 kW**, representing a >70% reduction in facility utilization.
- *Visualization*: [hourly_power_demand_profile.png](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/figures/hourly_power_demand_profile.png)

### 3. Weekly Seasonality & Variance
- Monday through Friday exhibit consistent load distributions with higher interquartile ranges, while Saturday and Sunday collapse to sparse weekend usage.
- *Visualization*: [weekly_demand_seasonality.png](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/figures/weekly_demand_seasonality.png)

### 4. Long-Term Trajectory & Structural Breaks (2018–2020)
- The 30-day moving average highlights significant non-stationary trend shifts corresponding to Caltech academic quarters, winter breaks, and a catastrophic structural break in **March 2020 (COVID-19 campus closure)**.
- Traditional stationary statistical models (e.g., standard ARIMA) fail during such regime changes, motivating Foundation Time Series Models (Chronos, TimesFM) and adaptive estimators.
- *Visualization*: [long_term_demand_nonstationarity.png](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/figures/long_term_demand_nonstationarity.png)

### 5. Session Duration & Energy Distributions
- Connection duration peaks around **3 to 5 hours** with a long right tail extending to overnight connections.
- Energy delivered per session follows a log-normal distribution with a median of **~8.5 kWh** and the majority of transactions completing between **4 kWh and 25 kWh**.
- *Visualization*: [session_duration_energy_distribution.png](file:///c:/Users/aayus/Desktop/EV/EV_Charging_Forecasting/reports/figures/session_duration_energy_distribution.png)

---

## 4. Phase 2 Model Formulation Roadmap
With the static time series ingested, validated, and statistically profiled, the project is positioned for Phase 2:
1. **Target Series Definition**: 1-hour facility power demand (`avg_power_kw`) for forecasting horizons $H \in \{24, 48, 168\}$ hours (1-day, 2-day, 7-day).
2. **Statistical Baselines**: Seasonal Naive, AutoARIMA, and Exponential Smoothing (Holt-Winters).
3. **Machine Learning Baselines**: LightGBM / XGBoost with rolling lag features and calendar encodings.
4. **Foundation Models**: Amazon Chronos (T5-based probabilistic forecaster) and Google TimesFM.
5. **Evaluation Metrics**: MAE, RMSE, MAPE, and Continuous Ranked Probability Score (CRPS).
