# Prediction Modeling of Non-Stationary EV Charging Demand Using ACN-Data and Foundation Models

## B.Tech Project Overview
Electric Vehicle (EV) charging demand exhibits strong non-stationary behavior due to varying user behaviors, seasonal trends, weather influences, campus schedules, and power grid constraints. This project focuses on modeling and predicting EV charging demand using the **Caltech ACN-Data** dataset and modern foundation models for time series forecasting.

---

## Tech Stack
- **Data Analysis & Modeling**: Python (Pandas, NumPy, Matplotlib, Seaborn, SciPy)
- **Forecasting Models (Upcoming Phases)**: Time Series Foundation Models & Statistical Baselines
- **Backend API**: FastAPI + Uvicorn
- **Frontend Dashboard**: React + Vite

---

## Project Structure
```text
EV_Charging_Forecasting/
├── data/
│   ├── raw/                  # Original, unedited downloaded data (JSON/CSV)
│   ├── processed/            # Cleaned, aggregated time-series data
│   └── README.md             # Dataset documentation and schema details
├── notebooks/
│   └── 01_data_acquisition_and_inspection.ipynb # Phase 1 EDA & inspection notebook
├── scripts/
│   └── download_acn_data.py  # Script to download real session data from Caltech ACN API
├── backend/
│   ├── main.py               # FastAPI application entry point
│   └── __init__.py
├── frontend/
│   └── README.md             # Instructions for React + Vite dashboard (Phase 3)
├── reports/
│   ├── figures/              # Saved visualization plots from EDA
│   └── phase1_eda_summary.md # Report summary template for Phase 1 findings
├── .env.example              # Sample environment configuration (API tokens, dates)
├── .gitignore                # Git ignore configuration
├── requirements.txt          # Python dependencies for Phase 1
└── README.md                 # Project documentation (this file)
```

---

## Phase 1 Objectives
1. **Workspace Setup**: Establish clean directory hierarchy and dependency management.
2. **Dataset Acquisition**: Guide retrieval of authentic EV charging data from Caltech's ACN-Data API.
3. **Data Inspection**: Examine schema, data types, missing values, timestamps, and key metrics.
4. **Initial Exploratory Data Analysis (EDA)**: Calculate session durations, energy delivered (kWh), hourly/daily distributions, and identify non-stationary patterns.

---

## Quickstart Guide

### 1. Python Environment Setup
Open your terminal (PowerShell or Bash) and navigate to the project directory:

```bash
cd EV_Charging_Forecasting
python -m venv venv
```

Activate the virtual environment:
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If script execution is disabled on Windows, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first)*
- **Windows (Command Prompt)**:
  ```cmd
  venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

### 2. Processing the Raw Time Series Dataset
The raw static dataset in `data/raw/ACN-Data-Static-main/time series data` contains over 85,000 session files. To process and aggregate them into structured datasets:

```bash
# Process Caltech facility time series (multiprocessing enabled)
python scripts/process_time_series.py --site caltech

# Generate publication-grade EDA figures and run statistical stationarity tests
python scripts/generate_eda_visualizations.py
```

Generated artifacts in `data/processed/`:
- `caltech_sessions_summary.csv` & `.parquet`: 56,673 session records with connection duration, kWh delivered, and peak currents.
- `caltech_hourly_demand.csv` & `.parquet`: 22,780 continuous hourly intervals of facility power demand (kW) from May 2018 to Dec 2020.

---

### 3. Running Data Inspection & EDA
Launch Jupyter Notebook to inspect the dataset:
```bash
jupyter lab
# or:
jupyter notebook notebooks/01_data_acquisition_and_inspection.ipynb
```

---

### 4. Running the Backend Server
Start the FastAPI server:
```bash
uvicorn backend.main:app --reload --port 8000
```

Available REST API Endpoints:
- `GET /`: API status catalog
- `GET /health`: Service health check
- `GET /api/stats`: Facility metrics, total sessions (56,673), energy delivered (532,270 kWh), and timeline coverage
- `GET /api/demand/hourly?limit=500`: Continuous hourly power demand (kW)
- `GET /api/demand/diurnal`: 24-hour diurnal power curve (Weekday vs. Weekend split)
- `GET /api/demand/stationarity`: ADF & KPSS statistical stationarity test diagnostics
- `GET /api/sessions/recent?limit=25`: Recent charging session records
- `GET /docs`: Interactive Swagger UI (`http://127.0.0.1:8000/docs`)
