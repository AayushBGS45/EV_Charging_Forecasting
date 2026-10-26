# EV Charging Forecasting - Frontend Dashboard

## Overview
This directory will host the interactive web dashboard built with **React** and **Vite** in Phase 3.

## Planned Features
- Real-time time series charts (historical demand vs. model forecasts).
- Station-level and site-level aggregation toggles (Caltech, JPL, Office001).
- Metrics display (MAE, RMSE, Peak Load, Capacity Utilization).
- Fast, modern responsive UI connecting to the FastAPI backend at `http://127.0.0.1:8000`.

## Initialization Plan (Phase 3)
When we reach Phase 3, we will initialize the Vite app:
```bash
npm create vite@latest . -- --template react
npm install
npm run dev
```
