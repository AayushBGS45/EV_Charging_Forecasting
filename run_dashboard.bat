@echo off
title EV Demand Forecasting Runner
echo ========================================================
echo Starting EV Charging Demand Forecasting Backend and Frontend...
echo ========================================================

:: 1. Launch FastAPI Backend in a new terminal window
start "FastAPI Backend (Port 8000)" cmd /k "cd /d %~dp0EV_Charging_Forecasting && ..\.venv\Scripts\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000 --reload"

:: 2. Launch React + Vite Frontend in a new terminal window
start "Vite React Dashboard (Port 5173)" cmd /k "cd /d %~dp0EV_Charging_Forecasting\frontend && npm run dev"

echo.
echo Both servers have been launched in separate windows!
echo.
echo Backend API:     http://127.0.0.1:8000
echo Swagger Docs:    http://127.0.0.1:8000/docs
echo Forecast Route:  http://127.0.0.1:8000/api/forecast?horizon=24
echo Frontend App:    http://localhost:5173
echo.
echo ========================================================
pause
