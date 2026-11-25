# Launch FastAPI Backend in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot\EV_Charging_Forecasting'; & '$PSScriptRoot\.venv\Scripts\uvicorn.exe' backend.main:app --host 127.0.0.1 --port 8000 --reload"

# Launch React + Vite Frontend in a new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot\EV_Charging_Forecasting\frontend'; npm run dev"

Write-Host "`n========================================================" -ForegroundColor Cyan
Write-Host "Both servers launched in separate windows!" -ForegroundColor Green
Write-Host "Backend API:    http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "Swagger Docs:   http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "Forecast Route: http://127.0.0.1:8000/api/forecast?horizon=24" -ForegroundColor Yellow
Write-Host "Frontend App:   http://localhost:5173" -ForegroundColor Green
Write-Host "========================================================`n" -ForegroundColor Cyan
