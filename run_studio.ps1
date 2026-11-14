# RoomForge 3D Studio - Launch All Services (PowerShell)
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       Starting RoomForge 3D Generative Studio          " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Starting FastAPI Backend on port 8000..." -ForegroundColor Yellow
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "`$env:PYTHONPATH='backend;worker'; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

Start-Sleep -Seconds 2

Write-Host "[2/3] Starting Next.js Frontend on port 3000..." -ForegroundColor Yellow
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "npm --prefix frontend run dev -- -p 3000"

Start-Sleep -Seconds 3

Write-Host "[3/3] Starting Cloudflare Quick Tunnel for Mobile Access..." -ForegroundColor Yellow
Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cloudflared tunnel --url http://localhost:3000"

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "All 3 services are running in separate windows!" -ForegroundColor Green
Write-Host "Desktop Studio: http://localhost:3000" -ForegroundColor White
Write-Host "Backend API:    http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Green
