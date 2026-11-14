@echo off
title RoomForge 3D Studio - Launch All Services
echo ========================================================
echo        Starting RoomForge 3D Generative Studio
echo ========================================================
echo.

echo [1/3] Starting FastAPI Backend on port 8000...
start "RoomForge Backend (FastAPI)" cmd /k "set PYTHONPATH=backend;worker && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

timeout /t 2 /nobreak >nul

echo [2/3] Starting Next.js Frontend on port 3000...
start "RoomForge Frontend (Next.js)" cmd /k "npm --prefix frontend run dev -- -p 3000"

timeout /t 3 /nobreak >nul

echo [3/3] Starting Cloudflare Quick Tunnel for Mobile Access...
start "RoomForge Cloudflare Tunnel" cmd /k "cloudflared tunnel --url http://localhost:3000"

echo.
echo ========================================================
echo All 3 services have been launched in separate terminals!
echo - Desktop Studio: http://localhost:3000
echo - Backend API:    http://127.0.0.1:8000/docs
echo - Check the Cloudflare window for your live mobile URL.
echo ========================================================
pause
