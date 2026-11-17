@echo off
title RoomForge 3D Studio
echo ========================================================
echo        Starting RoomForge 3D Generative Studio
echo ========================================================
echo.
echo Starting Backend and Frontend services...
start "RoomForge Backend (FastAPI)" cmd /k "set PYTHONPATH=backend;worker && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
start "RoomForge Frontend (Next.js)" cmd /k "npm --prefix frontend run dev -- -p 3000"

echo.
echo Waiting for servers to initialize...
timeout /t 3 /nobreak >nul
echo.
echo Opening Studio in default browser: http://localhost:3000 ...
start http://localhost:3000
echo.
echo ========================================================
echo RoomForge is running:
echo - Desktop Web Studio: http://localhost:3000
echo - Backend API Docs:   http://127.0.0.1:8000/docs
echo ========================================================
