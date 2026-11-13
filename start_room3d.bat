@echo off
title Room3D Local Servers
echo ========================================================
echo Starting Room3D Full-Stack Application
echo Control Plane: http://127.0.0.1:8000
echo Web Studio:    http://localhost:3000/projects/demo-room
echo ========================================================

start "Room3D FastAPI Backend" cmd /k "set PYTHONPATH=backend;worker && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
start "Room3D Next.js Frontend" cmd /k "cd frontend && npm run dev -- -p 3000"

echo Both servers launched in background windows.
echo Opening browser at http://localhost:3000/projects/demo-room ...
timeout /t 3 /nobreak > nul
start http://localhost:3000/projects/demo-room
exit
