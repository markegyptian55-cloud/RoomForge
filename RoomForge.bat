@echo off
setlocal
title RoomForge Desktop Studio

echo =================================================================
echo   ROOMFORGE 3D SPATIAL STUDIO (PORTABLE)
echo =================================================================
echo.

:: Detect Python environment
set "PY_CMD="
if exist "%~dp0runtime\python\python.exe" (
    set "PY_CMD=%~dp0runtime\python\python.exe"
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set "PY_CMD=python"
    )
)

if "%PY_CMD%"=="" (
    echo [ERROR] Python 3.11+ was not found on your system or in ./runtime/python.
    echo Please install Python 3.11+ or run RoomForge-Setup.exe installer.
    pause
    exit /b 1
)

:: Launch desktop supervisor in portable mode
"%PY_CMD%" "%~dp0desktop\launcher.py" --portable

endlocal
