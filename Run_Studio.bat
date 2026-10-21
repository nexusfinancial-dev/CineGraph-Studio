@echo off
chcp 65001 >nul
title Studio Director - Autonomous AI Film Pre-Production Studio

echo ======================================================================
echo    STUDIO DIRECTOR: AUTONOMOUS AI FILM PRE-PRODUCTION STUDIO
echo ======================================================================
echo.
echo [1/3] Locating Python environment...

set PYTHON_EXE=C:\Users\MiroAI\.conda\envs\qwen_env\python.exe

if not exist "%PYTHON_EXE%" (
    set PYTHON_EXE=python
)

echo Python Path: %PYTHON_EXE%
echo.

echo [2/3] Checking Port 8555...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8555 ^| findstr LISTENING') do (
    echo Found previous server instance on PID: %%a. Cleaning up...
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

echo.
echo [3/3] Starting Studio Director Server on http://127.0.0.1:8555 ...
echo ======================================================================
echo  Server is LIVE! 
echo  Opening studio UI at: http://127.0.0.1:8555/agent
echo  To shut down, close this window or press Ctrl + C.
echo ======================================================================
echo.

start "" "http://127.0.0.1:8555/agent"

"%PYTHON_EXE%" -m uvicorn server:app --host 127.0.0.1 --port 8555

pause
