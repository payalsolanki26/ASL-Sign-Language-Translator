@echo off
title Sign Reader - Live ASL Hand Sign Translator
cd /d "%~dp0"

echo ========================================================
echo   Starting Sign Reader - Live ASL Translator
echo ========================================================
echo.
echo URL: http://127.0.0.1:8000
echo Press CTRL+C to stop the server anytime.
echo.

:: Open the browser automatically after 2 seconds
start "" http://127.0.0.1:8000

:: Run the FastAPI Uvicorn Server
python -m uvicorn asl.serve.api:app --app-dir src --host 127.0.0.1 --port 8000
pause
