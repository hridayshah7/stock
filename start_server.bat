@echo off
echo 🎯 Stock Sniper API - Starting Server
echo ====================================

REM Install requirements
echo 📦 Installing requirements...
pip install -r requirements.txt

REM Start the server
echo 🚀 Starting FastAPI server...
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop
echo.

uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

pause
