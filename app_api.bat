for /f "tokens=5" %%a in ('netstat -aon ^| findstr :48188') do (
    taskkill /F /PID %%a >nul 2>&1
)

cd /d "%~dp0"
uv run uvicorn app.main:app --host 0.0.0.0 --port 48188

pause
