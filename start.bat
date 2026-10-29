@echo off
REM Starts the app locally: http://localhost:8000
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py) || (set PY=python)
start "" http://localhost:8000
%PY% -m uvicorn app.main:app --port 8000
pause
