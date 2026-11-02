@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The project virtual environment was not found.
    echo Run "uv sync" in this folder first, then run this file again.
    pause
    exit /b 1
)

echo Starting Brand Guardian at http://127.0.0.1:8000
echo Keep this window open while using the browser interface.
echo.
".venv\Scripts\python.exe" -m uvicorn backend.src.api.server:app --host 127.0.0.1 --port 8000 --reload

echo.
echo The server stopped. The error above explains why it could not start.
pause
