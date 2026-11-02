@echo off
setlocal
cd /d "%~dp0"

where ollama >nul 2>nul
if errorlevel 1 (
    echo ERROR: Ollama is not installed or is not on PATH.
    echo Install it from https://ollama.com, reopen PowerShell, then run this file again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: The Python virtual environment was not found.
    echo Run "uv sync" in this folder first, then run this file again.
    pause
    exit /b 1
)

echo Checking Ollama service...
curl --silent --fail --max-time 2 http://127.0.0.1:11434/api/tags >nul 2>nul
if not errorlevel 1 goto ollama_ready

echo Starting Ollama in a separate window...
start "Ollama Server" /min cmd /c "ollama serve"
for /L %%I in (1,1,10) do (
    timeout /t 1 >nul
    curl --silent --fail --max-time 2 http://127.0.0.1:11434/api/tags >nul 2>nul
    if not errorlevel 1 goto ollama_ready
)

echo ERROR: Ollama did not start at http://127.0.0.1:11434.
echo Open the Ollama desktop app or run "ollama serve", then try again.
pause
exit /b 1

:ollama_ready
echo Pulling the embedding model...
ollama pull nomic-embed-text
if errorlevel 1 goto failed

echo Pulling the chat model...
ollama pull qwen2.5:7b-instruct
if errorlevel 1 goto failed

echo Indexing PDFs from backend\data into ChromaDB...
".venv\Scripts\python.exe" backend\scripts\index_documents.py
if errorlevel 1 goto failed

echo.
echo Setup complete. Starting the browser app...
call start_server.bat
exit /b %errorlevel%

:failed
echo.
echo Setup failed. Read the error above, fix it, and run this file again.
pause
exit /b 1
