@echo off
REM Launch (or restart) the CREMA spectrum-library / isotope-shift analysis GUI.
REM Run this from the user session so the server survives across agent turns
REM (tool-shell launches get reaped by the harness).
cd /d "%~dp0"

echo Stopping any existing GUI listening on port 8766...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr "127.0.0.1:8766" ^| findstr "LISTENING"') do taskkill /F /PID %%a >nul 2>&1

echo Starting spectrum library GUI (http://127.0.0.1:8766) ...
".venv\Scripts\python.exe" "spectrum_library_gui.py"

REM Keep the window open if the server exits so any error stays readable.
echo.
echo Server stopped. Press any key to close.
pause >nul
