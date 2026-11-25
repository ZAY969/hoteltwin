@echo off
cd /d "%~dp0"
echo [1/3] Checking Python...
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10+ from https://www.python.org/downloads/ and tick "Add python.exe to PATH", then run this file again.
  pause
  exit /b 1
)
if not exist .venv (
  echo [2/3] First run: creating environment and installing packages, this takes a few minutes...
  python -m venv .venv
  call .venv\Scripts\activate.bat
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Package installation failed. Check the network and run this file again.
    pause
    exit /b 1
  )
) else (
  call .venv\Scripts\activate.bat
)
echo [3/3] Starting the system. The dashboard opens in your browser in about 15 seconds.
echo Keep this window open while you use it. Close it to stop the system.
start "" cmd /c "timeout /t 15 >nul & start http://localhost:8000"
python server.py
pause
