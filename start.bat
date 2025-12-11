@echo off
echo ============================================
echo AI Pump Detection Bot - Starting...
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo WARNING: .env file not found!
    echo Please create .env file with your Telegram credentials
    echo See SETUP_GUIDE.txt for instructions
    echo.
    pause
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import ccxt" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Starting bot in AUTO-RESTART mode...
echo Bot will restart automatically if it crashes
echo Press Ctrl+C to stop
echo.
echo ============================================
echo.

:restart
python main.py
if errorlevel 1 (
    echo.
    echo Bot stopped or crashed. Restarting in 5 seconds...
    echo Press Ctrl+C to stop completely
    timeout /t 5 /nobreak >nul
    goto restart
) else (
    echo.
    echo Bot stopped normally.
)

pause

