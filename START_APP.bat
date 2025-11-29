@echo off
REM Quick launcher - changes to correct directory and runs app
echo ================================================
echo Parameter Pruning Dashboard Launcher
echo ================================================
echo.

REM Change to project directory
cd /d "%~dp0"
echo Current directory: %CD%
echo.

REM Check if streamlit_app.py exists
if not exist "streamlit_app.py" (
    echo ERROR: streamlit_app.py not found in current directory!
    echo Please make sure you're running this from the project folder.
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo Error: Python is not installed or not in PATH
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=py
    )
) else (
    set PYTHON_CMD=python
)

echo Starting Streamlit Dashboard...
echo The dashboard will open in your browser.
echo Press Ctrl+C to stop the server.
echo.
echo ================================================
echo.

%PYTHON_CMD% -m streamlit run streamlit_app.py

pause

