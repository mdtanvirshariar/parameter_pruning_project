@echo off
REM Windows batch script to run the application
echo ================================================
echo Parameter Pruning Dashboard Launcher
echo ================================================
echo.

REM Change to script's directory (project folder)
cd /d "%~dp0"
echo Working directory: %CD%
echo.

REM Check if streamlit_app.py exists
if not exist "streamlit_app.py" (
    echo ERROR: streamlit_app.py not found!
    echo Please make sure you're running this from the project folder.
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo Error: Python is not installed or not in PATH
        echo Please install Python from https://www.python.org/
        pause
        exit /b 1
    ) else (
        set PYTHON_CMD=py
        set PIP_CMD=py -m pip
    )
) else (
    set PYTHON_CMD=python
    set PIP_CMD=python -m pip
)

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    set PIP_CMD=pip
)

REM Install dependencies if needed
echo Checking dependencies...
%PIP_CMD% show streamlit >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    echo This may take a few minutes...
    echo.
    echo Installing packages one by one (skipping pyarrow if it fails)...
    %PIP_CMD% install torch torchvision
    %PIP_CMD% install matplotlib numpy
    %PIP_CMD% install scikit-learn
    %PIP_CMD% install streamlit
    %PIP_CMD% install tqdm
    %PIP_CMD% install reportlab
    echo.
    echo Attempting to install pyarrow (may fail - that's OK)...
    %PIP_CMD% install pyarrow 2>nul
    if errorlevel 1 (
        echo.
        echo ⚠️  pyarrow installation failed (requires cmake) - this is OK!
        echo Streamlit will work without it, but some features may be limited.
        echo.
    ) else (
        echo ✅ pyarrow installed successfully!
    )
    echo.
    echo ✅ Core dependencies installed!
)

REM Create necessary directories
if not exist "saved" mkdir saved
if not exist "assets" mkdir assets
if not exist "data" mkdir data

REM Run the application
echo.
echo Starting Streamlit Dashboard...
echo The dashboard will open in your browser.
echo Press Ctrl+C to stop the server.
echo.
%PYTHON_CMD% run_app.py

pause

