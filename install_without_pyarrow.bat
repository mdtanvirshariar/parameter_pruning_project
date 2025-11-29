@echo off
REM Install dependencies without pyarrow (which requires cmake)
echo ================================================
echo Installing Dependencies (without pyarrow)
echo ================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    py --version >nul 2>&1
    if errorlevel 1 (
        echo Error: Python is not installed or not in PATH
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

echo Installing core packages...
%PIP_CMD% install torch torchvision
%PIP_CMD% install matplotlib numpy
%PIP_CMD% install scikit-learn
%PIP_CMD% install streamlit
%PIP_CMD% install tqdm
%PIP_CMD% install reportlab

echo.
echo ✅ Core packages installed!
echo.
echo Note: pyarrow was skipped (requires cmake to build)
echo Streamlit will work without it, but some features may be limited.
echo.
echo To install pyarrow later (if needed):
echo   %PIP_CMD% install pyarrow
echo.
pause

