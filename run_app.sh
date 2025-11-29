#!/bin/bash
# Unix/Linux/Mac script to run the application

echo "================================================"
echo "Parameter Pruning Dashboard Launcher"
echo "================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check and install dependencies
echo "Checking dependencies..."
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p saved assets data

# Run the application
echo ""
echo "Starting Streamlit Dashboard..."
echo "The dashboard will open in your browser."
echo "Press Ctrl+C to stop the server."
echo ""
python3 run_app.py

