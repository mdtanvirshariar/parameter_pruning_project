"""
Unified startup script to run both frontend and backend together.
This script can start the Streamlit app and optionally run backend services.
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def install_dependencies():
    """Install dependencies using pip"""
    print("📦 Installing dependencies...")
    print("This may take a few minutes. Please wait...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Dependencies installed successfully!")
            return True
        else:
            print(f"❌ Error installing dependencies:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import torch
        import matplotlib
        import numpy
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nAttempting to install dependencies automatically...")
        if install_dependencies():
            # Try importing again
            try:
                import streamlit
                import torch
                import matplotlib
                import numpy
                print("✅ Dependencies are now installed!")
                return True
            except ImportError as e2:
                print(f"❌ Still missing: {e2}")
                print("\nPlease install manually:")
                print(f"  {sys.executable} -m pip install -r requirements.txt")
                return False
        else:
            print("\nPlease install manually:")
            print(f"  {sys.executable} -m pip install -r requirements.txt")
            return False

def create_directories():
    """Create necessary directories"""
    dirs = ['saved', 'assets', 'data']
    for d in dirs:
        Path(d).mkdir(exist_ok=True)
    print("✅ Directories created/verified")

def run_streamlit():
    """Run the Streamlit app"""
    print("\n" + "="*50)
    print("🚀 Starting Streamlit Dashboard...")
    print("="*50)
    print("\nThe dashboard will open in your browser automatically.")
    print("If it doesn't, navigate to: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server.\n")
    
    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("="*50)
    print("🔬 Parameter Pruning Dashboard Launcher")
    print("="*50)
    print()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run Streamlit
    run_streamlit()

if __name__ == "__main__":
    main()

