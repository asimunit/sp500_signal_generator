#!/usr/bin/env python3
"""
Script to run the Streamlit frontend for SP500 Signal Generator
"""
import os
import sys
import subprocess
import requests
import time
from pathlib import Path

# Add frontend directory to Python path
frontend_dir = Path(__file__).parent / "frontend"
sys.path.insert(0, str(frontend_dir))

# Add backend for settings
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from config.settings import settings


def check_backend_connection():
    """Check if the backend API is running"""
    try:
        response = requests.get(
            f"http://{settings.API_HOST}:{settings.API_PORT}/health",
            timeout=5)
        return response.status_code == 200
    except:
        return False


def wait_for_backend(max_wait_time=30):
    """Wait for backend to be available"""
    print("🔍 Checking backend connection...")

    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        if check_backend_connection():
            print("✅ Backend is running and accessible")
            return True

        print("⏳ Waiting for backend to start...")
        time.sleep(2)

    return False


def main():
    """Main function to start the Streamlit frontend"""

    print("🎨 Starting SP500 Signal Generator Frontend...")
    print(f"📊 Configuration:")
    print(f"   - Frontend Host: {settings.STREAMLIT_HOST}")
    print(f"   - Frontend Port: {settings.STREAMLIT_PORT}")
    print(f"   - Backend API: {settings.API_HOST}:{settings.API_PORT}")
    print()

    # Check for required dependencies
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        import requests
        print("✅ All required frontend dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print(
            "Please install all dependencies using: pip install -r requirements.txt")
        sys.exit(1)

    print()

    # Check backend connection
    if not wait_for_backend():
        print("⚠️  Backend API is not responding")
        print("   Please ensure the backend is running:")
        print("   python run_backend.py")
        print()

        response = input(
            "Continue anyway? The frontend will have limited functionality. (y/N): ")
        if response.lower() != 'y':
            print("👋 Exiting...")
            sys.exit(1)

    print()
    print("🔧 Starting Streamlit app...")
    print(
        f"🌐 Frontend will be available at: http://{settings.STREAMLIT_HOST}:{settings.STREAMLIT_PORT}")
    print()
    print("Press Ctrl+C to stop the application")
    print("=" * 60)

    try:
        # Build the streamlit command
        streamlit_cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(frontend_dir / "app.py"),
            "--server.address", settings.STREAMLIT_HOST,
            "--server.port", str(settings.STREAMLIT_PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--theme.primaryColor", "#1f77b4",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6",
            "--theme.textColor", "#262730"
        ]

        # Start Streamlit
        subprocess.run(streamlit_cmd, cwd=frontend_dir)

    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
        print("👋 Goodbye!")

    except Exception as e:
        print(f"\n❌ Error starting Streamlit: {e}")
        sys.exit(1)


def check_streamlit_installation():
    """Check if Streamlit is properly installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False


def install_streamlit():
    """Install Streamlit if not available"""
    print("📦 Installing Streamlit...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Streamlit")
        return False


if __name__ == "__main__":
    # Check if Streamlit is installed
    if not check_streamlit_installation():
        print("⚠️  Streamlit is not installed")
        response = input("Would you like to install it now? (y/N): ")
        if response.lower() == 'y':
            if install_streamlit():
                main()
            else:
                sys.exit(1)
        else:
            print("Please install Streamlit manually: pip install streamlit")
            sys.exit(1)
    else:
        main()