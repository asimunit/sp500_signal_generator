#!/usr/bin/env python3
"""
Script to run the FastAPI backend for SP500 Signal Generator
"""
import os
import sys
import uvicorn
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from config.settings import settings
from core.utils import setup_logging


def main():
    """Main function to start the FastAPI backend server"""

    print("🚀 Starting SP500 Signal Generator Backend...")
    print(f"📊 Configuration:")
    print(f"   - Host: {settings.API_HOST}")
    print(f"   - Port: {settings.API_PORT}")
    print(
        f"   - Environment: {'Development' if settings.API_RELOAD else 'Production'}")
    print(f"   - Log Level: {settings.LOG_LEVEL}")
    print()

    # Setup logging
    setup_logging(settings.LOG_LEVEL, settings.LOG_FILE)

    # Create necessary directories
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)

    print("📁 Created necessary directories:")
    print(f"   - Data: {settings.DATA_DIR}")
    print(f"   - Models: {settings.MODELS_DIR}")
    print(f"   - Results: {settings.RESULTS_DIR}")
    print()

    # Check for required dependencies
    try:
        import fastapi
        import uvicorn
        import pandas
        import numpy
        import plotly
        import yfinance
        import statsmodels
        import sklearn
        import tensorflow
        print("✅ All required dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print(
            "Please install all dependencies using: pip install -r requirements.txt")
        sys.exit(1)

    print()
    print("🔧 Starting API server...")
    print(
        f"🌐 API Documentation will be available at: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print(
        f"📚 ReDoc Documentation will be available at: http://{settings.API_HOST}:{settings.API_PORT}/redoc")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    try:
        # Start the FastAPI server
        uvicorn.run(
            "main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            reload=settings.API_RELOAD,
            log_level=settings.LOG_LEVEL.lower(),
            access_log=True,
            app_dir=str(backend_dir)
        )

    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        print("👋 Goodbye!")

    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()