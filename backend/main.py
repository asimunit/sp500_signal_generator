"""
Main FastAPI application for SP500 Signal Generator
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import services
from services.data_service import router as data_router
from services.model_service import router as model_router
from services.prediction_service import router as prediction_router
from services.backtesting_service import router as backtesting_router

# Import configuration
from config.settings import settings
from core.utils import setup_logging

# Setup logging
setup_logging(settings.LOG_LEVEL, settings.LOG_FILE)

# Create FastAPI app
app = FastAPI(
    title="SP500 Signal Generator API",
    description="""
    A comprehensive API for statistical signal generation using time series forecasting and backtesting.

    Features:
    - Multiple time series models (ARIMA, SARIMA, GARCH, LSTM, Prophet)
    - Dynamic signal generation with adaptive thresholds
    - Comprehensive backtesting with performance metrics
    - Rolling validation and Monte Carlo simulation
    - Real-time data processing and visualization
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data_router)
app.include_router(model_router)
app.include_router(prediction_router)
app.include_router(backtesting_router)


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "SP500 Signal Generator API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "data": "/data",
            "models": "/models",
            "predictions": "/predictions",
            "backtesting": "/backtest",
            "docs": "/docs",
            "health": "/health"
        },
        "description": "Statistical signal generation using time series forecasting and backtesting"
    }


@app.get("/health")
async def health_check():
    """
    Overall health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "data_service": "active",
            "model_service": "active",
            "prediction_service": "active",
            "backtesting_service": "active"
        },
        "configuration": {
            "api_host": settings.API_HOST,
            "api_port": settings.API_PORT,
            "data_symbol": settings.SP500_SYMBOL,
            "forecast_horizon": settings.FORECAST_HORIZON
        }
    }


@app.get("/status")
async def get_system_status():
    """
    Get detailed system status
    """
    # Check if services are responding
    service_status = {}

    try:
        # Import and check data processor
        from core.data_processor import DataProcessor
        data_processor = DataProcessor()
        service_status["data_processor"] = {
            "status": "active",
            "data_loaded": data_processor.processed_data is not None
        }
    except Exception as e:
        service_status["data_processor"] = {
            "status": "error",
            "error": str(e)
        }

    try:
        # Check trained models
        from services.model_service import trained_models
        service_status["model_service"] = {
            "status": "active",
            "trained_models": len(trained_models)
        }
    except Exception as e:
        service_status["model_service"] = {
            "status": "error",
            "error": str(e)
        }

    try:
        # Check prediction cache
        from services.prediction_service import prediction_cache
        service_status["prediction_service"] = {
            "status": "active",
            "cached_predictions": len(prediction_cache)
        }
    except Exception as e:
        service_status["prediction_service"] = {
            "status": "error",
            "error": str(e)
        }

    try:
        # Check backtest results
        from services.backtesting_service import backtest_results
        service_status["backtesting_service"] = {
            "status": "active",
            "stored_results": len(backtest_results)
        }
    except Exception as e:
        service_status["backtesting_service"] = {
            "status": "error",
            "error": str(e)
        }

    return {
        "system_status": "operational",
        "timestamp": datetime.now().isoformat(),
        "services": service_status,
        "memory_info": _get_memory_info(),
        "disk_info": _get_disk_info()
    }


def _get_memory_info():
    """
    Get memory usage information
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "used_percent": memory.percent
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


def _get_disk_info():
    """
    Get disk usage information
    """
    try:
        import psutil
        disk = psutil.disk_usage('/')
        return {
            "total_gb": round(disk.total / (1024 ** 3), 2),
            "free_gb": round(disk.free / (1024 ** 3), 2),
            "used_percent": round((disk.used / disk.total) * 100, 1)
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/config")
async def get_configuration():
    """
    Get current configuration settings
    """
    return {
        "api_configuration": {
            "host": settings.API_HOST,
            "port": settings.API_PORT,
            "reload": settings.API_RELOAD
        },
        "data_configuration": {
            "symbol": settings.SP500_SYMBOL,
            "period": settings.DATA_PERIOD,
            "interval": settings.DATA_INTERVAL
        },
        "model_configuration": {
            "forecast_horizon": settings.FORECAST_HORIZON,
            "train_test_split": settings.TRAIN_TEST_SPLIT,
            "validation_window": settings.VALIDATION_WINDOW
        },
        "signal_configuration": {
            "volatility_threshold": settings.VOLATILITY_THRESHOLD,
            "return_threshold": settings.RETURN_THRESHOLD,
            "signal_smoothing_window": settings.SIGNAL_SMOOTHING_WINDOW
        },
        "backtesting_configuration": {
            "initial_capital": settings.INITIAL_CAPITAL,
            "transaction_cost": settings.TRANSACTION_COST,
            "max_position_size": settings.MAX_POSITION_SIZE
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """
    HTTP exception handler
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler
    """
    print("🚀 SP500 Signal Generator API starting up...")
    print(f"📊 API Host: {settings.API_HOST}:{settings.API_PORT}")
    print(f"📈 Target Symbol: {settings.SP500_SYMBOL}")
    print(f"🔮 Forecast Horizon: {settings.FORECAST_HORIZON} days")
    print(f"📁 Data Directory: {settings.DATA_DIR}")
    print(f"🤖 Models Directory: {settings.MODELS_DIR}")

    # Create directories if they don't exist
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    os.makedirs(settings.RESULTS_DIR, exist_ok=True)

    print("✅ SP500 Signal Generator API ready!")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler
    """
    print("🔄 SP500 Signal Generator API shutting down...")
    print("✅ Shutdown complete!")


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )