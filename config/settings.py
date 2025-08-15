"""
Configuration settings for SP500 Signal Generator
"""
from pydantic_settings import BaseSettings
from typing import List, Dict, Any
import os


class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # Streamlit Configuration
    STREAMLIT_HOST: str = "localhost"
    STREAMLIT_PORT: int = 8501

    # Data Configuration
    SP500_SYMBOL: str = "^GSPC"
    DATA_PERIOD: str = "5y"  # 5 years of historical data
    DATA_INTERVAL: str = "1d"  # Daily data

    # Model Configuration
    FORECAST_HORIZON: int = 20  # 20-day forecast
    TRAIN_TEST_SPLIT: float = 0.8
    VALIDATION_WINDOW: int = 252  # 1 year rolling window

    # ARIMA/SARIMA Parameters
    ARIMA_MAX_P: int = 5
    ARIMA_MAX_D: int = 2
    ARIMA_MAX_Q: int = 5
    SARIMA_SEASONAL_PERIODS: List[int] = [5, 10,
                                          21]  # Weekly, bi-weekly, monthly

    # GARCH Parameters
    GARCH_MAX_P: int = 3
    GARCH_MAX_Q: int = 3

    # ML Model Parameters
    LSTM_UNITS: List[int] = [50, 100, 150]
    LSTM_EPOCHS: int = 100
    LSTM_BATCH_SIZE: int = 32
    LSTM_LOOKBACK: int = 60

    # Prophet Parameters
    PROPHET_SEASONALITY_MODE: str = "multiplicative"
    PROPHET_CHANGEPOINT_PRIOR_SCALE: float = 0.05

    # Signal Generation
    VOLATILITY_THRESHOLD: float = 0.02
    RETURN_THRESHOLD: float = 0.01
    SIGNAL_SMOOTHING_WINDOW: int = 5

    # Backtesting
    INITIAL_CAPITAL: float = 100000.0
    TRANSACTION_COST: float = 0.001  # 0.1% transaction cost

    # Risk Management
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    STOP_LOSS: float = 0.05  # 5% stop loss
    TAKE_PROFIT: float = 0.10  # 10% take profit

    # File Paths
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    RESULTS_DIR: str = "results"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "sp500_signals.log"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Model configurations
MODEL_CONFIGS = {
    "arima": {
        "name": "ARIMA",
        "description": "AutoRegressive Integrated Moving Average",
        "hyperparameters": ["p", "d", "q"],
        "color": "#1f77b4"
    },
    "sarima": {
        "name": "SARIMA",
        "description": "Seasonal ARIMA",
        "hyperparameters": ["p", "d", "q", "P", "D", "Q", "s"],
        "color": "#ff7f0e"
    },
    "garch": {
        "name": "GARCH",
        "description": "Generalized Autoregressive Conditional Heteroskedasticity",
        "hyperparameters": ["p", "q"],
        "color": "#2ca02c"
    },
    "lstm": {
        "name": "LSTM",
        "description": "Long Short-Term Memory Neural Network",
        "hyperparameters": ["units", "epochs", "batch_size", "lookback"],
        "color": "#d62728"
    },
    "prophet": {
        "name": "Prophet",
        "description": "Facebook Prophet Forecasting",
        "hyperparameters": ["seasonality_mode", "changepoint_prior_scale"],
        "color": "#9467bd"
    }
}

# Performance metrics configuration
METRICS_CONFIG = {
    "directional_accuracy": {
        "name": "Directional Accuracy",
        "description": "Percentage of correct direction predictions",
        "format": "percentage",
        "target": 0.72
    },
    "mse": {
        "name": "Mean Squared Error",
        "description": "Average squared prediction errors",
        "format": "float",
        "target": None
    },
    "mae": {
        "name": "Mean Absolute Error",
        "description": "Average absolute prediction errors",
        "format": "float",
        "target": None
    },
    "sharpe_ratio": {
        "name": "Sharpe Ratio",
        "description": "Risk-adjusted return metric",
        "format": "float",
        "target": 1.5
    },
    "max_drawdown": {
        "name": "Maximum Drawdown",
        "description": "Largest peak-to-trough decline",
        "format": "percentage",
        "target": -0.15
    }
}

# Initialize settings
settings = Settings()