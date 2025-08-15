"""
FastAPI service for generating predictions and forecasts
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.signal_generator import SignalGenerator
from core.data_processor import DataProcessor
from services.model_service import trained_models
from config.settings import settings
from loguru import logger

router = APIRouter(prefix="/predictions", tags=["predictions"])


# Pydantic models
class PredictionRequest(BaseModel):
    model_key: str
    steps: int = 20
    confidence_level: float = 0.05


class SignalRequest(BaseModel):
    model_key: str
    threshold_method: str = "adaptive"
    smoothing_window: int = 5
    apply_filters: bool = True


class EnsemblePredictionRequest(BaseModel):
    model_keys: List[str]
    steps: int = 20
    ensemble_method: str = "equal_weight"
    custom_weights: Optional[Dict[str, float]] = None


# Global instances
data_processor = DataProcessor()
signal_generator = SignalGenerator()
prediction_cache = {}


@router.post("/forecast/{model_key}")
async def generate_forecast(
        model_key: str,
        request: PredictionRequest
) -> Dict[str, Any]:
    """
    Generate forecast using a specific trained model
    """
    if model_key not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    if data_processor.processed_data is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    try:
        model_info = trained_models[model_key]
        model = model_info["model"]
        model_type = model_info["model_type"]

        logger.info(f"Generating forecast with {model_type} model")

        # Generate forecast based on model type
        if model_type == "arima":
            forecast_result = model.forecast(steps=request.steps,
                                             alpha=request.confidence_level)

        elif model_type == "sarima":
            forecast_result = model.forecast(steps=request.steps,
                                             alpha=request.confidence_level)

        elif model_type == "garch":
            forecast_result = model.forecast_volatility(horizon=request.steps)

        elif model_type == "prophet":
            forecast_result = model.forecast(periods=request.steps)

        elif model_type in ["lstm", "gru", "cnn_lstm", "attention_lstm",
                            "random_forest", "gradient_boosting", "svr"]:
            feature_groups = data_processor.get_feature_columns(
                data_processor.processed_data)
            feature_columns = feature_groups['all_features'][:20]
            forecast_result = model.forecast(model_key,
                                             data_processor.processed_data,
                                             feature_columns, request.steps)

        else:
            raise HTTPException(status_code=400,
                                detail=f"Forecasting not supported for model type: {model_type}")

        if not forecast_result.get('success', False):
            raise ValueError(
                f"Forecast generation failed: {forecast_result.get('error', 'Unknown error')}")

        # Cache the forecast
        cache_key = f"{model_key}_{request.steps}_{datetime.now().strftime('%Y%m%d_%H')}"
        prediction_cache[cache_key] = {
            "forecast_result": forecast_result,
            "model_key": model_key,
            "steps": request.steps,
            "generated_at": datetime.now().isoformat()
        }

        # Calculate additional metrics
        forecast_values = forecast_result.get('forecast_values',
                                              forecast_result.get('forecasts',
                                                                  []))
        if forecast_values:
            forecast_stats = {
                "mean": np.mean(forecast_values),
                "std": np.std(forecast_values),
                "min": np.min(forecast_values),
                "max": np.max(forecast_values),
                "trend": "increasing" if forecast_values[-1] > forecast_values[
                    0] else "decreasing"
            }
        else:
            forecast_stats = {}

        return {
            "success": True,
            "model_key": model_key,
            "model_type": model_type,
            "forecast": forecast_result,
            "forecast_stats": forecast_stats,
            "cache_key": cache_key,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/signals/{model_key}")
async def generate_signals(
        model_key: str,
        request: SignalRequest
) -> Dict[str, Any]:
    """
    Generate trading signals based on model predictions
    """
    if model_key not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # First generate forecast
        forecast_request = PredictionRequest(model_key=model_key, steps=20)
        forecast_response = await generate_forecast(model_key,
                                                    forecast_request)

        if not forecast_response["success"]:
            raise ValueError("Failed to generate forecast for signals")

        forecast_result = forecast_response["forecast"]
        forecast_values = forecast_result.get('forecast_values',
                                              forecast_result.get('forecasts',
                                                                  []))

        if not forecast_values:
            raise ValueError(
                "No forecast values available for signal generation")

        # Convert to pandas Series
        forecast_dates = forecast_result.get('forecast_dates', [])
        if forecast_dates:
            forecast_series = pd.Series(forecast_values,
                                        index=pd.to_datetime(forecast_dates))
        else:
            forecast_series = pd.Series(forecast_values)

        # Calculate volatility for dynamic thresholds
        if data_processor.processed_data is not None:
            volatility = data_processor.processed_data['returns'].rolling(
                20).std().fillna(0.02)
            current_volatility = volatility.iloc[-20:] if len(
                volatility) >= 20 else volatility
        else:
            current_volatility = pd.Series([0.02] * len(forecast_series),
                                           index=forecast_series.index)

        # Generate dynamic thresholds
        thresholds = signal_generator.calculate_dynamic_thresholds(
            forecast_series,
            current_volatility,
            method=request.threshold_method
        )

        # Generate basic signals
        signals = signal_generator.generate_basic_signals(
            forecast_series,
            thresholds,
            smoothing_window=request.smoothing_window
        )

        # Apply filters if requested
        if request.apply_filters and data_processor.processed_data is not None:
            prices = data_processor.processed_data['close']

            # Apply momentum filter
            signals = signal_generator.apply_momentum_filter(signals, prices)

            # Apply volatility filter
            signals = signal_generator.apply_volatility_filter(signals,
                                                               current_volatility)

        # Generate signal summary
        signal_summary = signal_generator.generate_signal_summary(signals)

        # Convert signals to JSON-serializable format
        signals_data = {
            "dates": [str(date) for date in signals.index],
            "signals": signals.tolist(),
            "signal_names": {-1: "SELL", 0: "HOLD", 1: "BUY"}
        }

        return {
            "success": True,
            "model_key": model_key,
            "signals": signals_data,
            "thresholds": thresholds,
            "signal_summary": signal_summary,
            "parameters": {
                "threshold_method": request.threshold_method,
                "smoothing_window": request.smoothing_window,
                "filters_applied": request.apply_filters
            },
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ensemble/forecast")
async def generate_ensemble_forecast(
        request: EnsemblePredictionRequest
) -> Dict[str, Any]:
    """
    Generate ensemble forecast from multiple models
    """
    # Validate all models exist
    missing_models = [key for key in request.model_keys if
                      key not in trained_models]
    if missing_models:
        raise HTTPException(
            status_code=404,
            detail=f"Models not found: {missing_models}"
        )

    try:
        individual_forecasts = {}

        # Generate individual forecasts
        for model_key in request.model_keys:
            forecast_request = PredictionRequest(model_key=model_key,
                                                 steps=request.steps)
            forecast_response = await generate_forecast(model_key,
                                                        forecast_request)

            if forecast_response["success"]:
                forecast_result = forecast_response["forecast"]
                forecast_values = forecast_result.get('forecast_values',
                                                      forecast_result.get(
                                                          'forecasts', []))
                individual_forecasts[model_key] = forecast_values
            else:
                logger.warning(f"Failed to generate forecast for {model_key}")

        if not individual_forecasts:
            raise ValueError("No successful forecasts generated for ensemble")

        # Combine forecasts
        if request.ensemble_method == "equal_weight":
            weights = {key: 1.0 / len(individual_forecasts) for key in
                       individual_forecasts.keys()}
        elif request.ensemble_method == "custom_weight" and request.custom_weights:
            weights = request.custom_weights
        else:
            # Performance-weighted (simplified)
            weights = {}
            total_weight = 0
            for model_key in individual_forecasts.keys():
                model_info = trained_models[model_key]
                # Use inverse MSE as weight if available
                mse = model_info["results"].get("mse", 1.0)
                weight = 1.0 / (
                            mse + 0.001)  # Add small value to avoid division by zero
                weights[model_key] = weight
                total_weight += weight

            # Normalize weights
            weights = {key: w / total_weight for key, w in weights.items()}

        # Calculate ensemble forecast
        ensemble_forecast = np.zeros(request.steps)
        for model_key, forecast_values in individual_forecasts.items():
            weight = weights.get(model_key, 0)
            if len(forecast_values) >= request.steps:
                ensemble_forecast += np.array(
                    forecast_values[:request.steps]) * weight

        # Calculate ensemble statistics
        forecast_variance = np.zeros(request.steps)
        for model_key, forecast_values in individual_forecasts.items():
            if len(forecast_values) >= request.steps:
                diff = np.array(
                    forecast_values[:request.steps]) - ensemble_forecast
                forecast_variance += diff ** 2 * weights.get(model_key, 0)

        forecast_std = np.sqrt(forecast_variance)

        # Create confidence intervals (approximation)
        z_score = 1.96  # 95% confidence
        lower_bounds = ensemble_forecast - z_score * forecast_std
        upper_bounds = ensemble_forecast + z_score * forecast_std

        # Create forecast dates
        last_date = data_processor.processed_data.index[
            -1] if data_processor.processed_data is not None else datetime.now()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1),
                                       periods=request.steps, freq='D')

        ensemble_result = {
            "success": True,
            "forecast_values": ensemble_forecast.tolist(),
            "lower_bounds": lower_bounds.tolist(),
            "upper_bounds": upper_bounds.tolist(),
            "forecast_dates": [str(date) for date in forecast_dates],
            "forecast_std": forecast_std.tolist()
        }

        return {
            "success": True,
            "ensemble_forecast": ensemble_result,
            "individual_forecasts": {key: values[:request.steps] for
                                     key, values in
                                     individual_forecasts.items()},
            "weights": weights,
            "ensemble_method": request.ensemble_method,
            "models_used": list(individual_forecasts.keys()),
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating ensemble forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rolling/{model_key}")
async def generate_rolling_forecast(
        model_key: str,
        train_size: int = Query(252, description="Training window size"),
        forecast_horizon: int = Query(1, description="Forecast horizon"),
        step_size: int = Query(21, description="Step size for rolling window")
) -> Dict[str, Any]:
    """
    Generate rolling window forecasts for backtesting
    """
    if model_key not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    if data_processor.processed_data is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    try:
        model_info = trained_models[model_key]
        model = model_info["model"]
        model_type = model_info["model_type"]
        target_column = model_info["target_column"]

        data = data_processor.processed_data[target_column]

        logger.info(f"Generating rolling forecasts with {model_type} model")

        # Generate rolling forecasts based on model type
        if model_type == "arima":
            rolling_result = model.rolling_forecast(data, train_size,
                                                    forecast_horizon)

        elif model_type == "sarima":
            rolling_result = model.rolling_forecast(data, train_size,
                                                    forecast_horizon)

        elif model_type == "garch":
            rolling_result = model.rolling_volatility_forecast(data,
                                                               train_size,
                                                               forecast_horizon)

        elif model_type == "prophet":
            rolling_result = model.rolling_forecast(data, train_size,
                                                    forecast_horizon)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Rolling forecast not supported for model type: {model_type}"
            )

        # Convert results to JSON-serializable format
        results_df = rolling_result.get('results', pd.DataFrame())
        if not results_df.empty:
            rolling_data = {
                "dates": [str(date) for date in results_df.index],
                "forecasts": results_df.get('forecast', results_df.get(
                    'volatility_forecast', [])).tolist(),
                "actuals": results_df.get('actual',
                                          results_df.get('actual_volatility',
                                                         [])).tolist(),
                "errors": results_df.get('error',
                                         results_df.get('forecast_error',
                                                        [])).tolist()
            }
        else:
            rolling_data = {"dates": [], "forecasts": [], "actuals": [],
                            "errors": []}

        # Extract performance metrics
        performance_metrics = {
            key: value for key, value in rolling_result.items()
            if key in ["mse", "mae", "rmse", "directional_accuracy",
                       "correlation", "forecast_count"]
        }

        return {
            "success": True,
            "model_key": model_key,
            "model_type": model_type,
            "rolling_forecasts": rolling_data,
            "performance_metrics": performance_metrics,
            "parameters": {
                "train_size": train_size,
                "forecast_horizon": forecast_horizon,
                "step_size": step_size
            },
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error generating rolling forecasts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache")
async def list_cached_predictions() -> Dict[str, Any]:
    """
    List all cached predictions
    """
    cache_info = {}

    for cache_key, cache_data in prediction_cache.items():
        cache_info[cache_key] = {
            "model_key": cache_data["model_key"],
            "steps": cache_data["steps"],
            "generated_at": cache_data["generated_at"]
        }

    return {
        "success": True,
        "cached_predictions": cache_info,
        "count": len(prediction_cache)
    }


@router.get("/cache/{cache_key}")
async def get_cached_prediction(cache_key: str) -> Dict[str, Any]:
    """
    Retrieve a cached prediction
    """
    if cache_key not in prediction_cache:
        raise HTTPException(status_code=404,
                            detail="Cached prediction not found")

    return {
        "success": True,
        "cached_prediction": prediction_cache[cache_key]
    }


@router.delete("/cache/{cache_key}")
async def delete_cached_prediction(cache_key: str) -> Dict[str, Any]:
    """
    Delete a cached prediction
    """
    if cache_key not in prediction_cache:
        raise HTTPException(status_code=404,
                            detail="Cached prediction not found")

    del prediction_cache[cache_key]

    return {
        "success": True,
        "message": f"Cached prediction {cache_key} deleted"
    }


@router.post("/batch")
async def batch_predictions(
        background_tasks: BackgroundTasks,
        model_keys: List[str],
        steps: int = Query(20, description="Forecast steps for each model")
) -> Dict[str, Any]:
    """
    Generate predictions for multiple models in batch
    """
    # Validate models
    missing_models = [key for key in model_keys if key not in trained_models]
    if missing_models:
        raise HTTPException(
            status_code=404,
            detail=f"Models not found: {missing_models}"
        )

    task_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Start batch processing in background
    background_tasks.add_task(
        _process_batch_predictions,
        task_id,
        model_keys,
        steps
    )

    return {
        "success": True,
        "message": "Batch prediction started",
        "task_id": task_id,
        "models": model_keys,
        "steps": steps
    }


async def _process_batch_predictions(task_id: str, model_keys: List[str],
                                     steps: int):
    """
    Background task for batch predictions
    """
    batch_results = {}

    for model_key in model_keys:
        try:
            request = PredictionRequest(model_key=model_key, steps=steps)
            result = await generate_forecast(model_key, request)
            batch_results[model_key] = result
        except Exception as e:
            logger.error(f"Batch prediction failed for {model_key}: {str(e)}")
            batch_results[model_key] = {"success": False, "error": str(e)}

    # Cache batch results
    prediction_cache[task_id] = {
        "type": "batch",
        "results": batch_results,
        "generated_at": datetime.now().isoformat()
    }


@router.get("/accuracy/{model_key}")
async def calculate_prediction_accuracy(
        model_key: str,
        lookback_days: int = Query(60,
                                   description="Days to look back for accuracy calculation")
) -> Dict[str, Any]:
    """
    Calculate historical prediction accuracy for a model
    """
    if model_key not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    if data_processor.processed_data is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    try:
        # Generate rolling forecasts for accuracy assessment
        rolling_result = await generate_rolling_forecast(
            model_key=model_key,
            train_size=lookback_days,
            forecast_horizon=1
        )

        if not rolling_result["success"]:
            raise ValueError(
                "Failed to generate rolling forecasts for accuracy calculation")

        rolling_data = rolling_result["rolling_forecasts"]
        forecasts = np.array(rolling_data["forecasts"])
        actuals = np.array(rolling_data["actuals"])

        if len(forecasts) == 0:
            raise ValueError("No forecasts available for accuracy calculation")

        # Calculate various accuracy metrics
        errors = forecasts - actuals
        absolute_errors = np.abs(errors)

        accuracy_metrics = {
            "mean_absolute_error": float(np.mean(absolute_errors)),
            "root_mean_squared_error": float(np.sqrt(np.mean(errors ** 2))),
            "mean_absolute_percentage_error": float(
                np.mean(np.abs(errors / actuals)) * 100),
            "directional_accuracy": float(np.mean(
                np.sign(forecasts[1:] - forecasts[:-1]) == np.sign(
                    actuals[1:] - actuals[:-1]))),
            "correlation": float(np.corrcoef(forecasts, actuals)[0, 1]) if len(
                forecasts) > 1 else 0,
            "bias": float(np.mean(errors)),
            "forecast_count": len(forecasts)
        }

        # Calculate accuracy by time periods
        if len(forecasts) >= 30:
            recent_period = min(30, len(forecasts))
            recent_mae = float(np.mean(absolute_errors[-recent_period:]))
            older_mae = float(
                np.mean(absolute_errors[:-recent_period])) if len(
                forecasts) > recent_period else recent_mae

            accuracy_metrics["recent_30d_mae"] = recent_mae
            accuracy_metrics["older_period_mae"] = older_mae
            accuracy_metrics[
                "accuracy_trend"] = "improving" if recent_mae < older_mae else "declining"

        return {
            "success": True,
            "model_key": model_key,
            "accuracy_metrics": accuracy_metrics,
            "evaluation_period": lookback_days,
            "calculated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error calculating accuracy: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for prediction service
    """
    return {
        "service": "prediction_service",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cached_predictions": len(prediction_cache),
        "available_models": len(trained_models)
    }