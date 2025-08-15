"""
FastAPI service for model training and management
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.arima_model import ARIMAModel
from models.sarima_model import SARIMAModel
from models.garch_model import GARCHModel
from models.ml_models import MLModels
from models.prophet_model import ProphetModel
from core.data_processor import DataProcessor
from core.utils import save_model, load_model
from config.settings import settings, MODEL_CONFIGS
from loguru import logger

router = APIRouter(prefix="/models", tags=["models"])


# Pydantic models for requests
class ModelTrainingRequest(BaseModel):
    model_type: str
    target_column: str = "returns_1d"
    hyperparameter_tune: bool = True
    train_size: float = 0.8
    custom_params: Optional[Dict[str, Any]] = None


class EnsembleRequest(BaseModel):
    model_types: List[str]
    target_column: str = "returns_1d"
    weights: Optional[Dict[str, float]] = None


# Global instances
data_processor = DataProcessor()
trained_models = {}
training_status = {}


@router.post("/train/{model_type}")
async def train_model(
        model_type: str,
        background_tasks: BackgroundTasks,
        request: ModelTrainingRequest
) -> Dict[str, Any]:
    """
    Train a specific model type
    """
    if model_type not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    if data_processor.processed_data is None:
        raise HTTPException(status_code=404,
                            detail="No data loaded. Please fetch data first.")

    # Start training in background
    task_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_status[task_id] = {
        "status": "started",
        "model_type": model_type,
        "start_time": datetime.now().isoformat(),
        "progress": 0
    }

    background_tasks.add_task(
        _train_model_background,
        task_id,
        model_type,
        request
    )

    return {
        "success": True,
        "message": f"Training started for {model_type}",
        "task_id": task_id,
        "model_config": MODEL_CONFIGS[model_type]
    }


async def _train_model_background(task_id: str, model_type: str,
                                  request: ModelTrainingRequest):
    """
    Background task for model training
    """
    try:
        training_status[task_id]["status"] = "training"
        training_status[task_id]["progress"] = 10

        data = data_processor.processed_data
        target_col = request.target_column

        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Split data
        split_idx = int(len(data) * request.train_size)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]

        training_status[task_id]["progress"] = 20

        # Train based on model type
        if model_type == "arima":
            model = ARIMAModel()
            results = model.fit(
                train_data[target_col],
                order=request.custom_params.get(
                    'order') if request.custom_params else None
            )

        elif model_type == "sarima":
            model = SARIMAModel()
            results = model.fit(
                train_data[target_col],
                order=request.custom_params.get(
                    'order') if request.custom_params else None,
                seasonal_order=request.custom_params.get(
                    'seasonal_order') if request.custom_params else None
            )

        elif model_type == "garch":
            model = GARCHModel()
            returns = model.prepare_returns(train_data['close'])
            results = model.fit(
                returns,
                model_type=request.custom_params.get('garch_type',
                                                     'GARCH') if request.custom_params else 'GARCH',
                p=request.custom_params.get(
                    'p') if request.custom_params else None,
                q=request.custom_params.get(
                    'q') if request.custom_params else None
            )

        elif model_type == "prophet":
            model = ProphetModel()
            results = model.fit(
                train_data[target_col],
                hyperparameter_tune=request.hyperparameter_tune
            )

        elif model_type in ["lstm", "gru", "cnn_lstm", "attention_lstm"]:
            model = MLModels()
            feature_groups = data_processor.get_feature_columns(data)
            feature_columns = feature_groups['all_features'][
                              :20]  # Limit features for demo

            results = model.train_neural_network(
                train_data,
                feature_columns,
                target_col,
                model_type=model_type,
                hyperparameter_tune=request.hyperparameter_tune
            )

        else:  # Ensemble ML models
            model = MLModels()
            feature_groups = data_processor.get_feature_columns(data)
            feature_columns = feature_groups['all_features'][:20]

            results = model.train_ensemble_models(
                train_data,
                feature_columns,
                target_col
            )

        training_status[task_id]["progress"] = 80

        if results.get('success', False):
            # Save model
            model_key = f"{model_type}_{target_col}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            trained_models[model_key] = {
                "model": model,
                "results": results,
                "model_type": model_type,
                "target_column": target_col,
                "train_data_shape": train_data.shape,
                "test_data_shape": test_data.shape,
                "created_at": datetime.now().isoformat()
            }

            # Save to disk
            model_path = os.path.join(settings.MODELS_DIR, f"{model_key}.pkl")
            os.makedirs(settings.MODELS_DIR, exist_ok=True)

            save_model(model, model_path, {
                "model_type": model_type,
                "target_column": target_col,
                "results": results,
                "created_at": datetime.now().isoformat()
            })

            training_status[task_id].update({
                "status": "completed",
                "progress": 100,
                "model_key": model_key,
                "results": results,
                "end_time": datetime.now().isoformat()
            })

        else:
            raise ValueError(
                f"Model training failed: {results.get('error', 'Unknown error')}")

    except Exception as e:
        logger.error(f"Training failed for {model_type}: {str(e)}")
        training_status[task_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        })


@router.get("/train/status/{task_id}")
async def get_training_status(task_id: str) -> Dict[str, Any]:
    """
    Get training status for a specific task
    """
    if task_id not in training_status:
        raise HTTPException(status_code=404, detail="Training task not found")

    return training_status[task_id]


@router.get("/trained")
async def list_trained_models() -> Dict[str, Any]:
    """
    List all trained models
    """
    models_info = {}

    for model_key, model_info in trained_models.items():
        models_info[model_key] = {
            "model_type": model_info["model_type"],
            "target_column": model_info["target_column"],
            "created_at": model_info["created_at"],
            "train_data_shape": model_info["train_data_shape"],
            "performance": {
                key: value for key, value in model_info["results"].items()
                if key in ["mse", "mae", "rmse", "aic", "bic",
                           "directional_accuracy", "mape"]
            }
        }

    return {
        "success": True,
        "trained_models": models_info,
        "count": len(trained_models)
    }


@router.get("/trained/{model_key}/details")
async def get_model_details(model_key: str) -> Dict[str, Any]:
    """
    Get detailed information about a trained model
    """
    if model_key not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    model_info = trained_models[model_key]

    # Get model summary if available
    summary = {}
    if hasattr(model_info["model"], 'get_model_summary'):
        summary = model_info["model"].get_model_summary(model_key)

    return {
        "success": True,
        "model_key": model_key,
        "model_info": {
            "model_type": model_info["model_type"],
            "target_column": model_info["target_column"],
            "created_at": model_info["created_at"],
            "train_data_shape": model_info["train_data_shape"],
            "test_data_shape": model_info["test_data_shape"]
        },
        "training_results": model_info["results"],
        "model_summary": summary
    }


@router.post("/ensemble/train")
async def train_ensemble(
        background_tasks: BackgroundTasks,
        request: EnsembleRequest
) -> Dict[str, Any]:
    """
    Train ensemble of multiple models
    """
    if data_processor.processed_data is None:
        raise HTTPException(status_code=404, detail="No data loaded.")

    # Validate model types
    invalid_models = [m for m in request.model_types if m not in MODEL_CONFIGS]
    if invalid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model types: {invalid_models}. Available: {list(MODEL_CONFIGS.keys())}"
        )

    task_id = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_status[task_id] = {
        "status": "started",
        "model_type": "ensemble",
        "models": request.model_types,
        "start_time": datetime.now().isoformat(),
        "progress": 0
    }

    background_tasks.add_task(
        _train_ensemble_background,
        task_id,
        request
    )

    return {
        "success": True,
        "message": f"Ensemble training started with models: {request.model_types}",
        "task_id": task_id
    }


async def _train_ensemble_background(task_id: str, request: EnsembleRequest):
    """
    Background task for ensemble training
    """
    try:
        training_status[task_id]["status"] = "training"

        data = data_processor.processed_data
        target_col = request.target_column

        ensemble_models = {}
        ensemble_predictions = {}

        total_models = len(request.model_types)

        for i, model_type in enumerate(request.model_types):
            try:
                logger.info(f"Training {model_type} for ensemble")

                # Create training request for individual model
                model_request = ModelTrainingRequest(
                    model_type=model_type,
                    target_column=target_col,
                    hyperparameter_tune=False
                    # Skip hyperparameter tuning for ensemble
                )

                # Train individual model (simplified version)
                temp_task_id = f"temp_{model_type}_{i}"
                await _train_model_background(temp_task_id, model_type,
                                              model_request)

                if training_status[temp_task_id]["status"] == "completed":
                    model_key = training_status[temp_task_id]["model_key"]
                    ensemble_models[model_type] = trained_models[model_key][
                        "model"]

                progress = int(((i + 1) / total_models) * 80)
                training_status[task_id]["progress"] = progress

            except Exception as e:
                logger.warning(
                    f"Failed to train {model_type} for ensemble: {str(e)}")
                continue

        if not ensemble_models:
            raise ValueError("No models successfully trained for ensemble")

        # Create ensemble predictions (simplified)
        ensemble_key = f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        trained_models[ensemble_key] = {
            "model": ensemble_models,
            "results": {
                "success": True,
                "model_count": len(ensemble_models),
                "model_types": list(ensemble_models.keys()),
                "ensemble_type": "equal_weight" if not request.weights else "weighted"
            },
            "model_type": "ensemble",
            "target_column": target_col,
            "created_at": datetime.now().isoformat()
        }

        training_status[task_id].update({
            "status": "completed",
            "progress": 100,
            "model_key": ensemble_key,
            "models_trained": list(ensemble_models.keys()),
            "end_time": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Ensemble training failed: {str(e)}")
        training_status[task_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        })


@router.delete("/trained/{model_key}")
async def delete_model(model_key: str) -> Dict[str, Any]:
    """
    Delete a trained model
    """
    if model_key not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    # Remove from memory
    del trained_models[model_key]

    # Remove from disk if exists
    model_path = os.path.join(settings.MODELS_DIR, f"{model_key}.pkl")
    if os.path.exists(model_path):
        os.remove(model_path)

    metadata_path = model_path.replace('.pkl', '_metadata.json')
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    return {
        "success": True,
        "message": f"Model {model_key} deleted successfully"
    }


@router.get("/compare")
async def compare_models(
        model_keys: str = Query(...,
                                description="Comma-separated model keys to compare")
) -> Dict[str, Any]:
    """
    Compare performance of multiple trained models
    """
    model_list = [key.strip() for key in model_keys.split(',')]

    # Validate all models exist
    missing_models = [key for key in model_list if key not in trained_models]
    if missing_models:
        raise HTTPException(
            status_code=404,
            detail=f"Models not found: {missing_models}"
        )

    comparison = {}

    for model_key in model_list:
        model_info = trained_models[model_key]
        results = model_info["results"]

        # Extract comparable metrics
        metrics = {}
        for metric in ["mse", "mae", "rmse", "aic", "bic",
                       "directional_accuracy", "mape", "sharpe_ratio"]:
            if metric in results:
                metrics[metric] = results[metric]

        comparison[model_key] = {
            "model_type": model_info["model_type"],
            "target_column": model_info["target_column"],
            "metrics": metrics,
            "created_at": model_info["created_at"]
        }

    # Determine best model for each metric
    best_models = {}
    all_metrics = set()
    for model_data in comparison.values():
        all_metrics.update(model_data["metrics"].keys())

    for metric in all_metrics:
        metric_values = {}
        for model_key, model_data in comparison.items():
            if metric in model_data["metrics"]:
                metric_values[model_key] = model_data["metrics"][metric]

        if metric_values:
            # Determine if lower or higher is better
            if metric in ["mse", "mae", "rmse", "aic", "bic"]:
                best_model = min(metric_values.items(), key=lambda x: x[1])
            else:  # Higher is better for accuracy, sharpe ratio, etc.
                best_model = max(metric_values.items(), key=lambda x: x[1])

            best_models[metric] = {
                "model": best_model[0],
                "value": best_model[1]
            }

    return {
        "success": True,
        "comparison": comparison,
        "best_models": best_models,
        "models_compared": len(model_list)
    }


@router.get("/hyperparameters/{model_type}")
async def get_hyperparameter_info(model_type: str) -> Dict[str, Any]:
    """
    Get hyperparameter information for a model type
    """
    if model_type not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail="Model type not found")

    config = MODEL_CONFIGS[model_type].copy()

    # Add detailed hyperparameter descriptions
    detailed_params = {
        "arima": {
            "p": "Number of autoregressive terms (1-5)",
            "d": "Degree of differencing (0-2)",
            "q": "Number of moving average terms (1-5)"
        },
        "sarima": {
            "p": "Non-seasonal autoregressive order (1-3)",
            "d": "Non-seasonal differencing order (0-2)",
            "q": "Non-seasonal moving average order (1-3)",
            "P": "Seasonal autoregressive order (0-2)",
            "D": "Seasonal differencing order (0-1)",
            "Q": "Seasonal moving average order (0-2)",
            "s": "Seasonal period (5, 10, 21)"
        },
        "garch": {
            "p": "Number of GARCH terms (1-3)",
            "q": "Number of ARCH terms (1-3)",
            "model_type": "GARCH variant (GARCH, EGARCH, GJR-GARCH)"
        },
        "lstm": {
            "units": "Number of LSTM units per layer",
            "epochs": "Training epochs (50-200)",
            "batch_size": "Batch size (16, 32, 64)",
            "lookback": "Sequence length (30-90)"
        },
        "prophet": {
            "seasonality_mode": "additive or multiplicative",
            "changepoint_prior_scale": "Flexibility of trend (0.001-0.5)"
        }
    }

    config["parameter_details"] = detailed_params.get(model_type, {})

    return {
        "success": True,
        "model_type": model_type,
        "config": config
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for model service
    """
    return {
        "service": "model_service",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "trained_models_count": len(trained_models),
        "active_training_tasks": len(
            [t for t in training_status.values() if t["status"] == "training"])
    }