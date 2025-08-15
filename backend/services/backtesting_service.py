"""
FastAPI service for backtesting trading strategies
"""
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.backtester import Backtester
from core.signal_generator import SignalGenerator
from core.data_processor import DataProcessor
from services.model_service import trained_models
from services.prediction_service import generate_signals
from config.settings import settings
from loguru import logger

router = APIRouter(prefix="/backtest", tags=["backtesting"])


# Pydantic models
class BacktestRequest(BaseModel):
    model_key: str
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    slippage: float = 0.0001
    max_position_size: float = 0.1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class RollingBacktestRequest(BaseModel):
    model_key: str
    window_size: int = 252
    step_size: int = 21
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001


class StrategyComparisonRequest(BaseModel):
    model_keys: List[str]
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001


class MonteCarloRequest(BaseModel):
    model_key: str
    n_simulations: int = 1000
    initial_capital: float = 100000.0
    bootstrap_block_size: int = 21


# Global instances
data_processor = DataProcessor()
signal_generator = SignalGenerator()
backtest_results = {}
backtest_status = {}


@router.post("/run/{model_key}")
async def run_backtest(
        model_key: str,
        background_tasks: BackgroundTasks,
        request: BacktestRequest
) -> Dict[str, Any]:
    """
    Run backtest for a specific model
    """
    if model_key not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    if data_processor.processed_data is None:
        raise HTTPException(status_code=404, detail="No data loaded")

    task_id = f"backtest_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    backtest_status[task_id] = {
        "status": "started",
        "model_key": model_key,
        "start_time": datetime.now().isoformat(),
        "progress": 0
    }

    background_tasks.add_task(
        _run_backtest_background,
        task_id,
        model_key,
        request
    )

    return {
        "success": True,
        "message": f"Backtest started for {model_key}",
        "task_id": task_id
    }


async def _run_backtest_background(task_id: str, model_key: str,
                                   request: BacktestRequest):
    """
    Background task for running backtest
    """
    try:
        backtest_status[task_id]["status"] = "generating_signals"
        backtest_status[task_id]["progress"] = 20

        # Generate signals
        from services.prediction_service import SignalRequest
        signal_request = SignalRequest(
            model_key=model_key,
            threshold_method="adaptive",
            smoothing_window=5,
            apply_filters=True
        )

        signal_response = await generate_signals(model_key, signal_request)

        if not signal_response["success"]:
            raise ValueError("Failed to generate signals for backtesting")

        backtest_status[task_id]["progress"] = 40

        # Prepare data
        data = data_processor.processed_data.copy()

        # Filter by date range if specified
        if request.start_date:
            data = data[data.index >= pd.to_datetime(request.start_date)]
        if request.end_date:
            data = data[data.index <= pd.to_datetime(request.end_date)]

        prices = data['close']

        # Convert signals to pandas Series
        signal_data = signal_response["signals"]
        signal_dates = pd.to_datetime(signal_data["dates"])
        signals = pd.Series(signal_data["signals"], index=signal_dates)

        # Align signals with price data
        common_index = signals.index.intersection(prices.index)
        signals = signals.reindex(common_index).fillna(0)
        prices = prices.reindex(common_index)

        backtest_status[task_id]["status"] = "running_backtest"
        backtest_status[task_id]["progress"] = 60

        # Initialize backtester
        backtester = Backtester(initial_capital=request.initial_capital)

        # Run backtest
        backtest_result = backtester.run_backtest(
            signals=signals,
            prices=prices,
            transaction_cost=request.transaction_cost,
            slippage=request.slippage,
            max_position_size=request.max_position_size,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit
        )

        backtest_status[task_id]["progress"] = 90

        # Store results
        backtest_results[task_id] = {
            "model_key": model_key,
            "backtest_result": backtest_result,
            "request_params": request.dict(),
            "data_period": {
                "start": str(common_index.min()),
                "end": str(common_index.max()),
                "days": len(common_index)
            },
            "created_at": datetime.now().isoformat()
        }

        backtest_status[task_id].update({
            "status": "completed",
            "progress": 100,
            "end_time": datetime.now().isoformat(),
            "final_value": backtest_result["final_value"],
            "total_return": backtest_result["total_return"]
        })

    except Exception as e:
        logger.error(f"Backtest failed for {model_key}: {str(e)}")
        backtest_status[task_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        })


@router.get("/status/{task_id}")
async def get_backtest_status(task_id: str) -> Dict[str, Any]:
    """
    Get backtest status
    """
    if task_id not in backtest_status:
        raise HTTPException(status_code=404, detail="Backtest task not found")

    return backtest_status[task_id]


@router.get("/results/{task_id}")
async def get_backtest_results(task_id: str) -> Dict[str, Any]:
    """
    Get detailed backtest results
    """
    if task_id not in backtest_results:
        raise HTTPException(status_code=404,
                            detail="Backtest results not found")

    results = backtest_results[task_id]
    backtest_result = results["backtest_result"]

    # Convert pandas objects to JSON-serializable format
    portfolio_values = backtest_result["portfolio_values"]
    positions = backtest_result["positions"]
    trades = backtest_result["trades"]

    serializable_result = {
        "model_key": results["model_key"],
        "request_params": results["request_params"],
        "data_period": results["data_period"],
        "performance_summary": {
            "final_value": backtest_result["final_value"],
            "total_return": backtest_result["total_return"],
            "trade_count": backtest_result["trade_count"]
        },
        "performance_metrics": backtest_result["performance_metrics"],
        "portfolio_data": {
            "dates": [str(date) for date in portfolio_values.index],
            "portfolio_values": portfolio_values.tolist(),
            "positions": positions.tolist()
        },
        "trades": trades.to_dict('records') if not trades.empty else [],
        "created_at": results["created_at"]
    }

    return {
        "success": True,
        "backtest_results": serializable_result
    }


@router.post("/rolling/{model_key}")
async def run_rolling_backtest(
        model_key: str,
        background_tasks: BackgroundTasks,
        request: RollingBacktestRequest
) -> Dict[str, Any]:
    """
    Run rolling window backtest to assess strategy stability
    """
    if model_key not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    task_id = f"rolling_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    backtest_status[task_id] = {
        "status": "started",
        "type": "rolling",
        "model_key": model_key,
        "start_time": datetime.now().isoformat(),
        "progress": 0
    }

    background_tasks.add_task(
        _run_rolling_backtest_background,
        task_id,
        model_key,
        request
    )

    return {
        "success": True,
        "message": f"Rolling backtest started for {model_key}",
        "task_id": task_id
    }


async def _run_rolling_backtest_background(task_id: str, model_key: str,
                                           request: RollingBacktestRequest):
    """
    Background task for rolling backtest
    """
    try:
        backtest_status[task_id]["status"] = "running"

        # Generate signals first
        from services.prediction_service import SignalRequest
        signal_request = SignalRequest(model_key=model_key)
        signal_response = await generate_signals(model_key, signal_request)

        if not signal_response["success"]:
            raise ValueError("Failed to generate signals")

        # Prepare data
        data = data_processor.processed_data
        prices = data['close']

        signal_data = signal_response["signals"]
        signals = pd.Series(signal_data["signals"],
                            index=pd.to_datetime(signal_data["dates"]))

        # Align data
        common_index = signals.index.intersection(prices.index)
        signals = signals.reindex(common_index).fillna(0)
        prices = prices.reindex(common_index)

        backtest_status[task_id]["progress"] = 30

        # Run rolling backtest
        backtester = Backtester(initial_capital=request.initial_capital)
        rolling_result = backtester.rolling_backtest(
            signals=signals,
            prices=prices,
            window_size=request.window_size,
            step_size=request.step_size
        )

        backtest_status[task_id]["progress"] = 90

        # Store results
        backtest_results[task_id] = {
            "type": "rolling",
            "model_key": model_key,
            "rolling_result": rolling_result,
            "request_params": request.dict(),
            "created_at": datetime.now().isoformat()
        }

        backtest_status[task_id].update({
            "status": "completed",
            "progress": 100,
            "end_time": datetime.now().isoformat(),
            "stability_metrics": rolling_result["stability_metrics"]
        })

    except Exception as e:
        logger.error(f"Rolling backtest failed: {str(e)}")
        backtest_status[task_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        })


@router.post("/compare")
async def compare_strategies(
        background_tasks: BackgroundTasks,
        request: StrategyComparisonRequest
) -> Dict[str, Any]:
    """
    Compare multiple trading strategies
    """
    # Validate models
    missing_models = [key for key in request.model_keys if
                      key not in trained_models]
    if missing_models:
        raise HTTPException(
            status_code=404,
            detail=f"Models not found: {missing_models}"
        )

    task_id = f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    backtest_status[task_id] = {
        "status": "started",
        "type": "comparison",
        "model_keys": request.model_keys,
        "start_time": datetime.now().isoformat(),
        "progress": 0
    }

    background_tasks.add_task(
        _compare_strategies_background,
        task_id,
        request
    )

    return {
        "success": True,
        "message": f"Strategy comparison started for {len(request.model_keys)} models",
        "task_id": task_id
    }


async def _compare_strategies_background(task_id: str,
                                         request: StrategyComparisonRequest):
    """
    Background task for strategy comparison
    """
    try:
        backtest_status[task_id]["status"] = "running"

        comparison_results = {}

        for i, model_key in enumerate(request.model_keys):
            try:
                # Run backtest for each model
                backtest_request = BacktestRequest(
                    model_key=model_key,
                    initial_capital=request.initial_capital,
                    transaction_cost=request.transaction_cost
                )

                temp_task_id = f"temp_{model_key}_{i}"
                await _run_backtest_background(temp_task_id, model_key,
                                               backtest_request)

                if backtest_status[temp_task_id]["status"] == "completed":
                    comparison_results[model_key] = \
                    backtest_results[temp_task_id]["backtest_result"]

                progress = int(((i + 1) / len(request.model_keys)) * 90)
                backtest_status[task_id]["progress"] = progress

            except Exception as e:
                logger.warning(f"Failed to backtest {model_key}: {str(e)}")
                continue

        if not comparison_results:
            raise ValueError("No successful backtests for comparison")

        # Calculate comparison metrics
        comparison_summary = {}
        metrics = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate",
                   "volatility"]

        for metric in metrics:
            metric_values = {}
            for model_key, result in comparison_results.items():
                if metric in result["performance_metrics"]:
                    metric_values[model_key] = result["performance_metrics"][
                        metric]
                elif metric == "total_return":
                    metric_values[model_key] = result["total_return"]

            if metric_values:
                if metric in ["total_return", "sharpe_ratio", "win_rate"]:
                    best_model = max(metric_values.items(), key=lambda x: x[1])
                else:  # For max_drawdown and volatility, lower is better
                    best_model = min(metric_values.items(),
                                     key=lambda x: abs(x[1]))

                comparison_summary[metric] = {
                    "values": metric_values,
                    "best_model": best_model[0],
                    "best_value": best_model[1]
                }

        # Store results
        backtest_results[task_id] = {
            "type": "comparison",
            "comparison_results": comparison_results,
            "comparison_summary": comparison_summary,
            "request_params": request.dict(),
            "created_at": datetime.now().isoformat()
        }

        backtest_status[task_id].update({
            "status": "completed",
            "progress": 100,
            "end_time": datetime.now().isoformat(),
            "models_compared": len(comparison_results)
        })

    except Exception as e:
        logger.error(f"Strategy comparison failed: {str(e)}")
        backtest_status[task_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        })


@router.post("/montecarlo/{model_key}")
async def run_monte_carlo(
        model_key: str,
        background_tasks: BackgroundTasks,
        request: MonteCarloRequest
) -> Dict[str, Any]:
    """
    Run Monte Carlo simulation for strategy robustness testing
    """
    if model_key not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")

    task_id = f"montecarlo_{model_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    backtest_status[task_id] = {
        "status": "started",
        "type": "monte_carlo",
        "model_key": model_key,
        "n_simulations": request.n_simulations,
        "start_time": datetime.now().isoformat(),
        "progress": 0
    }

    background_tasks.add_task(
        _run_monte_carlo_background,
        task_id,
        model_key,
        request
    )

    return {
        "success": True,
        "message": f"Monte Carlo simulation started for {model_key}",
        "task_id": task_id,
        "n_simulations": request.n_simulations
    }


async def _run_monte_carlo_background(task_id: str, model_key: str,
                                      request: MonteCarloRequest):
    """
    Background task for Monte Carlo simulation
    """
    try:
        backtest_status[task_id]["status"] = "running"
        backtest_status[task_id]["progress"] = 10

        # Generate signals
        from services.prediction_service import SignalRequest
        signal_request = SignalRequest(model_key=model_key)
        signal_response = await generate_signals(model_key, signal_request)

        if not signal_response["success"]:
            raise ValueError("Failed to generate signals")

        # Prepare data
        data = data_processor.processed_data
        prices = data['close']

        signal_data = signal_response["signals"]
        signals = pd.Series(signal_data["signals"],
                            index=pd.to_datetime(signal_data["dates"]))

        # Align data
        common_index = signals.index.intersection(prices.index)
        signals = signals.reindex(common_index).fillna(0)
        prices = prices.reindex(common_index)

        backtest_status[task_id]["progress"] = 30

        # Run Monte Carlo simulation
        backtester = Backtester(initial_capital=request.initial_capital)
        mc_result = backtester.monte_carlo_simulation(
            signals=signals,
            prices=prices,
            n_simulations=request.n_simulations,
            bootstrap_block_size=request.bootstrap_block_size
        )

        backtest_status[task_id]["progress"] = 90

        # Store results
        backtest_results[task_id] = {
            "type": "monte_carlo",
            "model_key": model_key,
            "mc_result": mc_result,
            "request_params": request.dict(),
            "created_at": datetime.now().isoformat()
        }

        backtest_status[task_id].update({
            "status": "completed",
            "progress": 100,
            "end_time": datetime.now().isoformat(),
            "simulations_completed": mc_result["simulations_completed"]
        })

    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {str(e)}")
        backtest_status[task_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now().isoformat()
        })


@router.get("/benchmark/{task_id}")
async def compare_to_benchmark(task_id: str) -> Dict[str, Any]:
    """
    Compare backtest results to buy-and-hold benchmark
    """
    if task_id not in backtest_results:
        raise HTTPException(status_code=404,
                            detail="Backtest results not found")

    try:
        results = backtest_results[task_id]

        if results.get("type") == "comparison":
            raise HTTPException(
                status_code=400,
                detail="Benchmark comparison not available for strategy comparison results"
            )

        backtest_result = results["backtest_result"]

        # Get benchmark performance (buy and hold)
        data = data_processor.processed_data
        start_date = pd.to_datetime(results["data_period"]["start"])
        end_date = pd.to_datetime(results["data_period"]["end"])

        benchmark_data = data[
            (data.index >= start_date) & (data.index <= end_date)]
        benchmark_return = (benchmark_data['close'].iloc[-1] -
                            benchmark_data['close'].iloc[0]) / \
                           benchmark_data['close'].iloc[0]

        # Calculate benchmark metrics
        benchmark_returns = benchmark_data['close'].pct_change().dropna()
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        benchmark_sharpe = (
                                       benchmark_return * 252) / benchmark_volatility if benchmark_volatility > 0 else 0

        # Strategy metrics
        strategy_return = backtest_result["total_return"]
        strategy_metrics = backtest_result["performance_metrics"]

        # Comparison
        comparison = {
            "strategy": {
                "total_return": strategy_return,
                "sharpe_ratio": strategy_metrics.get("sharpe_ratio", 0),
                "volatility": strategy_metrics.get("volatility", 0),
                "max_drawdown": strategy_metrics.get("max_drawdown", 0)
            },
            "benchmark": {
                "total_return": benchmark_return,
                "sharpe_ratio": benchmark_sharpe,
                "volatility": benchmark_volatility,
                "max_drawdown": 0  # Simplified for buy-and-hold
            },
            "outperformance": {
                "excess_return": strategy_return - benchmark_return,
                "sharpe_improvement": strategy_metrics.get("sharpe_ratio",
                                                           0) - benchmark_sharpe,
                "outperformed": strategy_return > benchmark_return
            }
        }

        return {
            "success": True,
            "benchmark_comparison": comparison,
            "period": results["data_period"]
        }

    except Exception as e:
        logger.error(f"Error calculating benchmark comparison: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_backtest_results() -> Dict[str, Any]:
    """
    List all backtest results
    """
    results_summary = {}

    for task_id, result in backtest_results.items():
        summary = {
            "type": result.get("type", "single"),
            "model_key": result.get("model_key"),
            "created_at": result["created_at"]
        }

        if "backtest_result" in result:
            summary.update({
                "final_value": result["backtest_result"]["final_value"],
                "total_return": result["backtest_result"]["total_return"],
                "trade_count": result["backtest_result"]["trade_count"]
            })

        results_summary[task_id] = summary

    return {
        "success": True,
        "backtest_results": results_summary,
        "count": len(backtest_results)
    }


@router.delete("/results/{task_id}")
async def delete_backtest_results(task_id: str) -> Dict[str, Any]:
    """
    Delete backtest results
    """
    if task_id not in backtest_results:
        raise HTTPException(status_code=404,
                            detail="Backtest results not found")

    del backtest_results[task_id]

    if task_id in backtest_status:
        del backtest_status[task_id]

    return {
        "success": True,
        "message": f"Backtest results {task_id} deleted"
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for backtesting service
    """
    return {
        "service": "backtesting_service",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "stored_results": len(backtest_results),
        "active_tasks": len([t for t in backtest_status.values() if
                             t["status"] in ["started", "running"]])
    }