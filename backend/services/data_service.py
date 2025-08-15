"""
FastAPI service for data operations and preprocessing
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor
from core.utils import validate_data_quality, clean_data
from config.settings import settings
from loguru import logger

router = APIRouter(prefix="/data", tags=["data"])


# Pydantic models for request/response
class DataFetchRequest(BaseModel):
    symbol: str = "^GSPC"
    period: str = "5y"
    interval: str = "1d"


class DataQualityResponse(BaseModel):
    quality_score: float
    total_rows: int
    missing_data_percentage: float
    issues: List[str]
    date_range: Dict[str, Any]


class DataSummaryResponse(BaseModel):
    shape: List[int]
    columns: List[str]
    dtypes: Dict[str, str]
    statistics: Dict[str, Any]
    quality_metrics: DataQualityResponse


# Global data processor instance
data_processor = DataProcessor()


@router.post("/fetch")
async def fetch_data(request: DataFetchRequest) -> Dict[str, Any]:
    """
    Fetch SP500 data from Yahoo Finance
    """
    try:
        logger.info(f"Fetching data for {request.symbol}")

        # Fetch raw data
        raw_data = data_processor.fetch_sp500_data(
            symbol=request.symbol,
            period=request.period,
            interval=request.interval
        )

        # Process data
        processed_results = data_processor.process_data(
            symbol=request.symbol,
            period=request.period
        )

        # Validate data quality
        quality_metrics = validate_data_quality(
            processed_results['processed_data'],
            required_columns=['close', 'volume', 'returns']
        )

        return {
            "success": True,
            "message": f"Successfully fetched {len(raw_data)} records",
            "data_shape": list(processed_results['processed_data'].shape),
            "date_range": {
                "start": str(raw_data.index.min()),
                "end": str(raw_data.index.max())
            },
            "quality_score": quality_metrics['quality_score'],
            "columns": list(processed_results['processed_data'].columns),
            "feature_groups": {
                key: len(value) for key, value in
                processed_results['feature_groups'].items()
            }
        }

    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_data_summary() -> DataSummaryResponse:
    """
    Get summary of currently loaded data
    """
    try:
        if data_processor.processed_data is None:
            raise HTTPException(status_code=404,
                                detail="No data loaded. Please fetch data first.")

        data = data_processor.processed_data

        # Basic statistics
        numeric_data = data.select_dtypes(include=[np.number])
        statistics = {
            "mean": numeric_data.mean().to_dict(),
            "std": numeric_data.std().to_dict(),
            "min": numeric_data.min().to_dict(),
            "max": numeric_data.max().to_dict(),
            "median": numeric_data.median().to_dict()
        }

        # Data quality metrics
        quality_metrics = validate_data_quality(data)

        quality_response = DataQualityResponse(
            quality_score=quality_metrics['quality_score'],
            total_rows=quality_metrics['total_rows'],
            missing_data_percentage=sum([
                info['percentage'] for info in
                quality_metrics['missing_data'].values()
            ]) / len(data.columns),
            issues=quality_metrics['issues'],
            date_range=quality_metrics.get('date_range', {})
        )

        return DataSummaryResponse(
            shape=list(data.shape),
            columns=list(data.columns),
            dtypes={col: str(dtype) for col, dtype in data.dtypes.items()},
            statistics=statistics,
            quality_metrics=quality_response
        )

    except Exception as e:
        logger.error(f"Error getting data summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/price-data")
async def get_price_data(
        start_date: Optional[str] = Query(None,
                                          description="Start date (YYYY-MM-DD)"),
        end_date: Optional[str] = Query(None,
                                        description="End date (YYYY-MM-DD)"),
        columns: Optional[str] = Query("close,volume",
                                       description="Comma-separated column names")
) -> Dict[str, Any]:
    """
    Get price data for specified date range and columns
    """
    try:
        if data_processor.data is None:
            raise HTTPException(status_code=404,
                                detail="No data loaded. Please fetch data first.")

        data = data_processor.data.copy()

        # Filter by date range
        if start_date:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data.index <= pd.to_datetime(end_date)]

        # Select columns
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            available_columns = [col for col in column_list if
                                 col in data.columns]
            if available_columns:
                data = data[available_columns]

        # Convert to JSON-serializable format
        data_dict = {
            "dates": [str(date) for date in data.index],
            "data": data.round(4).to_dict('records')
        }

        return {
            "success": True,
            "data": data_dict,
            "shape": list(data.shape),
            "columns": list(data.columns)
        }

    except Exception as e:
        logger.error(f"Error getting price data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/technical-indicators")
async def get_technical_indicators(
        indicators: Optional[str] = Query("sma_20,rsi,macd",
                                          description="Comma-separated indicator names")
) -> Dict[str, Any]:
    """
    Get technical indicators data
    """
    try:
        if data_processor.processed_data is None:
            raise HTTPException(status_code=404,
                                detail="No processed data available. Please fetch data first.")

        data = data_processor.processed_data

        # Parse requested indicators
        if indicators:
            indicator_list = [ind.strip() for ind in indicators.split(',')]
            available_indicators = [col for col in indicator_list if
                                    col in data.columns]
        else:
            # Default technical indicators
            available_indicators = [col for col in data.columns if
                                    any(x in col for x in
                                        ['sma', 'ema', 'rsi', 'macd', 'bb_',
                                         'atr', 'volatility'])]

        if not available_indicators:
            raise HTTPException(status_code=404,
                                detail="No technical indicators found")

        indicator_data = data[available_indicators].copy()

        # Convert to JSON format
        result_data = {
            "dates": [str(date) for date in indicator_data.index],
            "indicators": indicator_data.round(4).to_dict('series')
        }

        return {
            "success": True,
            "data": result_data,
            "available_indicators": available_indicators,
            "shape": list(indicator_data.shape)
        }

    except Exception as e:
        logger.error(f"Error getting technical indicators: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/feature-groups")
async def get_feature_groups() -> Dict[str, Any]:
    """
    Get categorized feature groups
    """
    try:
        if data_processor.processed_data is None:
            raise HTTPException(status_code=404,
                                detail="No processed data available.")

        feature_groups = data_processor.get_feature_columns(
            data_processor.processed_data)

        return {
            "success": True,
            "feature_groups": {
                group_name: {
                    "columns": columns,
                    "count": len(columns)
                }
                for group_name, columns in feature_groups.items()
            }
        }

    except Exception as e:
        logger.error(f"Error getting feature groups: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clean")
async def clean_data_endpoint(
        fill_method: str = Query("forward",
                                 description="Method for filling missing values"),
        remove_outliers: bool = Query(False,
                                      description="Whether to remove outliers"),
        outlier_threshold: float = Query(3.0,
                                         description="Z-score threshold for outlier removal")
) -> Dict[str, Any]:
    """
    Clean the currently loaded data
    """
    try:
        if data_processor.processed_data is None:
            raise HTTPException(status_code=404,
                                detail="No data loaded. Please fetch data first.")

        original_shape = data_processor.processed_data.shape

        # Clean data
        cleaned_data = clean_data(
            data_processor.processed_data,
            fill_method=fill_method,
            remove_outliers=remove_outliers,
            outlier_threshold=outlier_threshold
        )

        # Update processed data
        data_processor.processed_data = cleaned_data

        # Recalculate quality metrics
        quality_metrics = validate_data_quality(cleaned_data)

        return {
            "success": True,
            "message": "Data cleaned successfully",
            "original_shape": list(original_shape),
            "cleaned_shape": list(cleaned_data.shape),
            "rows_removed": original_shape[0] - cleaned_data.shape[0],
            "quality_score": quality_metrics['quality_score'],
            "cleaning_parameters": {
                "fill_method": fill_method,
                "remove_outliers": remove_outliers,
                "outlier_threshold": outlier_threshold
            }
        }

    except Exception as e:
        logger.error(f"Error cleaning data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics(
        column: Optional[str] = Query(None,
                                      description="Specific column for detailed statistics")
) -> Dict[str, Any]:
    """
    Get detailed statistics for data
    """
    try:
        if data_processor.processed_data is None:
            raise HTTPException(status_code=404, detail="No data loaded.")

        data = data_processor.processed_data

        if column:
            if column not in data.columns:
                raise HTTPException(status_code=404,
                                    detail=f"Column '{column}' not found")

            series = data[column].dropna()

            statistics = {
                "count": len(series),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "25%": float(series.quantile(0.25)),
                "50%": float(series.quantile(0.50)),
                "75%": float(series.quantile(0.75)),
                "max": float(series.max()),
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "missing_values": int(data[column].isnull().sum()),
                "missing_percentage": float(data[column].isnull().mean() * 100)
            }

            return {
                "success": True,
                "column": column,
                "statistics": statistics
            }

        else:
            # Summary statistics for all numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            summary_stats = numeric_data.describe().round(4).to_dict()

            return {
                "success": True,
                "summary_statistics": summary_stats,
                "numeric_columns": list(numeric_data.columns),
                "total_columns": len(data.columns)
            }

    except Exception as e:
        logger.error(f"Error getting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/correlation")
async def get_correlation_matrix(
        method: str = Query("pearson",
                            description="Correlation method: pearson, spearman, kendall"),
        columns: Optional[str] = Query(None,
                                       description="Comma-separated column names")
) -> Dict[str, Any]:
    """
    Get correlation matrix for specified columns
    """
    try:
        if data_processor.processed_data is None:
            raise HTTPException(status_code=404, detail="No data loaded.")

        data = data_processor.processed_data

        # Select columns
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            available_columns = [col for col in column_list if
                                 col in data.columns]
            if not available_columns:
                raise HTTPException(status_code=404,
                                    detail="None of the specified columns found")
            data = data[available_columns]
        else:
            # Use only numeric columns
            data = data.select_dtypes(include=[np.number])

        # Calculate correlation matrix
        corr_matrix = data.corr(method=method)

        return {
            "success": True,
            "correlation_matrix": corr_matrix.round(4).to_dict(),
            "method": method,
            "columns": list(corr_matrix.columns),
            "shape": list(corr_matrix.shape)
        }

    except Exception as e:
        logger.error(f"Error calculating correlation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export")
async def export_data(
        format: str = Query("csv",
                            description="Export format: csv, json, parquet"),
        columns: Optional[str] = Query(None,
                                       description="Comma-separated column names")
) -> Dict[str, Any]:
    """
    Export data in specified format
    """
    try:
        if data_processor.processed_data is None:
            raise HTTPException(status_code=404, detail="No data loaded.")

        data = data_processor.processed_data

        # Select columns if specified
        if columns:
            column_list = [col.strip() for col in columns.split(',')]
            available_columns = [col for col in column_list if
                                 col in data.columns]
            if available_columns:
                data = data[available_columns]

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sp500_data_{timestamp}.{format}"
        filepath = os.path.join(settings.DATA_DIR, filename)

        # Ensure directory exists
        os.makedirs(settings.DATA_DIR, exist_ok=True)

        # Export data
        if format == "csv":
            data.to_csv(filepath)
        elif format == "json":
            data.to_json(filepath, orient='index', date_format='iso')
        elif format == "parquet":
            data.to_parquet(filepath)
        else:
            raise HTTPException(status_code=400,
                                detail="Unsupported format. Use csv, json, or parquet.")

        return {
            "success": True,
            "message": f"Data exported successfully",
            "filename": filename,
            "filepath": filepath,
            "format": format,
            "shape": list(data.shape)
        }

    except Exception as e:
        logger.error(f"Error exporting data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for data service
    """
    return {
        "service": "data_service",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": data_processor.processed_data is not None,
        "data_shape": list(
            data_processor.processed_data.shape) if data_processor.processed_data is not None else None
    }