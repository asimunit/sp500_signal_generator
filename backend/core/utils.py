"""
General utility functions for SP500 Signal Generator
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import pickle
import os
from datetime import datetime, timedelta
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    """
    logger.remove()  # Remove default handler

    # Console handler
    logger.add(
        sink=lambda message: print(message, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # File handler
    if log_file:
        logger.add(
            sink=log_file,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB"
        )


def validate_data_quality(data: pd.DataFrame,
                          required_columns: List[str] = None) -> Dict[
    str, Any]:
    """
    Validate data quality and return quality metrics
    """
    quality_report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_data': {},
        'data_types': {},
        'date_range': {},
        'outliers': {},
        'quality_score': 0.0,
        'issues': []
    }

    if required_columns:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            quality_report['issues'].append(
                f"Missing required columns: {missing_columns}")

    # Check missing data
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        missing_pct = (missing_count / len(data)) * 100
        quality_report['missing_data'][col] = {
            'count': missing_count,
            'percentage': missing_pct
        }

        if missing_pct > 5:
            quality_report['issues'].append(
                f"Column '{col}' has {missing_pct:.1f}% missing data")

    # Check data types
    for col in data.columns:
        quality_report['data_types'][col] = str(data[col].dtype)

    # Date range analysis if index is datetime
    if isinstance(data.index, pd.DatetimeIndex):
        quality_report['date_range'] = {
            'start': data.index.min(),
            'end': data.index.max(),
            'frequency': pd.infer_freq(data.index),
            'gaps': len(pd.date_range(data.index.min(), data.index.max(),
                                      freq='D')) - len(data)
        }

    # Outlier detection for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].notna().sum() > 0:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((data[col] < lower_bound) | (
                        data[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(data)) * 100

            quality_report['outliers'][col] = {
                'count': outliers,
                'percentage': outlier_pct,
                'bounds': [lower_bound, upper_bound]
            }

    # Calculate overall quality score
    missing_penalty = sum([info['percentage'] for info in
                           quality_report['missing_data'].values()]) / len(
        data.columns)
    issue_penalty = len(quality_report['issues']) * 10
    quality_report['quality_score'] = max(0,
                                          100 - missing_penalty - issue_penalty)

    return quality_report


def clean_data(data: pd.DataFrame,
               fill_method: str = 'forward',
               remove_outliers: bool = False,
               outlier_threshold: float = 3.0) -> pd.DataFrame:
    """
    Clean and preprocess data
    """
    cleaned_data = data.copy()

    # Handle missing values
    if fill_method == 'forward':
        cleaned_data = cleaned_data.fillna(method='ffill')
    elif fill_method == 'backward':
        cleaned_data = cleaned_data.fillna(method='bfill')
    elif fill_method == 'interpolate':
        cleaned_data = cleaned_data.interpolate()
    elif fill_method == 'drop':
        cleaned_data = cleaned_data.dropna()

    # Remove outliers if requested
    if remove_outliers:
        numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if cleaned_data[col].notna().sum() > 0:
                z_scores = np.abs(
                    (cleaned_data[col] - cleaned_data[col].mean()) /
                    cleaned_data[col].std())
                cleaned_data = cleaned_data[z_scores < outlier_threshold]

    logger.info(f"Data cleaned. Shape: {data.shape} -> {cleaned_data.shape}")
    return cleaned_data


def calculate_returns(prices: pd.Series,
                      return_type: str = 'simple',
                      periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Calculate various types of returns
    """
    returns_df = pd.DataFrame(index=prices.index)

    for period in periods:
        if return_type == 'simple':
            returns_df[f'return_{period}d'] = prices.pct_change(period)
        elif return_type == 'log':
            returns_df[f'log_return_{period}d'] = np.log(
                prices / prices.shift(period))
        elif return_type == 'forward':
            returns_df[f'forward_return_{period}d'] = prices.shift(
                -period) / prices - 1

    return returns_df


def calculate_volatility_metrics(returns: pd.Series,
                                 windows: List[int] = [10, 20,
                                                       60]) -> pd.DataFrame:
    """
    Calculate various volatility metrics
    """
    vol_df = pd.DataFrame(index=returns.index)

    for window in windows:
        # Historical volatility (annualized)
        vol_df[f'volatility_{window}d'] = returns.rolling(
            window).std() * np.sqrt(252)

        # Realized volatility (using squared returns)
        vol_df[f'realized_vol_{window}d'] = np.sqrt(
            returns.rolling(window).apply(lambda x: (x ** 2).sum()))

        # Parkinson volatility (using high-low if available)
        # This would require OHLC data, skipping for now

        # GARCH-like volatility (exponentially weighted)
        vol_df[f'ewm_vol_{window}d'] = returns.ewm(
            span=window).std() * np.sqrt(252)

    return vol_df


def detect_regime_changes(data: pd.Series,
                          method: str = 'variance',
                          window: int = 60) -> pd.Series:
    """
    Detect regime changes in time series data
    """
    if method == 'variance':
        # Variance-based regime detection
        rolling_var = data.rolling(window).var()
        regime_threshold = rolling_var.median()
        regimes = pd.Series('normal', index=data.index)
        regimes[rolling_var > regime_threshold * 2] = 'high_volatility'
        regimes[rolling_var < regime_threshold * 0.5] = 'low_volatility'

    elif method == 'trend':
        # Trend-based regime detection
        rolling_mean = data.rolling(window).mean()
        trend = rolling_mean.diff()
        regimes = pd.Series('sideways', index=data.index)
        regimes[trend > trend.std()] = 'uptrend'
        regimes[trend < -trend.std()] = 'downtrend'

    else:
        regimes = pd.Series('normal', index=data.index)

    return regimes


def save_model(model: Any, filepath: str, metadata: Dict[str, Any] = None):
    """
    Save model with metadata
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    # Save metadata
    if metadata:
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model with metadata
    """
    # Load model
    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    # Load metadata
    metadata_path = filepath.replace('.pkl', '_metadata.json')
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    logger.info(f"Model loaded from {filepath}")
    return model, metadata


def performance_summary(metrics: Dict[str, float]) -> str:
    """
    Generate human-readable performance summary
    """
    summary_lines = []

    # Key metrics
    if 'total_return' in metrics:
        summary_lines.append(f"Total Return: {metrics['total_return']:.2%}")

    if 'annualized_return' in metrics:
        summary_lines.append(
            f"Annualized Return: {metrics['annualized_return']:.2%}")

    if 'sharpe_ratio' in metrics:
        sharpe = metrics['sharpe_ratio']
        if sharpe > 2:
            rating = "Excellent"
        elif sharpe > 1:
            rating = "Good"
        elif sharpe > 0.5:
            rating = "Fair"
        else:
            rating = "Poor"
        summary_lines.append(f"Sharpe Ratio: {sharpe:.2f} ({rating})")

    if 'max_drawdown' in metrics:
        summary_lines.append(f"Max Drawdown: {metrics['max_drawdown']:.2%}")

    if 'win_rate' in metrics:
        summary_lines.append(f"Win Rate: {metrics['win_rate']:.2%}")

    return "\n".join(summary_lines)


def format_number(value: Union[int, float],
                  format_type: str = 'auto') -> str:
    """
    Format numbers for display
    """
    if pd.isna(value):
        return "N/A"

    if format_type == 'percentage':
        return f"{value:.2%}"
    elif format_type == 'currency':
        return f"${value:,.2f}"
    elif format_type == 'float':
        return f"{value:.4f}"
    elif format_type == 'integer':
        return f"{int(value):,}"
    else:  # auto
        if abs(value) >= 1000000:
            return f"{value / 1000000:.1f}M"
        elif abs(value) >= 1000:
            return f"{value / 1000:.1f}K"
        elif abs(value) >= 1:
            return f"{value:.2f}"
        else:
            return f"{value:.4f}"


def calculate_correlation_matrix(data: pd.DataFrame,
                                 method: str = 'pearson') -> pd.DataFrame:
    """
    Calculate correlation matrix with different methods
    """
    if method == 'pearson':
        return data.corr(method='pearson')
    elif method == 'spearman':
        return data.corr(method='spearman')
    elif method == 'kendall':
        return data.corr(method='kendall')
    else:
        return data.corr()


def rolling_correlation(series1: pd.Series,
                        series2: pd.Series,
                        window: int = 60) -> pd.Series:
    """
    Calculate rolling correlation between two series
    """
    return series1.rolling(window).corr(series2)


def create_lag_features(data: pd.DataFrame,
                        columns: List[str],
                        lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for time series modeling
    """
    lagged_data = data.copy()

    for col in columns:
        if col in data.columns:
            for lag in lags:
                lagged_data[f"{col}_lag_{lag}"] = data[col].shift(lag)

    return lagged_data


def create_moving_average_features(data: pd.DataFrame,
                                   columns: List[str],
                                   windows: List[int]) -> pd.DataFrame:
    """
    Create moving average features
    """
    ma_data = data.copy()

    for col in columns:
        if col in data.columns:
            for window in windows:
                ma_data[f"{col}_ma_{window}"] = data[col].rolling(
                    window).mean()
                ma_data[f"{col}_ema_{window}"] = data[col].ewm(
                    span=window).mean()
                ma_data[f"{col}_std_{window}"] = data[col].rolling(
                    window).std()

    return ma_data


def detect_anomalies(data: pd.Series,
                     method: str = 'isolation_forest',
                     threshold: float = 0.1) -> pd.Series:
    """
    Detect anomalies in time series data
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    if method == 'isolation_forest':
        # Prepare data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

        # Fit isolation forest
        iso_forest = IsolationForest(contamination=threshold, random_state=42)
        anomalies = iso_forest.fit_predict(data_scaled)

        return pd.Series(anomalies == -1, index=data.index)

    elif method == 'zscore':
        # Z-score method
        z_scores = np.abs((data - data.mean()) / data.std())
        return z_scores > 3

    elif method == 'iqr':
        # Interquartile range method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)

    else:
        return pd.Series(False, index=data.index)


def create_feature_importance_plot_data(feature_importance: Dict[str, float],
                                        top_n: int = 20) -> Dict[str, List]:
    """
    Prepare data for feature importance plots
    """
    # Sort by importance
    sorted_features = sorted(feature_importance.items(),
                             key=lambda x: abs(x[1]), reverse=True)

    # Take top N
    top_features = sorted_features[:top_n]

    return {
        'features': [item[0] for item in top_features],
        'importance': [item[1] for item in top_features]
    }


def create_date_range(start_date: str,
                      end_date: str,
                      freq: str = 'D') -> pd.DatetimeIndex:
    """
    Create date range for time series
    """
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def resample_data(data: pd.DataFrame,
                  freq: str,
                  agg_method: str = 'last') -> pd.DataFrame:
    """
    Resample time series data to different frequency
    """
    if agg_method == 'last':
        return data.resample(freq).last()
    elif agg_method == 'first':
        return data.resample(freq).first()
    elif agg_method == 'mean':
        return data.resample(freq).mean()
    elif agg_method == 'sum':
        return data.resample(freq).sum()
    elif agg_method == 'ohlc':
        # For price data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 1:
            return data[numeric_cols[0]].resample(freq).ohlc()
        else:
            return data.resample(freq).last()
    else:
        return data.resample(freq).last()


def validate_model_inputs(data: pd.DataFrame,
                          target_column: str,
                          feature_columns: List[str]) -> Dict[str, Any]:
    """
    Validate inputs for model training
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check if target column exists
    if target_column not in data.columns:
        validation_results['valid'] = False
        validation_results['errors'].append(
            f"Target column '{target_column}' not found")

    # Check if feature columns exist
    missing_features = [col for col in feature_columns if
                        col not in data.columns]
    if missing_features:
        validation_results['valid'] = False
        validation_results['errors'].append(
            f"Missing feature columns: {missing_features}")

    # Check for sufficient data
    if len(data) < 100:
        validation_results['warnings'].append(
            "Less than 100 observations available")

    # Check for missing values in target
    if target_column in data.columns:
        missing_target = data[target_column].isnull().sum()
        if missing_target > 0:
            validation_results['warnings'].append(
                f"Target column has {missing_target} missing values")

    # Check for constant features
    constant_features = []
    for col in feature_columns:
        if col in data.columns and data[col].nunique() <= 1:
            constant_features.append(col)

    if constant_features:
        validation_results['warnings'].append(
            f"Constant features detected: {constant_features}")

    return validation_results


def memory_usage_info(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Get memory usage information for DataFrame
    """
    memory_info = {
        'total_memory_mb': data.memory_usage(deep=True).sum() / 1024 ** 2,
        'memory_by_column': {},
        'dtypes': data.dtypes.to_dict(),
        'suggestions': []
    }

    for col in data.columns:
        memory_info['memory_by_column'][col] = data[col].memory_usage(
            deep=True) / 1024 ** 2

    # Suggestions for memory optimization
    for col in data.columns:
        if data[col].dtype == 'object':
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio < 0.5:
                memory_info['suggestions'].append(
                    f"Consider converting '{col}' to category type")

        elif data[col].dtype == 'float64':
            if data[col].min() >= np.finfo(np.float32).min and data[
                col].max() <= np.finfo(np.float32).max:
                memory_info['suggestions'].append(
                    f"Consider converting '{col}' to float32")

        elif data[col].dtype == 'int64':
            if data[col].min() >= np.iinfo(np.int32).min and data[
                col].max() <= np.iinfo(np.int32).max:
                memory_info['suggestions'].append(
                    f"Consider converting '{col}' to int32")

    return memory_info


def optimize_dataframe_memory(data: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage
    """
    optimized_data = data.copy()

    for col in optimized_data.columns:
        col_type = optimized_data[col].dtype

        if col_type != 'object':
            c_min = optimized_data[col].min()
            c_max = optimized_data[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    optimized_data[col] = optimized_data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    optimized_data[col] = optimized_data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    optimized_data[col] = optimized_data[col].astype(np.int32)

            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(
                        np.float16).max:
                    optimized_data[col] = optimized_data[col].astype(
                        np.float32)

        else:
            unique_ratio = optimized_data[col].nunique() / len(optimized_data)
            if unique_ratio < 0.5:
                optimized_data[col] = optimized_data[col].astype('category')

    memory_saved = (data.memory_usage(
        deep=True).sum() - optimized_data.memory_usage(
        deep=True).sum()) / 1024 ** 2
    logger.info(f"Memory optimization saved {memory_saved:.2f} MB")

    return optimized_data