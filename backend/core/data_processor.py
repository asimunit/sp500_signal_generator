"""
Data processing utilities for SP500 signal generation
"""
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.price_scaler = MinMaxScaler()
        self.data = None
        self.processed_data = None

    def fetch_sp500_data(self, symbol: str = "^GSPC", period: str = "5y",
                         interval: str = "1d") -> pd.DataFrame:
        """
        Fetch S&P 500 data from Yahoo Finance
        """
        try:
            logger.info(f"Fetching {symbol} data for period: {period}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)

            if data.empty:
                raise ValueError("No data retrieved")

            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in
                            data.columns]

            # Remove timezone info if present
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)

            logger.info(f"Successfully fetched {len(data)} records")
            self.data = data
            return data

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    def calculate_technical_indicators(self,
                                       data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for feature engineering
        """
        df = data.copy()

        # Price-based indicators
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        df['bb_middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std_dev = df['close'].rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std_dev * bb_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std_dev * bb_std)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / df['bb_width']

        # Volatility indicators
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        df['atr'] = self._calculate_atr(df)

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']

        return df

    def _calculate_atr(self, data: pd.DataFrame,
                       period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        """
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
            axis=1)
        return true_range.rolling(window=period).mean()

    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return metrics
        """
        df = data.copy()

        # Simple returns
        df['returns'] = df['close'].pct_change()
        df['returns_1d'] = df['returns']
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_10d'] = df['close'].pct_change(10)
        df['returns_20d'] = df['close'].pct_change(20)

        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Rolling returns statistics
        df['returns_mean_20d'] = df['returns'].rolling(window=20).mean()
        df['returns_std_20d'] = df['returns'].rolling(window=20).std()
        df['returns_skew_20d'] = df['returns'].rolling(window=20).skew()
        df['returns_kurt_20d'] = df['returns'].rolling(window=20).kurt()

        return df

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set for ML models
        """
        logger.info("Creating features for ML models")

        # Start with technical indicators
        df = self.calculate_technical_indicators(data)

        # Add return metrics
        df = self.calculate_returns(df)

        # Price momentum features
        for period in [3, 5, 10, 20]:
            df[f'price_momentum_{period}d'] = df['close'] / df['close'].shift(
                period) - 1
            df[f'volume_momentum_{period}d'] = df['volume'] / df[
                'volume'].shift(period) - 1

        # Trend features
        df['trend_5_20'] = df['sma_5'] / df['sma_20'] - 1
        df['trend_10_50'] = df['sma_10'] / df['sma_50'] - 1

        # Volatility regime
        df['vol_regime'] = pd.qcut(df['volatility'].rank(method='first'),
                                   q=3, labels=['low', 'medium', 'high'])
        df['vol_regime_low'] = (df['vol_regime'] == 'low').astype(int)
        df['vol_regime_medium'] = (df['vol_regime'] == 'medium').astype(int)
        df['vol_regime_high'] = (df['vol_regime'] == 'high').astype(int)

        # Day of week effects
        df['day_of_week'] = df.index.dayofweek
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)

        # Month effects
        df['month'] = df.index.month
        df['is_january'] = (df['month'] == 1).astype(int)
        df['is_december'] = (df['month'] == 12).astype(int)

        return df

    def prepare_target_variables(self, data: pd.DataFrame,
                                 forecast_horizon: int = 20) -> pd.DataFrame:
        """
        Prepare target variables for different prediction tasks
        """
        df = data.copy()

        # Price targets
        for h in range(1, forecast_horizon + 1):
            df[f'price_target_{h}d'] = df['close'].shift(-h)
            df[f'return_target_{h}d'] = (
                        df['close'].shift(-h) / df['close'] - 1)

        # Direction targets (binary classification)
        for h in [1, 5, 10, 20]:
            df[f'direction_target_{h}d'] = (
                        df[f'return_target_{h}d'] > 0).astype(int)

        # Volatility targets
        for h in [5, 10, 20]:
            future_returns = df['returns'].shift(-h).rolling(window=h).std()
            df[f'volatility_target_{h}d'] = future_returns

        return df

    def split_data(self, data: pd.DataFrame,
                   train_ratio: float = 0.8) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets
        """
        split_idx = int(len(data) * train_ratio)
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()

        logger.info(
            f"Data split - Train: {len(train_data)}, Test: {len(test_data)}")
        return train_data, test_data

    def scale_features(self, train_data: pd.DataFrame,
                       test_data: pd.DataFrame,
                       feature_columns: list) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler
        """
        train_scaled = train_data.copy()
        test_scaled = test_data.copy()

        # Fit scaler on training data
        self.scaler.fit(train_data[feature_columns])

        # Transform both datasets
        train_scaled[feature_columns] = self.scaler.transform(
            train_data[feature_columns])
        test_scaled[feature_columns] = self.scaler.transform(
            test_data[feature_columns])

        return train_scaled, test_scaled

    def create_sequences(self, data: pd.DataFrame,
                         feature_columns: list,
                         target_column: str,
                         sequence_length: int = 60) -> Tuple[
        np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM models
        """
        X, y = [], []

        for i in range(sequence_length, len(data)):
            X.append(data[feature_columns].iloc[i - sequence_length:i].values)
            y.append(data[target_column].iloc[i])

        return np.array(X), np.array(y)

    def get_feature_columns(self, data: pd.DataFrame) -> Dict[str, list]:
        """
        Get categorized feature columns
        """
        all_columns = data.columns.tolist()

        # Price features
        price_features = [col for col in all_columns if any(x in col for x in
                                                            ['close', 'high',
                                                             'low', 'open',
                                                             'sma', 'ema',
                                                             'bb_'])]

        # Technical indicators
        technical_features = [col for col in all_columns if
                              any(x in col for x in
                                  ['rsi', 'macd', 'atr', 'volatility',
                                   'momentum'])]

        # Volume features
        volume_features = [col for col in all_columns if 'volume' in col]

        # Return features
        return_features = [col for col in all_columns if
                           'return' in col and 'target' not in col]

        # Categorical features
        categorical_features = [col for col in all_columns if
                                any(x in col for x in
                                    ['regime', 'is_', 'day_of_week', 'month'])]

        # Target columns
        target_features = [col for col in all_columns if 'target' in col]

        return {
            'price': price_features,
            'technical': technical_features,
            'volume': volume_features,
            'returns': return_features,
            'categorical': categorical_features,
            'targets': target_features,
            'all_features': price_features + technical_features + volume_features +
                            return_features + categorical_features
        }

    def process_data(self, symbol: str = "^GSPC", period: str = "5y") -> Dict[
        str, Any]:
        """
        Complete data processing pipeline
        """
        logger.info("Starting data processing pipeline")

        # Fetch data
        raw_data = self.fetch_sp500_data(symbol, period)

        # Create features
        featured_data = self.create_features(raw_data)

        # Create targets
        processed_data = self.prepare_target_variables(featured_data)

        # Remove rows with NaN values
        processed_data = processed_data.dropna()

        # Get feature columns
        feature_groups = self.get_feature_columns(processed_data)

        # Split data
        train_data, test_data = self.split_data(processed_data)

        self.processed_data = processed_data

        logger.info("Data processing pipeline completed")

        return {
            'raw_data': raw_data,
            'processed_data': processed_data,
            'train_data': train_data,
            'test_data': test_data,
            'feature_groups': feature_groups
        }