"""
Trading Signal Generation using Time Series Predictions and Dynamic Thresholds
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')


class SignalGenerator:
    def __init__(self):
        self.signals = {}
        self.signal_history = {}
        self.thresholds = {}

    def calculate_dynamic_thresholds(self, predictions: pd.Series,
                                     volatility: pd.Series,
                                     method: str = 'adaptive',
                                     lookback_period: int = 252) -> Dict[
        str, float]:
        """
        Calculate dynamic thresholds based on predictions and volatility
        """
        if method == 'adaptive':
            # Adaptive thresholds based on rolling statistics
            rolling_std = predictions.rolling(window=lookback_period,
                                              min_periods=50).std()
            rolling_vol = volatility.rolling(window=lookback_period,
                                             min_periods=50).mean()

            # Base threshold on prediction volatility
            base_threshold = rolling_std.iloc[-1] if not pd.isna(
                rolling_std.iloc[-1]) else 0.02

            # Adjust for market volatility regime
            vol_adjustment = min(rolling_vol.iloc[-1] / rolling_vol.median(),
                                 2.0) if len(rolling_vol.dropna()) > 0 else 1.0

            buy_threshold = base_threshold * vol_adjustment
            sell_threshold = -base_threshold * vol_adjustment

        elif method == 'percentile':
            # Percentile-based thresholds
            buy_threshold = predictions.quantile(0.75)
            sell_threshold = predictions.quantile(0.25)

        elif method == 'volatility_adjusted':
            # Volatility-adjusted fixed thresholds
            current_vol = volatility.iloc[-1] if len(volatility) > 0 else 0.02
            vol_multiplier = max(current_vol / 0.02,
                                 0.5)  # Normalize to 2% base volatility

            buy_threshold = 0.01 * vol_multiplier
            sell_threshold = -0.01 * vol_multiplier

        else:  # 'fixed'
            buy_threshold = 0.01
            sell_threshold = -0.01

        return {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'neutral_upper': buy_threshold * 0.5,
            'neutral_lower': sell_threshold * 0.5
        }

    def generate_basic_signals(self, predictions: pd.Series,
                               thresholds: Dict[str, float],
                               smoothing_window: int = 5) -> pd.Series:
        """
        Generate basic buy/sell/hold signals from predictions
        """
        # Smooth predictions to reduce noise
        if smoothing_window > 1 and len(predictions) >= smoothing_window:
            smoothed_predictions = pd.Series(
                savgol_filter(predictions.values, smoothing_window, 2),
                index=predictions.index
            )
        else:
            smoothed_predictions = predictions

        signals = pd.Series(0, index=predictions.index)  # 0 = hold

        # Generate signals based on thresholds
        signals[smoothed_predictions >= thresholds['buy_threshold']] = 1  # Buy
        signals[
            smoothed_predictions <= thresholds['sell_threshold']] = -1  # Sell

        return signals

    def generate_confidence_weighted_signals(self, predictions: pd.Series,
                                             prediction_intervals: pd.DataFrame,
                                             thresholds: Dict[str, float]) -> \
    Tuple[pd.Series, pd.Series]:
        """
        Generate signals weighted by prediction confidence
        """
        # Calculate prediction confidence (inverse of interval width)
        if 'upper_ci' in prediction_intervals.columns and 'lower_ci' in prediction_intervals.columns:
            interval_width = prediction_intervals['upper_ci'] - \
                             prediction_intervals['lower_ci']
            # Normalize confidence to 0-1 range
            max_width = interval_width.quantile(0.95)
            confidence = 1 - (interval_width / max_width).clip(0, 1)
        else:
            confidence = pd.Series(1.0, index=predictions.index)

        # Generate basic signals
        basic_signals = self.generate_basic_signals(predictions, thresholds)

        # Weight signals by confidence
        weighted_signals = basic_signals * confidence

        return basic_signals, weighted_signals

    def apply_momentum_filter(self, signals: pd.Series,
                              prices: pd.Series,
                              momentum_period: int = 20) -> pd.Series:
        """
        Apply momentum filter to signals
        """
        # Calculate momentum
        momentum = prices.pct_change(momentum_period).fillna(0)

        filtered_signals = signals.copy()

        # Only allow buy signals when momentum is positive
        filtered_signals[(signals == 1) & (momentum < 0)] = 0

        # Only allow sell signals when momentum is negative
        filtered_signals[(signals == -1) & (momentum > 0)] = 0

        return filtered_signals

    def apply_volatility_filter(self, signals: pd.Series,
                                volatility: pd.Series,
                                vol_threshold_percentile: float = 0.8) -> pd.Series:
        """
        Filter signals based on volatility regime
        """
        # Define high volatility threshold
        vol_threshold = volatility.quantile(vol_threshold_percentile)

        filtered_signals = signals.copy()

        # Reduce signal strength during high volatility periods
        high_vol_mask = volatility > vol_threshold
        filtered_signals[high_vol_mask] = filtered_signals[high_vol_mask] * 0.5

        return filtered_signals

    def apply_regime_filter(self, signals: pd.Series,
                            regime_indicator: pd.Series) -> pd.Series:
        """
        Apply market regime filter to signals

        regime_indicator: Series with values like 'bull', 'bear', 'sideways'
        """
        filtered_signals = signals.copy()

        # Modify signals based on regime
        bear_mask = regime_indicator == 'bear'
        sideways_mask = regime_indicator == 'sideways'

        # Reduce buy signals in bear markets
        filtered_signals[(signals == 1) & bear_mask] = 0

        # Reduce all signals in sideways markets
        filtered_signals[sideways_mask] = filtered_signals[sideways_mask] * 0.5

        return filtered_signals

    def generate_ensemble_signals(self,
                                  model_predictions: Dict[str, pd.Series],
                                  model_weights: Optional[
                                      Dict[str, float]] = None,
                                  consensus_threshold: float = 0.6) -> Dict[
        str, Any]:
        """
        Generate ensemble signals from multiple model predictions
        """
        if model_weights is None:
            model_weights = {model: 1.0 / len(model_predictions) for model in
                             model_predictions.keys()}

        # Align all predictions to common index
        common_index = None
        for predictions in model_predictions.values():
            if common_index is None:
                common_index = predictions.index
            else:
                common_index = common_index.intersection(predictions.index)

        # Calculate weighted average predictions
        weighted_predictions = pd.Series(0.0, index=common_index)
        total_weight = 0

        for model_name, predictions in model_predictions.items():
            weight = model_weights.get(model_name, 1.0)
            aligned_predictions = predictions.reindex(common_index).fillna(0)
            weighted_predictions += aligned_predictions * weight
            total_weight += weight

        if total_weight > 0:
            weighted_predictions /= total_weight

        # Generate individual model signals
        individual_signals = {}
        for model_name, predictions in model_predictions.items():
            aligned_predictions = predictions.reindex(common_index).fillna(0)
            volatility = aligned_predictions.rolling(20).std().fillna(0.02)
            thresholds = self.calculate_dynamic_thresholds(aligned_predictions,
                                                           volatility)
            individual_signals[model_name] = self.generate_basic_signals(
                aligned_predictions, thresholds)

        # Consensus signals - require agreement from multiple models
        signal_matrix = pd.DataFrame(individual_signals)

        # Count votes for each direction
        buy_votes = (signal_matrix == 1).sum(axis=1) / len(model_predictions)
        sell_votes = (signal_matrix == -1).sum(axis=1) / len(model_predictions)

        # Generate consensus signals
        consensus_signals = pd.Series(0, index=common_index)
        consensus_signals[buy_votes >= consensus_threshold] = 1
        consensus_signals[sell_votes >= consensus_threshold] = -1

        # Generate ensemble signals from weighted predictions
        ensemble_volatility = weighted_predictions.rolling(20).std().fillna(
            0.02)
        ensemble_thresholds = self.calculate_dynamic_thresholds(
            weighted_predictions, ensemble_volatility)
        ensemble_signals = self.generate_basic_signals(weighted_predictions,
                                                       ensemble_thresholds)

        return {
            'ensemble_predictions': weighted_predictions,
            'ensemble_signals': ensemble_signals,
            'consensus_signals': consensus_signals,
            'individual_signals': individual_signals,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
            'agreement_score': np.maximum(buy_votes, sell_votes)
        }

    def generate_multi_horizon_signals(self,
                                       predictions_dict: Dict[int, pd.Series],
                                       prices: pd.Series) -> Dict[str, Any]:
        """
        Generate signals for multiple forecast horizons
        """
        horizon_signals = {}
        horizon_strength = {}

        for horizon, predictions in predictions_dict.items():
            # Calculate volatility for this horizon
            volatility = prices.pct_change().rolling(horizon).std().fillna(
                0.02)

            # Generate thresholds and signals
            thresholds = self.calculate_dynamic_thresholds(predictions,
                                                           volatility)
            signals = self.generate_basic_signals(predictions, thresholds)

            # Calculate signal strength (how far from threshold)
            strength = pd.Series(0.0, index=predictions.index)
            buy_mask = signals == 1
            sell_mask = signals == -1

            if buy_mask.any():
                strength[buy_mask] = (predictions[buy_mask] - thresholds[
                    'buy_threshold']) / thresholds['buy_threshold']
            if sell_mask.any():
                strength[sell_mask] = (predictions[sell_mask] - thresholds[
                    'sell_threshold']) / abs(thresholds['sell_threshold'])

            horizon_signals[horizon] = signals
            horizon_strength[horizon] = strength

        # Combine signals across horizons with weights (shorter horizons get higher weight)
        horizon_weights = {h: 1.0 / h for h in predictions_dict.keys()}
        total_weight = sum(horizon_weights.values())
        horizon_weights = {h: w / total_weight for h, w in
                           horizon_weights.items()}

        # Calculate weighted signal strength
        combined_strength = pd.Series(0.0, index=predictions_dict[
            list(predictions_dict.keys())[0]].index)
        for horizon, strength in horizon_strength.items():
            weight = horizon_weights[horizon]
            combined_strength += strength * weight

        # Generate final signals based on combined strength
        final_signals = pd.Series(0, index=combined_strength.index)
        final_signals[combined_strength > 0.5] = 1
        final_signals[combined_strength < -0.5] = -1

        return {
            'horizon_signals': horizon_signals,
            'horizon_strength': horizon_strength,
            'combined_strength': combined_strength,
            'final_signals': final_signals,
            'horizon_weights': horizon_weights
        }

    def apply_position_sizing(self, signals: pd.Series,
                              volatility: pd.Series,
                              base_position_size: float = 0.1,
                              volatility_scaling: bool = True) -> pd.Series:
        """
        Apply position sizing to signals based on volatility
        """
        position_sizes = signals.abs() * base_position_size

        if volatility_scaling:
            # Scale position size inversely with volatility
            normalized_vol = volatility / volatility.median()
            vol_adjustment = 1.0 / normalized_vol.clip(lower=0.5, upper=3.0)
            position_sizes *= vol_adjustment

        # Apply signal direction
        position_sizes *= signals.replace(0, np.nan).fillna(
            method='ffill').fillna(0)

        return position_sizes.fillna(0)

    def calculate_signal_quality_metrics(self, signals: pd.Series,
                                         future_returns: pd.Series,
                                         holding_period: int = 5) -> Dict[
        str, float]:
        """
        Calculate quality metrics for generated signals
        """
        # Align signals and returns
        aligned_signals = signals.reindex(future_returns.index).fillna(0)

        # Calculate forward returns for each signal
        signal_returns = []
        signal_periods = []

        for i in range(len(aligned_signals) - holding_period):
            if aligned_signals.iloc[i] != 0:
                signal = aligned_signals.iloc[i]
                forward_return = future_returns.iloc[
                                 i:i + holding_period].sum()
                signal_returns.append(signal * forward_return)
                signal_periods.append(i)

        if not signal_returns:
            return {'signal_count': 0}

        signal_returns = np.array(signal_returns)

        # Calculate metrics
        win_rate = np.sum(signal_returns > 0) / len(signal_returns)
        avg_return = np.mean(signal_returns)
        avg_win = np.mean(signal_returns[signal_returns > 0]) if np.any(
            signal_returns > 0) else 0
        avg_loss = np.mean(signal_returns[signal_returns < 0]) if np.any(
            signal_returns < 0) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        # Sharpe ratio of signal returns
        sharpe_ratio = avg_return / np.std(signal_returns) if np.std(
            signal_returns) > 0 else 0

        # Maximum drawdown of signal cumulative returns
        cumulative_returns = np.cumsum(signal_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns)

        return {
            'signal_count': len(signal_returns),
            'win_rate': win_rate,
            'avg_return': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_return': np.sum(signal_returns)
        }

    def optimize_signal_parameters(self, predictions: pd.Series,
                                   future_returns: pd.Series,
                                   volatility: pd.Series,
                                   parameter_ranges: Dict[str, List]) -> Dict[
        str, Any]:
        """
        Optimize signal generation parameters
        """
        best_sharpe = -np.inf
        best_params = None
        optimization_results = []

        # Generate parameter combinations
        param_combinations = []
        for threshold_mult in parameter_ranges.get('threshold_multiplier',
                                                   [0.5, 1.0, 1.5, 2.0]):
            for smooth_window in parameter_ranges.get('smoothing_window',
                                                      [1, 3, 5, 7]):
                for momentum_period in parameter_ranges.get('momentum_period',
                                                            [10, 20, 30]):
                    param_combinations.append({
                        'threshold_multiplier': threshold_mult,
                        'smoothing_window': smooth_window,
                        'momentum_period': momentum_period
                    })

        for params in param_combinations:
            try:
                # Calculate thresholds with multiplier
                base_thresholds = self.calculate_dynamic_thresholds(
                    predictions, volatility)
                adjusted_thresholds = {
                    key: value * params['threshold_multiplier']
                    for key, value in base_thresholds.items()
                }

                # Generate signals
                signals = self.generate_basic_signals(
                    predictions,
                    adjusted_thresholds,
                    params['smoothing_window']
                )

                # Apply momentum filter
                if 'momentum_period' in params:
                    # Create dummy prices for momentum calculation
                    prices = (1 + future_returns).cumprod()
                    signals = self.apply_momentum_filter(
                        signals, prices, params['momentum_period']
                    )

                # Calculate quality metrics
                quality_metrics = self.calculate_signal_quality_metrics(
                    signals, future_returns)

                optimization_results.append({
                    'params': params,
                    'sharpe_ratio': quality_metrics.get('sharpe_ratio', 0),
                    'win_rate': quality_metrics.get('win_rate', 0),
                    'total_return': quality_metrics.get('total_return', 0),
                    'signal_count': quality_metrics.get('signal_count', 0)
                })

                if quality_metrics.get('sharpe_ratio', 0) > best_sharpe:
                    best_sharpe = quality_metrics.get('sharpe_ratio', 0)
                    best_params = params

            except Exception as e:
                logger.warning(
                    f"Parameter optimization failed for {params}: {str(e)}")
                continue

        return {
            'best_params': best_params,
            'best_sharpe': best_sharpe,
            'optimization_results': optimization_results
        }

    def generate_signal_summary(self, signals: pd.Series) -> Dict[str, Any]:
        """
        Generate summary statistics for signals
        """
        signal_counts = signals.value_counts().to_dict()

        # Calculate signal transitions
        signal_changes = signals.diff().fillna(0)
        buy_signals = (signal_changes == 1).sum() + (signal_changes == 2).sum()
        sell_signals = (signal_changes == -1).sum() + (
                    signal_changes == -2).sum()

        # Calculate signal persistence (average holding period)
        signal_periods = []
        current_signal = 0
        period_length = 0

        for signal in signals:
            if signal != current_signal:
                if period_length > 0:
                    signal_periods.append(period_length)
                current_signal = signal
                period_length = 1
            else:
                period_length += 1

        avg_holding_period = np.mean(signal_periods) if signal_periods else 0

        return {
            'total_periods': len(signals),
            'signal_distribution': signal_counts,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_periods': signal_counts.get(0, 0),
            'avg_holding_period': avg_holding_period,
            'signal_frequency': (buy_signals + sell_signals) / len(signals),
            'activity_ratio': 1 - (signal_counts.get(0, 0) / len(signals))
        }