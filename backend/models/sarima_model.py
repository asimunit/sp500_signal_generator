"""
SARIMA Model Implementation for SP500 Forecasting with Seasonality
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from itertools import product
from typing import Tuple, Dict, Any, Optional, List
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class SARIMAModel:
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.seasonal_periods = None
        self.decomposition = None

    def detect_seasonality(self, data: pd.Series,
                           candidate_periods: List[int] = [5, 10, 21, 63]) -> \
    Dict[str, Any]:
        """
        Detect seasonal patterns in the data
        """
        seasonality_results = {}

        for period in candidate_periods:
            if len(data) < 2 * period:
                continue

            try:
                # Perform seasonal decomposition
                decomp = seasonal_decompose(data, model='additive',
                                            period=period)

                # Calculate seasonal strength (variance of seasonal component vs residual)
                seasonal_var = np.var(decomp.seasonal.dropna())
                residual_var = np.var(decomp.resid.dropna())
                seasonal_strength = seasonal_var / (
                            seasonal_var + residual_var)

                # Autocorrelation at seasonal lag
                seasonal_autocorr = data.autocorr(lag=period)

                seasonality_results[period] = {
                    'seasonal_strength': seasonal_strength,
                    'seasonal_autocorr': seasonal_autocorr,
                    'decomposition': decomp
                }

            except Exception as e:
                logger.warning(
                    f"Seasonality detection failed for period {period}: {str(e)}")
                continue

        # Select best seasonal period
        if seasonality_results:
            best_period = max(seasonality_results.keys(),
                              key=lambda p: seasonality_results[p][
                                  'seasonal_strength'])

            logger.info(
                f"Detected strongest seasonality with period: {best_period}")
            self.seasonal_periods = best_period
            self.decomposition = seasonality_results[best_period][
                'decomposition']

        return seasonality_results

    def check_seasonal_stationarity(self, data: pd.Series,
                                    seasonal_period: int) -> Dict[str, Any]:
        """
        Check stationarity including seasonal differencing
        """
        results = {}

        # Test original series
        original_adf = adfuller(data.dropna())
        results['original'] = {
            'adf_statistic': original_adf[0],
            'p_value': original_adf[1],
            'is_stationary': original_adf[1] < 0.05
        }

        # Test first difference
        diff1 = data.diff().dropna()
        diff1_adf = adfuller(diff1)
        results['first_diff'] = {
            'adf_statistic': diff1_adf[0],
            'p_value': diff1_adf[1],
            'is_stationary': diff1_adf[1] < 0.05
        }

        # Test seasonal difference
        if len(data) > seasonal_period:
            seasonal_diff = data.diff(seasonal_period).dropna()
            if len(seasonal_diff) > 0:
                seasonal_adf = adfuller(seasonal_diff)
                results['seasonal_diff'] = {
                    'adf_statistic': seasonal_adf[0],
                    'p_value': seasonal_adf[1],
                    'is_stationary': seasonal_adf[1] < 0.05
                }

        # Test both differences
        if len(data) > seasonal_period + 1:
            both_diff = data.diff().diff(seasonal_period).dropna()
            if len(both_diff) > 0:
                both_adf = adfuller(both_diff)
                results['both_diff'] = {
                    'adf_statistic': both_adf[0],
                    'p_value': both_adf[1],
                    'is_stationary': both_adf[1] < 0.05
                }

        return results

    def determine_differencing(self, data: pd.Series, seasonal_period: int) -> \
    Tuple[int, int]:
        """
        Determine optimal non-seasonal and seasonal differencing orders
        """
        stationarity_tests = self.check_seasonal_stationarity(data,
                                                              seasonal_period)

        # Start with no differencing
        d, D = 0, 0

        # Check if original series is stationary
        if stationarity_tests['original']['is_stationary']:
            return d, D

        # Check seasonal differencing first
        if 'seasonal_diff' in stationarity_tests and \
                stationarity_tests['seasonal_diff']['is_stationary']:
            D = 1
            return d, D

        # Check first differencing
        if stationarity_tests['first_diff']['is_stationary']:
            d = 1
            return d, D

        # Check both differencing
        if 'both_diff' in stationarity_tests and \
                stationarity_tests['both_diff']['is_stationary']:
            d, D = 1, 1

        return d, D

    def grid_search_sarima(self, data: pd.Series, seasonal_period: int,
                           max_p: int = 3, max_d: int = 2, max_q: int = 3,
                           max_P: int = 2, max_D: int = 1, max_Q: int = 2) -> \
    Dict[str, Any]:
        """
        Grid search for optimal SARIMA parameters
        """
        logger.info(
            f"Starting SARIMA grid search with seasonal period {seasonal_period}")

        best_aic = np.inf
        best_bic = np.inf
        best_params_aic = None
        best_params_bic = None
        results = []

        # Determine differencing orders
        d_auto, D_auto = self.determine_differencing(data, seasonal_period)
        d_range = [d_auto] if d_auto <= max_d else range(max_d + 1)
        D_range = [D_auto] if D_auto <= max_D else range(max_D + 1)

        # Grid search
        total_combinations = (max_p + 1) * len(d_range) * (max_q + 1) * (
                    max_P + 1) * len(D_range) * (max_Q + 1)
        logger.info(f"Testing {total_combinations} parameter combinations")

        for p, d, q, P, D, Q in product(range(max_p + 1), d_range,
                                        range(max_q + 1),
                                        range(max_P + 1), D_range,
                                        range(max_Q + 1)):
            try:
                model = SARIMAX(data,
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, seasonal_period),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

                fitted_model = model.fit(disp=False, maxiter=100)

                aic = fitted_model.aic
                bic = fitted_model.bic

                results.append({
                    'order': (p, d, q),
                    'seasonal_order': (P, D, Q, seasonal_period),
                    'aic': aic,
                    'bic': bic,
                    'log_likelihood': fitted_model.llf,
                    'converged': fitted_model.mle_retvals['converged']
                })

                if aic < best_aic:
                    best_aic = aic
                    best_params_aic = ((p, d, q), (P, D, Q, seasonal_period))

                if bic < best_bic:
                    best_bic = bic
                    best_params_bic = ((p, d, q), (P, D, Q, seasonal_period))

            except Exception as e:
                logger.debug(
                    f"Failed SARIMA({p},{d},{q})x({P},{D},{Q},{seasonal_period}): {str(e)}")
                continue

        logger.info(f"Grid search completed. Best AIC: {best_aic:.2f}")
        logger.info(f"Best params (AIC): {best_params_aic}")

        return {
            'best_params_aic': best_params_aic,
            'best_params_bic': best_params_bic,
            'best_aic': best_aic,
            'best_bic': best_bic,
            'all_results': results,
            'tested_combinations': len(results)
        }

    def fit(self, data: pd.Series,
            order: Optional[Tuple[int, int, int]] = None,
            seasonal_order: Optional[Tuple[int, int, int, int]] = None) -> \
    Dict[str, Any]:
        """
        Fit SARIMA model with given or optimal parameters
        """
        if order is None or seasonal_order is None:
            logger.info("Performing automatic parameter selection")

            # Detect seasonality
            seasonality_results = self.detect_seasonality(data)

            if not seasonality_results:
                logger.warning(
                    "No seasonality detected, falling back to ARIMA")
                seasonal_period = 1
            else:
                seasonal_period = self.seasonal_periods

            # Grid search for optimal parameters
            search_results = self.grid_search_sarima(data, seasonal_period)

            if search_results['best_params_aic'] is None:
                raise ValueError("Could not find suitable SARIMA parameters")

            order, seasonal_order = search_results['best_params_aic']
            self.best_params = (order, seasonal_order)

        logger.info(f"Fitting SARIMA{order}x{seasonal_order} model")

        try:
            # Fit the model
            self.model = SARIMAX(data,
                                 order=order,
                                 seasonal_order=seasonal_order,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)

            self.fitted_model = self.model.fit(disp=False, maxiter=200)

            # Model diagnostics
            diagnostics = self.run_diagnostics()

            # Calculate metrics
            fitted_values = self.fitted_model.fittedvalues
            residuals = self.fitted_model.resid

            mse = np.mean(residuals ** 2)
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(mse)

            logger.info(
                f"SARIMA model fitted successfully. AIC: {self.fitted_model.aic:.2f}")

            return {
                'success': True,
                'order': order,
                'seasonal_order': seasonal_order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'log_likelihood': self.fitted_model.llf,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'fitted_values': fitted_values,
                'residuals': residuals,
                'diagnostics': diagnostics,
                'converged': self.fitted_model.mle_retvals['converged'],
                'summary': str(self.fitted_model.summary())
            }

        except Exception as e:
            logger.error(f"Error fitting SARIMA model: {str(e)}")
            return {'success': False, 'error': str(e)}

    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run comprehensive diagnostic tests
        """
        if self.fitted_model is None:
            return {}

        diagnostics = {}

        try:
            # Ljung-Box test for serial correlation
            ljung_box = self.fitted_model.test_serial_correlation('ljungbox',
                                                                  lags=10)
            diagnostics['ljung_box'] = {
                'statistic': ljung_box[0],
                'p_value': ljung_box[1],
                'no_serial_correlation': ljung_box[1] > 0.05
            }

            # Jarque-Bera test for normality
            jarque_bera = self.fitted_model.test_normality('jarquebera')
            diagnostics['jarque_bera'] = {
                'statistic': jarque_bera[0],
                'p_value': jarque_bera[1],
                'residuals_normal': jarque_bera[1] > 0.05
            }

            # Heteroskedasticity test
            het_test = self.fitted_model.test_heteroskedasticity('breakvar')
            diagnostics['heteroskedasticity'] = {
                'statistic': het_test[0],
                'p_value': het_test[1],
                'homoskedastic': het_test[1] > 0.05
            }

        except Exception as e:
            logger.warning(f"Error running diagnostics: {str(e)}")
            diagnostics['error'] = str(e)

        return diagnostics

    def forecast(self, steps: int = 20, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Generate forecasts with confidence intervals
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        logger.info(f"Generating {steps}-step SARIMA forecast")

        try:
            # Get forecast
            forecast_result = self.fitted_model.get_forecast(steps=steps,
                                                             alpha=alpha)
            forecast_values = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int()

            # Create forecast dates
            last_date = self.fitted_model.data.dates[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps, freq='D')

            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'forecast': forecast_values,
                'lower_ci': conf_int.iloc[:, 0],
                'upper_ci': conf_int.iloc[:, 1]
            }, index=forecast_dates)

            return {
                'success': True,
                'forecast': forecast_df,
                'forecast_values': forecast_values.tolist(),
                'confidence_interval': conf_int.values.tolist(),
                'forecast_dates': forecast_dates.tolist()
            }

        except Exception as e:
            logger.error(f"Error generating SARIMA forecast: {str(e)}")
            return {'success': False, 'error': str(e)}

    def rolling_forecast(self, data: pd.Series, train_size: int,
                         forecast_horizon: int = 1) -> Dict[str, Any]:
        """
        Perform rolling window forecasting for backtesting
        """
        logger.info(f"Performing SARIMA rolling forecast")

        if len(data) < train_size + forecast_horizon:
            raise ValueError("Insufficient data for rolling forecast")

        forecasts = []
        actuals = []
        forecast_dates = []

        for i in range(train_size, len(data) - forecast_horizon + 1):
            try:
                # Training data
                train_data = data.iloc[:i]

                # Use best parameters or default
                if self.best_params:
                    order, seasonal_order = self.best_params
                else:
                    order = (1, 1, 1)
                    seasonal_order = (1, 1, 1, 5)

                # Fit model
                temp_model = SARIMAX(train_data,
                                     order=order,
                                     seasonal_order=seasonal_order,
                                     enforce_stationarity=False,
                                     enforce_invertibility=False)

                temp_fitted = temp_model.fit(disp=False, maxiter=50)

                # Forecast
                forecast = temp_fitted.forecast(steps=forecast_horizon)[0]
                actual = data.iloc[i + forecast_horizon - 1]

                forecasts.append(forecast)
                actuals.append(actual)
                forecast_dates.append(data.index[i + forecast_horizon - 1])

            except Exception as e:
                logger.warning(
                    f"Rolling forecast failed at step {i}: {str(e)}")
                continue

        # Calculate metrics
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)

        mse = np.mean((forecasts - actuals) ** 2)
        mae = np.mean(np.abs(forecasts - actuals))
        rmse = np.sqrt(mse)

        # Directional accuracy
        if len(forecasts) > 1:
            forecast_directions = np.sign(np.diff(forecasts))
            actual_directions = np.sign(np.diff(actuals))
            directional_accuracy = np.mean(
                forecast_directions == actual_directions)
        else:
            directional_accuracy = 0.0

        results_df = pd.DataFrame({
            'forecast': forecasts,
            'actual': actuals,
            'error': forecasts - actuals,
            'absolute_error': np.abs(forecasts - actuals)
        }, index=forecast_dates)

        return {
            'results': results_df,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'directional_accuracy': directional_accuracy,
            'forecast_count': len(forecasts)
        }

    def analyze_seasonal_components(self) -> Dict[str, Any]:
        """
        Analyze seasonal components of the fitted model
        """
        if self.decomposition is None:
            return {}

        seasonal_component = self.decomposition.seasonal
        trend_component = self.decomposition.trend
        residual_component = self.decomposition.resid

        # Calculate component statistics
        seasonal_stats = {
            'mean': seasonal_component.mean(),
            'std': seasonal_component.std(),
            'min': seasonal_component.min(),
            'max': seasonal_component.max(),
            'amplitude': seasonal_component.max() - seasonal_component.min()
        }

        trend_stats = {
            'mean': trend_component.mean(),
            'std': trend_component.std(),
            'slope': np.polyfit(range(len(trend_component.dropna())),
                                trend_component.dropna(), 1)[0]
        }

        return {
            'seasonal_stats': seasonal_stats,
            'trend_stats': trend_stats,
            'seasonal_component': seasonal_component,
            'trend_component': trend_component,
            'residual_component': residual_component
        }