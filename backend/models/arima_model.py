"""
ARIMA Model Implementation for SP500 Forecasting
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from itertools import product
from typing import Tuple, Dict, Any, Optional
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class ARIMAModel:
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.aic_scores = {}
        self.bic_scores = {}

    def check_stationarity(self, data: pd.Series) -> Dict[str, Any]:
        """
        Check stationarity using Augmented Dickey-Fuller test
        """
        result = adfuller(data.dropna())

        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }

    def difference_series(self, data: pd.Series, max_diff: int = 2) -> Tuple[
        pd.Series, int]:
        """
        Difference the series until it becomes stationary
        """
        diff_data = data.copy()
        diff_order = 0

        for d in range(max_diff + 1):
            stationarity = self.check_stationarity(diff_data)

            if stationarity['is_stationary'] or d == max_diff:
                diff_order = d
                break

            diff_data = diff_data.diff().dropna()

        logger.info(f"Series became stationary with {diff_order} differences")
        return diff_data, diff_order

    def get_optimal_ar_ma_orders(self, data: pd.Series, max_lag: int = 20) -> \
    Dict[str, Any]:
        """
        Determine optimal AR and MA orders using ACF and PACF
        """
        # Calculate ACF and PACF
        acf_values = acf(data.dropna(), nlags=max_lag, fft=True)
        pacf_values = pacf(data.dropna(), nlags=max_lag)

        # Find significant lags (approximate method)
        # For AR order: look at PACF cutoff
        pacf_cutoff = 1.96 / np.sqrt(len(data))
        ar_order = 0
        for i in range(1, len(pacf_values)):
            if abs(pacf_values[i]) < pacf_cutoff:
                ar_order = i - 1
                break

        # For MA order: look at ACF cutoff
        acf_cutoff = 1.96 / np.sqrt(len(data))
        ma_order = 0
        for i in range(1, len(acf_values)):
            if abs(acf_values[i]) < acf_cutoff:
                ma_order = i - 1
                break

        return {
            'suggested_ar_order': min(ar_order, 5),
            'suggested_ma_order': min(ma_order, 5),
            'acf_values': acf_values,
            'pacf_values': pacf_values
        }

    def grid_search_arima(self, data: pd.Series,
                          max_p: int = 5, max_d: int = 2, max_q: int = 5) -> \
    Dict[str, Any]:
        """
        Grid search for optimal ARIMA parameters using AIC/BIC
        """
        logger.info(
            f"Starting ARIMA grid search with p≤{max_p}, d≤{max_d}, q≤{max_q}")

        best_aic = np.inf
        best_bic = np.inf
        best_params_aic = None
        best_params_bic = None
        results = []

        # Grid search over parameters
        for p, d, q in product(range(max_p + 1), range(max_d + 1),
                               range(max_q + 1)):
            try:
                model = ARIMA(data, order=(p, d, q))
                fitted_model = model.fit()

                aic = fitted_model.aic
                bic = fitted_model.bic

                results.append({
                    'order': (p, d, q),
                    'aic': aic,
                    'bic': bic,
                    'log_likelihood': fitted_model.llf,
                    'params': fitted_model.params.to_dict()
                })

                # Track best models
                if aic < best_aic:
                    best_aic = aic
                    best_params_aic = (p, d, q)

                if bic < best_bic:
                    best_bic = bic
                    best_params_bic = (p, d, q)

                self.aic_scores[(p, d, q)] = aic
                self.bic_scores[(p, d, q)] = bic

            except Exception as e:
                logger.warning(f"Failed to fit ARIMA({p},{d},{q}): {str(e)}")
                continue

        logger.info(
            f"Grid search completed. Best AIC: {best_aic:.2f} with {best_params_aic}")
        logger.info(f"Best BIC: {best_bic:.2f} with {best_params_bic}")

        return {
            'best_params_aic': best_params_aic,
            'best_params_bic': best_params_bic,
            'best_aic': best_aic,
            'best_bic': best_bic,
            'all_results': results
        }

    def fit(self, data: pd.Series,
            order: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """
        Fit ARIMA model with given or optimal parameters
        """
        if order is None:
            # Automatic parameter selection
            logger.info("Performing automatic parameter selection")

            # Check stationarity and difference if needed
            stationarity = self.check_stationarity(data)
            if not stationarity['is_stationary']:
                diff_data, d_order = self.difference_series(data)
                max_d = d_order
            else:
                max_d = 0

            # Grid search for optimal parameters
            search_results = self.grid_search_arima(data, max_d=max_d)
            order = search_results['best_params_aic']
            self.best_params = order

        logger.info(f"Fitting ARIMA{order} model")

        try:
            # Fit the model
            self.model = ARIMA(data, order=order)
            self.fitted_model = self.model.fit()

            # Model diagnostics
            diagnostics = self.run_diagnostics()

            # Calculate additional metrics
            fitted_values = self.fitted_model.fittedvalues
            residuals = self.fitted_model.resid

            # In-sample accuracy
            mse = np.mean(residuals ** 2)
            mae = np.mean(np.abs(residuals))
            rmse = np.sqrt(mse)

            logger.info(
                f"Model fitted successfully. AIC: {self.fitted_model.aic:.2f}")

            return {
                'success': True,
                'order': order,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'log_likelihood': self.fitted_model.llf,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'fitted_values': fitted_values,
                'residuals': residuals,
                'diagnostics': diagnostics,
                'summary': str(self.fitted_model.summary())
            }

        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            return {'success': False, 'error': str(e)}

    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run diagnostic tests on fitted model
        """
        if self.fitted_model is None:
            return {}

        diagnostics = {}

        try:
            # Ljung-Box test for residual autocorrelation
            lb_test = acorr_ljungbox(self.fitted_model.resid, lags=10,
                                     return_df=True)
            diagnostics['ljung_box'] = {
                'statistic': lb_test['lb_stat'].iloc[-1],
                'p_value': lb_test['lb_pvalue'].iloc[-1],
                'residuals_are_white_noise': lb_test['lb_pvalue'].iloc[
                                                 -1] > 0.05
            }

            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = self.fitted_model.test_normality()[0][:2]
            diagnostics['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_pvalue,
                'residuals_are_normal': jb_pvalue > 0.05
            }

            # Heteroskedasticity test
            het_stat, het_pvalue = self.fitted_model.test_heteroskedasticity()[
                                       0][:2]
            diagnostics['heteroskedasticity'] = {
                'statistic': het_stat,
                'p_value': het_pvalue,
                'homoskedastic': het_pvalue > 0.05
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

        logger.info(f"Generating {steps}-step forecast")

        try:
            # Generate forecast
            forecast_result = self.fitted_model.forecast(steps=steps,
                                                         alpha=alpha)
            forecast_values = forecast_result

            # Get confidence intervals
            conf_int = self.fitted_model.get_forecast(steps=steps,
                                                      alpha=alpha).conf_int()

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
            logger.error(f"Error generating forecast: {str(e)}")
            return {'success': False, 'error': str(e)}

    def rolling_forecast(self, data: pd.Series, train_size: int,
                         forecast_horizon: int = 1) -> Dict[str, Any]:
        """
        Perform rolling window forecasting for backtesting
        """
        logger.info(
            f"Performing rolling forecast with {train_size} training samples")

        if len(data) < train_size + forecast_horizon:
            raise ValueError("Insufficient data for rolling forecast")

        forecasts = []
        actuals = []
        forecast_dates = []

        for i in range(train_size, len(data) - forecast_horizon + 1):
            try:
                # Training data
                train_data = data.iloc[:i]

                # Fit model on training data
                temp_model = ARIMA(train_data,
                                   order=self.best_params or (1, 1, 1))
                temp_fitted = temp_model.fit()

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

        # Calculate accuracy metrics
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)

        mse = np.mean((forecasts - actuals) ** 2)
        mae = np.mean(np.abs(forecasts - actuals))
        rmse = np.sqrt(mse)

        # Directional accuracy
        forecast_directions = np.sign(np.diff(forecasts))
        actual_directions = np.sign(np.diff(actuals))
        directional_accuracy = np.mean(
            forecast_directions == actual_directions)

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

    def get_parameter_importance(self) -> Dict[str, Any]:
        """
        Analyze parameter importance and significance
        """
        if self.fitted_model is None:
            return {}

        params = self.fitted_model.params
        pvalues = self.fitted_model.pvalues
        conf_int = self.fitted_model.conf_int()

        importance = {}
        for param_name in params.index:
            importance[param_name] = {
                'coefficient': params[param_name],
                'p_value': pvalues[param_name],
                'significant': pvalues[param_name] < 0.05,
                'conf_int_lower': conf_int.loc[param_name, 0],
                'conf_int_upper': conf_int.loc[param_name, 1]
            }

        return importance