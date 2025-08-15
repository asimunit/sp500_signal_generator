"""
Facebook Prophet Model Implementation for SP500 Forecasting
"""
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import itertools
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class ProphetModel:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.cv_results = None
        self.performance_metrics = None

    def prepare_prophet_data(self, data: pd.Series) -> pd.DataFrame:
        """
        Prepare data in Prophet format (ds, y columns)
        """
        prophet_df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })

        # Remove any rows with missing values
        prophet_df = prophet_df.dropna()

        # Ensure ds column is datetime
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

        logger.info(
            f"Prepared Prophet data with {len(prophet_df)} observations")
        return prophet_df

    def add_external_regressors(self, prophet_df: pd.DataFrame,
                                external_data: pd.DataFrame,
                                regressor_columns: List[str]) -> pd.DataFrame:
        """
        Add external regressors to Prophet data
        """
        # Align indices
        combined_df = prophet_df.set_index('ds').join(
            external_data[regressor_columns], how='left'
        ).reset_index()

        # Forward fill missing values
        for col in regressor_columns:
            combined_df[col] = combined_df[col].fillna(method='ffill').fillna(
                method='bfill')

        logger.info(f"Added {len(regressor_columns)} external regressors")
        return combined_df

    def detect_outliers(self, data: pd.DataFrame,
                        method: str = 'iqr',
                        threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and optionally remove outliers
        """
        if method == 'iqr':
            Q1 = data['y'].quantile(0.25)
            Q3 = data['y'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data['y'] < lower_bound) | (data['y'] > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs((data['y'] - data['y'].mean()) / data['y'].std())
            outliers = z_scores > threshold

        else:
            outliers = pd.Series([False] * len(data))

        logger.info(
            f"Detected {outliers.sum()} outliers using {method} method")

        # Create outlier dataframe for Prophet
        outlier_df = data[outliers][['ds']].copy()
        outlier_df['outlier'] = 'outlier'

        return outlier_df

    def add_custom_seasonalities(self, model: Prophet,
                                 seasonalities: List[
                                     Dict[str, Any]]) -> Prophet:
        """
        Add custom seasonalities to Prophet model
        """
        for seasonality in seasonalities:
            model.add_seasonality(
                name=seasonality['name'],
                period=seasonality['period'],
                fourier_order=seasonality.get('fourier_order', 3),
                prior_scale=seasonality.get('prior_scale', 10.0),
                mode=seasonality.get('mode', 'additive')
            )
            logger.info(
                f"Added {seasonality['name']} seasonality with period {seasonality['period']}")

        return model

    def grid_search_prophet(self, data: pd.DataFrame,
                            param_grid: Optional[Dict[str, List]] = None) -> \
    Dict[str, Any]:
        """
        Grid search for optimal Prophet parameters
        """
        if param_grid is None:
            param_grid = {
                'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
                'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                'seasonality_mode': ['additive', 'multiplicative'],
                'changepoint_range': [0.8, 0.9, 0.95]
            }

        logger.info("Starting Prophet hyperparameter tuning")

        # Generate all combinations
        all_params = [dict(zip(param_grid.keys(), v))
                      for v in itertools.product(*param_grid.values())]

        best_mape = float('inf')
        best_params = None
        results = []

        for params in all_params:
            try:
                # Create and fit model
                model = Prophet(
                    changepoint_prior_scale=params['changepoint_prior_scale'],
                    seasonality_prior_scale=params['seasonality_prior_scale'],
                    seasonality_mode=params['seasonality_mode'],
                    changepoint_range=params['changepoint_range'],
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True
                )

                model.fit(data)

                # Cross validation
                cv_results = cross_validation(
                    model,
                    initial='730 days',  # 2 years
                    period='180 days',  # 6 months
                    horizon='30 days',  # 1 month
                    disable_tqdm=True
                )

                # Calculate performance metrics
                performance = performance_metrics(cv_results)
                mape = performance['mape'].mean()

                results.append({
                    'params': params,
                    'mape': mape,
                    'rmse': performance['rmse'].mean(),
                    'mae': performance['mae'].mean()
                })

                if mape < best_mape:
                    best_mape = mape
                    best_params = params

                logger.info(f"MAPE: {mape:.4f} for params: {params}")

            except Exception as e:
                logger.warning(f"Failed to evaluate params {params}: {str(e)}")
                continue

        logger.info(f"Best MAPE: {best_mape:.4f} with params: {best_params}")

        return {
            'best_params': best_params,
            'best_mape': best_mape,
            'all_results': results
        }

    def fit(self, data: pd.Series,
            external_regressors: Optional[pd.DataFrame] = None,
            regressor_columns: Optional[List[str]] = None,
            custom_seasonalities: Optional[List[Dict[str, Any]]] = None,
            hyperparameter_tune: bool = True,
            **prophet_kwargs) -> Dict[str, Any]:
        """
        Fit Prophet model with optional external regressors and custom seasonalities
        """
        logger.info("Fitting Prophet model")

        try:
            # Prepare data
            prophet_df = self.prepare_prophet_data(data)

            # Add external regressors if provided
            if external_regressors is not None and regressor_columns is not None:
                prophet_df = self.add_external_regressors(
                    prophet_df, external_regressors, regressor_columns
                )

            # Hyperparameter tuning
            if hyperparameter_tune:
                tuning_results = self.grid_search_prophet(prophet_df)
                best_params = tuning_results['best_params']
            else:
                best_params = {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10.0,
                    'seasonality_mode': 'multiplicative',
                    'changepoint_range': 0.8
                }

            # Merge with any additional kwargs
            model_params = {**best_params, **prophet_kwargs}
            self.best_params = model_params

            # Create model
            self.model = Prophet(
                changepoint_prior_scale=model_params[
                    'changepoint_prior_scale'],
                seasonality_prior_scale=model_params[
                    'seasonality_prior_scale'],
                seasonality_mode=model_params['seasonality_mode'],
                changepoint_range=model_params['changepoint_range'],
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                uncertainty_samples=1000
            )

            # Add external regressors to model
            if regressor_columns is not None:
                for regressor in regressor_columns:
                    self.model.add_regressor(regressor)

            # Add custom seasonalities
            if custom_seasonalities is not None:
                self.model = self.add_custom_seasonalities(self.model,
                                                           custom_seasonalities)

            # Detect and add outliers
            outliers = self.detect_outliers(prophet_df)
            if len(outliers) > 0:
                prophet_df = prophet_df.merge(outliers, on='ds', how='left')
                prophet_df['outlier'] = prophet_df['outlier'].fillna('normal')

            # Fit model
            self.model.fit(prophet_df)

            # Model diagnostics
            diagnostics = self.run_diagnostics(prophet_df)

            # In-sample predictions for evaluation
            forecast = self.model.predict(prophet_df)

            # Calculate metrics
            actual = prophet_df['y'].values
            predicted = forecast['yhat'].values

            mse = np.mean((actual - predicted) ** 2)
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            logger.info(
                f"Prophet model fitted successfully. MAPE: {mape:.2f}%")

            return {
                'success': True,
                'best_params': best_params,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'in_sample_forecast': forecast,
                'diagnostics': diagnostics,
                'changepoints': self.model.changepoints,
                'n_changepoints': len(self.model.changepoints)
            }

        except Exception as e:
            logger.error(f"Error fitting Prophet model: {str(e)}")
            return {'success': False, 'error': str(e)}

    def forecast(self, periods: int = 20,
                 freq: str = 'D',
                 include_history: bool = True) -> Dict[str, Any]:
        """
        Generate forecasts using fitted Prophet model
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")

        logger.info(f"Generating Prophet forecast for {periods} periods")

        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=periods,
                freq=freq,
                include_history=include_history
            )

            # Generate forecast
            forecast = self.model.predict(future)

            # Extract forecast components
            forecast_period = forecast.tail(
                periods) if include_history else forecast

            # Create forecast dates
            forecast_dates = forecast_period['ds'].tolist()
            forecast_values = forecast_period['yhat'].tolist()
            lower_bounds = forecast_period['yhat_lower'].tolist()
            upper_bounds = forecast_period['yhat_upper'].tolist()

            # Decompose forecast into components
            components = {
                'trend': forecast_period['trend'].tolist(),
                'seasonal': forecast_period.get('seasonal', []).tolist(),
                'weekly': forecast_period.get('weekly', []).tolist(),
                'yearly': forecast_period.get('yearly', []).tolist()
            }

            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'forecast': forecast_values,
                'lower_ci': lower_bounds,
                'upper_ci': upper_bounds,
                'trend': components['trend']
            }, index=pd.to_datetime(forecast_dates))

            return {
                'success': True,
                'forecast_values': forecast_values,
                'forecast_dates': forecast_dates,
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'forecast_df': forecast_df,
                'components': components,
                'full_forecast': forecast
            }

        except Exception as e:
            logger.error(f"Error generating Prophet forecast: {str(e)}")
            return {'success': False, 'error': str(e)}

    def run_diagnostics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run diagnostic tests and cross-validation
        """
        if self.model is None:
            return {}

        diagnostics = {}

        try:
            # Cross validation
            logger.info("Running cross-validation")
            cv_results = cross_validation(
                self.model,
                initial='365 days',
                period='90 days',
                horizon='30 days',
                disable_tqdm=True
            )

            # Performance metrics
            performance = performance_metrics(cv_results)

            self.cv_results = cv_results
            self.performance_metrics = performance

            diagnostics.update({
                'cross_validation': {
                    'mean_mape': performance['mape'].mean(),
                    'mean_rmse': performance['rmse'].mean(),
                    'mean_mae': performance['mae'].mean(),
                    'mape_std': performance['mape'].std(),
                    'rmse_std': performance['rmse'].std(),
                    'mae_std': performance['mae'].std()
                }
            })

            # Residual analysis
            in_sample_forecast = self.model.predict(data)
            residuals = data['y'] - in_sample_forecast['yhat']

            diagnostics.update({
                'residuals': {
                    'mean': residuals.mean(),
                    'std': residuals.std(),
                    'skewness': residuals.skew(),
                    'kurtosis': residuals.kurtosis(),
                    'ljung_box_p': self._ljung_box_test(residuals)
                }
            })

            # Trend analysis
            trend_changes = np.diff(in_sample_forecast['trend'])
            diagnostics.update({
                'trend_analysis': {
                    'trend_volatility': np.std(trend_changes),
                    'trend_direction_changes': np.sum(
                        np.diff(np.sign(trend_changes)) != 0),
                    'average_trend_change': np.mean(np.abs(trend_changes))
                }
            })

        except Exception as e:
            logger.warning(f"Error running Prophet diagnostics: {str(e)}")
            diagnostics['error'] = str(e)

        return diagnostics

    def _ljung_box_test(self, residuals: pd.Series, lags: int = 10) -> float:
        """
        Ljung-Box test for residual autocorrelation
        """
        try:
            from scipy import stats

            # Simple autocorrelation test
            autocorrs = [residuals.autocorr(lag=i) for i in range(1, lags + 1)]
            lb_stat = len(residuals) * (len(residuals) + 2) * sum(
                [autocorrs[i] ** 2 / (len(residuals) - i - 1) for i in
                 range(lags)]
            )
            p_value = 1 - stats.chi2.cdf(lb_stat, lags)
            return p_value

        except:
            return 0.5  # Default p-value if test fails

    def rolling_forecast(self, data: pd.Series, train_size: int,
                         forecast_horizon: int = 1) -> Dict[str, Any]:
        """
        Perform rolling window forecasting for backtesting
        """
        logger.info("Performing Prophet rolling forecast")

        if len(data) < train_size + forecast_horizon:
            raise ValueError("Insufficient data for rolling forecast")

        forecasts = []
        actuals = []
        forecast_dates = []
        lower_bounds = []
        upper_bounds = []

        for i in range(train_size, len(data) - forecast_horizon + 1):
            try:
                # Training data
                train_data = data.iloc[:i]
                prophet_df = self.prepare_prophet_data(train_data)

                # Create and fit model with best parameters
                if self.best_params:
                    model_params = self.best_params
                else:
                    model_params = {
                        'changepoint_prior_scale': 0.05,
                        'seasonality_mode': 'multiplicative'
                    }

                temp_model = Prophet(
                    changepoint_prior_scale=model_params.get(
                        'changepoint_prior_scale', 0.05),
                    seasonality_mode=model_params.get('seasonality_mode',
                                                      'multiplicative'),
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True
                )

                temp_model.fit(prophet_df)

                # Forecast
                future = temp_model.make_future_dataframe(
                    periods=forecast_horizon, freq='D')
                forecast = temp_model.predict(future)

                # Get forecast values
                forecast_value = forecast['yhat'].iloc[-1]
                lower_bound = forecast['yhat_lower'].iloc[-1]
                upper_bound = forecast['yhat_upper'].iloc[-1]
                actual = data.iloc[i + forecast_horizon - 1]

                forecasts.append(forecast_value)
                lower_bounds.append(lower_bound)
                upper_bounds.append(upper_bound)
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
        mape = np.mean(np.abs((actuals - forecasts) / actuals)) * 100

        # Directional accuracy
        if len(forecasts) > 1:
            forecast_directions = np.sign(np.diff(forecasts))
            actual_directions = np.sign(np.diff(actuals))
            directional_accuracy = np.mean(
                forecast_directions == actual_directions)
        else:
            directional_accuracy = 0.0

        # Coverage (percentage of actuals within confidence intervals)
        coverage = np.mean(
            (actuals >= lower_bounds) & (actuals <= upper_bounds))

        results_df = pd.DataFrame({
            'forecast': forecasts,
            'actual': actuals,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'error': forecasts - actuals,
            'absolute_error': np.abs(forecasts - actuals)
        }, index=forecast_dates)

        return {
            'results': results_df,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'coverage': coverage,
            'forecast_count': len(forecasts)
        }

    def analyze_components(self) -> Dict[str, Any]:
        """
        Analyze Prophet model components
        """
        if self.model is None:
            return {}

        # Get the last fitted forecast for component analysis
        if not hasattr(self, '_last_forecast'):
            logger.warning("No forecast available for component analysis")
            return {}

        forecast = self._last_forecast

        components_analysis = {}

        # Trend analysis
        trend = forecast['trend']
        components_analysis['trend'] = {
            'overall_direction': 'increasing' if trend.iloc[-1] > trend.iloc[
                0] else 'decreasing',
            'volatility': trend.std(),
            'total_change': trend.iloc[-1] - trend.iloc[0],
            'percentage_change': ((trend.iloc[-1] - trend.iloc[0]) /
                                  trend.iloc[0]) * 100
        }

        # Seasonal analysis
        if 'yearly' in forecast.columns:
            yearly = forecast['yearly']
            components_analysis['yearly_seasonality'] = {
                'amplitude': yearly.max() - yearly.min(),
                'peak_month': yearly.idxmax().month if hasattr(yearly.idxmax(),
                                                               'month') else None,
                'trough_month': yearly.idxmin().month if hasattr(
                    yearly.idxmin(), 'month') else None
            }

        if 'weekly' in forecast.columns:
            weekly = forecast['weekly']
            components_analysis['weekly_seasonality'] = {
                'amplitude': weekly.max() - weekly.min(),
                'strongest_day': weekly.idxmax().dayofweek if hasattr(
                    weekly.idxmax(), 'dayofweek') else None,
                'weakest_day': weekly.idxmin().dayofweek if hasattr(
                    weekly.idxmin(), 'dayofweek') else None
            }

        # Changepoint analysis
        changepoints = self.model.changepoints
        if len(changepoints) > 0:
            components_analysis['changepoints'] = {
                'count': len(changepoints),
                'dates': changepoints.tolist(),
                'most_recent': changepoints[-1] if len(
                    changepoints) > 0 else None
            }

        return components_analysis

    def predict_volatility_regime(self, forecast_df: pd.DataFrame) -> Dict[
        str, Any]:
        """
        Predict volatility regime based on Prophet uncertainty
        """
        # Calculate uncertainty width
        uncertainty = forecast_df['upper_ci'] - forecast_df['lower_ci']

        # Define regime thresholds based on historical uncertainty
        low_threshold = uncertainty.quantile(0.33)
        high_threshold = uncertainty.quantile(0.67)

        # Classify regimes
        regimes = pd.cut(uncertainty,
                         bins=[0, low_threshold, high_threshold,
                               uncertainty.max()],
                         labels=['Low Volatility', 'Medium Volatility',
                                 'High Volatility'])

        return {
            'regimes': regimes,
            'uncertainty_levels': uncertainty,
            'low_threshold': low_threshold,
            'high_threshold': high_threshold,
            'current_regime': regimes.iloc[-1] if len(regimes) > 0 else None
        }