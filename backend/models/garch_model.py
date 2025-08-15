"""
GARCH Model Implementation for SP500 Volatility Forecasting
"""
import pandas as pd
import numpy as np
from arch import arch_model
from arch.univariate import GARCH, EGARCH
from itertools import product
from typing import Tuple, Dict, Any, Optional
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class GARCHModel:
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.model_type = 'GARCH'
        self.returns_data = None

    def prepare_returns(self, data: pd.Series,
                        return_type: str = 'log') -> pd.Series:
        """
        Prepare returns data for GARCH modeling
        """
        if return_type == 'log':
            returns = np.log(data / data.shift(1)).dropna()
        else:
            returns = data.pct_change().dropna()

        # Remove extreme outliers (beyond 5 standard deviations)
        std_cutoff = 5
        mean_return = returns.mean()
        std_return = returns.std()

        outlier_mask = np.abs(
            (returns - mean_return) / std_return) < std_cutoff
        returns = returns[outlier_mask]

        # Scale returns to percentage (multiply by 100)
        returns = returns * 100

        self.returns_data = returns
        logger.info(
            f"Prepared {len(returns)} return observations for GARCH modeling")

        return returns

    def test_arch_effects(self, returns: pd.Series, lags: int = 5) -> Dict[
        str, Any]:
        """
        Test for ARCH effects using Lagrange Multiplier test
        """
        from arch.univariate import arch_model

        try:
            # Fit a simple AR model to get residuals
            ar_model = arch_model(returns, vol='Constant', mean='AR', lags=1)
            ar_fitted = ar_model.fit(disp='off')

            # Test for ARCH effects
            arch_test = ar_fitted.arch_lm_test(lags=lags)

            return {
                'lm_statistic': arch_test.stat,
                'p_value': arch_test.pvalue,
                'has_arch_effects': arch_test.pvalue < 0.05,
                'critical_value': arch_test.critical_values['5%'],
                'lags_tested': lags
            }

        except Exception as e:
            logger.warning(f"ARCH effects test failed: {str(e)}")
            return {'error': str(e)}

    def analyze_volatility_clustering(self, returns: pd.Series) -> Dict[
        str, Any]:
        """
        Analyze volatility clustering patterns
        """
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=20).std()

        # Autocorrelation of squared returns (sign of volatility clustering)
        squared_returns = returns ** 2
        autocorrs = [squared_returns.autocorr(lag=i) for i in range(1, 21)]

        # Volatility clustering score (sum of significant autocorrelations)
        clustering_score = sum([abs(ac) for ac in autocorrs if abs(ac) > 0.1])

        return {
            'rolling_volatility': rolling_vol,
            'squared_returns_autocorr': autocorrs,
            'clustering_score': clustering_score,
            'has_clustering': clustering_score > 1.0
        }

    def grid_search_garch(self, returns: pd.Series,
                          max_p: int = 3, max_q: int = 3,
                          model_types: list = ['GARCH', 'EGARCH',
                                               'GJR-GARCH']) -> Dict[str, Any]:
        """
        Grid search for optimal GARCH parameters across different model types
        """
        logger.info(f"Starting GARCH grid search with p≤{max_p}, q≤{max_q}")

        best_aic = np.inf
        best_bic = np.inf
        best_params_aic = None
        best_params_bic = None
        best_model_type = None
        results = []

        for model_type in model_types:
            for p, q in product(range(1, max_p + 1), range(1, max_q + 1)):
                try:
                    # Create model based on type
                    if model_type == 'GARCH':
                        model = arch_model(returns, vol='GARCH', p=p, q=q,
                                           mean='ARX')
                    elif model_type == 'EGARCH':
                        model = arch_model(returns, vol='EGARCH', p=p, q=q,
                                           mean='ARX')
                    elif model_type == 'GJR-GARCH':
                        model = arch_model(returns, vol='GJRGARCH', p=p, q=q,
                                           mean='ARX')
                    else:
                        continue

                    # Fit model
                    fitted_model = model.fit(disp='off', show_warning=False)

                    aic = fitted_model.aic
                    bic = fitted_model.bic

                    results.append({
                        'model_type': model_type,
                        'p': p,
                        'q': q,
                        'aic': aic,
                        'bic': bic,
                        'log_likelihood': fitted_model.loglikelihood,
                        'converged': fitted_model.convergence_flag == 0
                    })

                    # Track best models
                    if aic < best_aic:
                        best_aic = aic
                        best_params_aic = (model_type, p, q)

                    if bic < best_bic:
                        best_bic = bic
                        best_params_bic = (model_type, p, q)

                except Exception as e:
                    logger.debug(
                        f"Failed to fit {model_type}({p},{q}): {str(e)}")
                    continue

        logger.info(f"GARCH grid search completed. Best AIC: {best_aic:.2f}")
        logger.info(f"Best model: {best_params_aic}")

        return {
            'best_params_aic': best_params_aic,
            'best_params_bic': best_params_bic,
            'best_aic': best_aic,
            'best_bic': best_bic,
            'all_results': results
        }

    def fit(self, returns: pd.Series,
            model_type: str = 'GARCH',
            p: Optional[int] = None,
            q: Optional[int] = None) -> Dict[str, Any]:
        """
        Fit GARCH model with given or optimal parameters
        """
        if p is None or q is None:
            logger.info("Performing automatic parameter selection")

            # Test for ARCH effects
            arch_test = self.test_arch_effects(returns)
            if not arch_test.get('has_arch_effects', True):
                logger.warning("No significant ARCH effects detected")

            # Grid search for optimal parameters
            search_results = self.grid_search_garch(returns)

            if search_results['best_params_aic'] is None:
                # Default parameters
                model_type, p, q = 'GARCH', 1, 1
                logger.warning("Using default GARCH(1,1) parameters")
            else:
                model_type, p, q = search_results['best_params_aic']

            self.best_params = (model_type, p, q)

        self.model_type = model_type
        logger.info(f"Fitting {model_type}({p},{q}) model")

        try:
            # Create and fit model
            if model_type == 'GARCH':
                self.model = arch_model(returns, vol='GARCH', p=p, q=q,
                                        mean='ARX')
            elif model_type == 'EGARCH':
                self.model = arch_model(returns, vol='EGARCH', p=p, q=q,
                                        mean='ARX')
            elif model_type == 'GJR-GARCH':
                self.model = arch_model(returns, vol='GJRGARCH', p=p, q=q,
                                        mean='ARX')
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            self.fitted_model = self.model.fit(disp='off', show_warning=False)

            # Model diagnostics
            diagnostics = self.run_diagnostics()

            # Extract key metrics
            conditional_volatility = self.fitted_model.conditional_volatility
            standardized_residuals = self.fitted_model.std_resid

            # Calculate fit metrics
            log_likelihood = self.fitted_model.loglikelihood
            aic = self.fitted_model.aic
            bic = self.fitted_model.bic

            logger.info(
                f"{model_type} model fitted successfully. AIC: {aic:.2f}")

            return {
                'success': True,
                'model_type': model_type,
                'p': p,
                'q': q,
                'aic': aic,
                'bic': bic,
                'log_likelihood': log_likelihood,
                'conditional_volatility': conditional_volatility,
                'standardized_residuals': standardized_residuals,
                'fitted_values': self.fitted_model.fittedvalues,
                'residuals': self.fitted_model.resid,
                'diagnostics': diagnostics,
                'converged': self.fitted_model.convergence_flag == 0,
                'summary': str(self.fitted_model.summary())
            }

        except Exception as e:
            logger.error(f"Error fitting {model_type} model: {str(e)}")
            return {'success': False, 'error': str(e)}

    def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run diagnostic tests on fitted GARCH model
        """
        if self.fitted_model is None:
            return {}

        diagnostics = {}

        try:
            # Standardized residuals
            std_resid = self.fitted_model.std_resid

            # Ljung-Box test on standardized residuals
            from scipy import stats
            lb_stat, lb_pvalue = stats.jarque_bera(std_resid.dropna())

            # Ljung-Box test on squared standardized residuals (test for remaining ARCH)
            squared_resid = std_resid ** 2
            lb_squared_stat = squared_resid.autocorr(lag=5)

            # Jarque-Bera test for normality of standardized residuals
            jb_stat, jb_pvalue = stats.jarque_bera(std_resid.dropna())

            diagnostics.update({
                'ljung_box_residuals': {
                    'statistic': lb_stat,
                    'p_value': lb_pvalue,
                    'residuals_normal': lb_pvalue > 0.05
                },
                'ljung_box_squared': {
                    'autocorr_lag5': lb_squared_stat,
                    'no_remaining_arch': abs(lb_squared_stat) < 0.1
                },
                'jarque_bera': {
                    'statistic': jb_stat,
                    'p_value': jb_pvalue,
                    'residuals_normal': jb_pvalue > 0.05
                }
            })

            # Sign bias test for asymmetric effects
            pos_residuals = std_resid[std_resid > 0]
            neg_residuals = std_resid[std_resid < 0]

            diagnostics['asymmetry'] = {
                'pos_residuals_mean': pos_residuals.mean(),
                'neg_residuals_mean': neg_residuals.mean(),
                'asymmetry_ratio': abs(pos_residuals.mean()) / abs(
                    neg_residuals.mean()) if neg_residuals.mean() != 0 else np.inf
            }

        except Exception as e:
            logger.warning(f"Error running GARCH diagnostics: {str(e)}")
            diagnostics['error'] = str(e)

        return diagnostics

    def forecast_volatility(self, horizon: int = 20) -> Dict[str, Any]:
        """
        Forecast conditional volatility
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")

        logger.info(f"Forecasting volatility for {horizon} periods")

        try:
            # Generate volatility forecast
            forecast_result = self.fitted_model.forecast(horizon=horizon,
                                                         reindex=False)

            # Extract forecasts
            volatility_forecast = forecast_result.variance.iloc[-1, :].values
            volatility_forecast = np.sqrt(
                volatility_forecast)  # Convert variance to volatility

            # Create forecast dates
            last_date = self.fitted_model.data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=horizon, freq='D')

            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'volatility_forecast': volatility_forecast
            }, index=forecast_dates)

            # Calculate volatility percentiles (confidence bands)
            historical_vol = self.fitted_model.conditional_volatility
            vol_percentiles = np.percentile(historical_vol, [5, 25, 75, 95])

            return {
                'success': True,
                'volatility_forecast': volatility_forecast,
                'forecast_dates': forecast_dates.tolist(),
                'forecast_df': forecast_df,
                'volatility_percentiles': {
                    '5th': vol_percentiles[0],
                    '25th': vol_percentiles[1],
                    '75th': vol_percentiles[2],
                    '95th': vol_percentiles[3]
                },
                'mean_forecast': volatility_forecast.mean(),
                'forecast_trend': np.polyfit(range(len(volatility_forecast)),
                                             volatility_forecast, 1)[0]
            }

        except Exception as e:
            logger.error(f"Error forecasting volatility: {str(e)}")
            return {'success': False, 'error': str(e)}

    def rolling_volatility_forecast(self, returns: pd.Series, train_size: int,
                                    forecast_horizon: int = 1) -> Dict[
        str, Any]:
        """
        Perform rolling window volatility forecasting
        """
        logger.info("Performing rolling volatility forecast")

        if len(returns) < train_size + forecast_horizon:
            raise ValueError("Insufficient data for rolling forecast")

        vol_forecasts = []
        actual_vols = []
        forecast_dates = []

        for i in range(train_size, len(returns) - forecast_horizon + 1):
            try:
                # Training data
                train_data = returns.iloc[:i]

                # Use best parameters or default
                if self.best_params:
                    model_type, p, q = self.best_params
                else:
                    model_type, p, q = 'GARCH', 1, 1

                # Fit model
                if model_type == 'GARCH':
                    temp_model = arch_model(train_data, vol='GARCH', p=p, q=q,
                                            mean='ARX')
                elif model_type == 'EGARCH':
                    temp_model = arch_model(train_data, vol='EGARCH', p=p, q=q,
                                            mean='ARX')
                else:
                    temp_model = arch_model(train_data, vol='GJRGARCH', p=p,
                                            q=q, mean='ARX')

                temp_fitted = temp_model.fit(disp='off', show_warning=False)

                # Forecast volatility
                vol_forecast_result = temp_fitted.forecast(
                    horizon=forecast_horizon, reindex=False)
                vol_forecast = np.sqrt(
                    vol_forecast_result.variance.iloc[-1, 0])

                # Actual volatility (realized volatility)
                future_returns = returns.iloc[i:i + forecast_horizon]
                actual_vol = future_returns.std()

                vol_forecasts.append(vol_forecast)
                actual_vols.append(actual_vol)
                forecast_dates.append(returns.index[i + forecast_horizon - 1])

            except Exception as e:
                logger.warning(
                    f"Rolling volatility forecast failed at step {i}: {str(e)}")
                continue

        # Calculate metrics
        vol_forecasts = np.array(vol_forecasts)
        actual_vols = np.array(actual_vols)

        mse = np.mean((vol_forecasts - actual_vols) ** 2)
        mae = np.mean(np.abs(vol_forecasts - actual_vols))
        rmse = np.sqrt(mse)

        # Correlation between forecasted and actual volatility
        correlation = np.corrcoef(vol_forecasts, actual_vols)[0, 1] if len(
            vol_forecasts) > 1 else 0

        results_df = pd.DataFrame({
            'volatility_forecast': vol_forecasts,
            'actual_volatility': actual_vols,
            'forecast_error': vol_forecasts - actual_vols,
            'absolute_error': np.abs(vol_forecasts - actual_vols)
        }, index=forecast_dates)

        return {
            'results': results_df,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'forecast_count': len(vol_forecasts)
        }

    def calculate_var_es(self, confidence_level: float = 0.05,
                         horizon: int = 1) -> Dict[str, Any]:
        """
        Calculate Value at Risk and Expected Shortfall
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before calculating VaR/ES")

        try:
            # Get latest conditional volatility
            latest_vol = self.fitted_model.conditional_volatility.iloc[-1]

            # Standardized residuals for distribution
            std_resid = self.fitted_model.std_resid.dropna()

            # Calculate VaR
            var_quantile = np.percentile(std_resid, confidence_level * 100)
            var = var_quantile * latest_vol * np.sqrt(horizon)

            # Calculate Expected Shortfall (CVaR)
            tail_losses = std_resid[std_resid <= var_quantile]
            es_quantile = tail_losses.mean() if len(
                tail_losses) > 0 else var_quantile
            es = es_quantile * latest_vol * np.sqrt(horizon)

            return {
                'var': var,
                'expected_shortfall': es,
                'confidence_level': confidence_level,
                'horizon': horizon,
                'latest_volatility': latest_vol
            }

        except Exception as e:
            logger.error(f"Error calculating VaR/ES: {str(e)}")
            return {'error': str(e)}

    def volatility_regime_analysis(self) -> Dict[str, Any]:
        """
        Analyze volatility regimes
        """
        if self.fitted_model is None:
            return {}

        conditional_vol = self.fitted_model.conditional_volatility

        # Define volatility regimes based on percentiles
        low_vol_threshold = conditional_vol.quantile(0.33)
        high_vol_threshold = conditional_vol.quantile(0.67)

        # Classify periods
        regimes = pd.cut(conditional_vol,
                         bins=[0, low_vol_threshold, high_vol_threshold,
                               conditional_vol.max()],
                         labels=['Low', 'Medium', 'High'])

        # Calculate regime statistics
        regime_stats = {}
        for regime in ['Low', 'Medium', 'High']:
            regime_data = conditional_vol[regimes == regime]
            if len(regime_data) > 0:
                regime_stats[regime] = {
                    'count': len(regime_data),
                    'percentage': len(regime_data) / len(
                        conditional_vol) * 100,
                    'mean_volatility': regime_data.mean(),
                    'duration_stats': self._calculate_regime_durations(regimes,
                                                                       regime)
                }

        return {
            'regimes': regimes,
            'regime_stats': regime_stats,
            'low_vol_threshold': low_vol_threshold,
            'high_vol_threshold': high_vol_threshold
        }

    def _calculate_regime_durations(self, regimes: pd.Series,
                                    target_regime: str) -> Dict[str, float]:
        """
        Calculate duration statistics for a specific regime
        """
        regime_periods = (regimes == target_regime).astype(int)

        # Find continuous periods
        regime_changes = regime_periods.diff().fillna(0)
        starts = regime_changes[regime_changes == 1].index
        ends = regime_changes[regime_changes == -1].index

        # Handle edge cases
        if len(starts) == 0:
            return {'mean_duration': 0, 'max_duration': 0, 'count': 0}

        if len(ends) < len(starts):
            ends = list(ends) + [regime_periods.index[-1]]

        # Calculate durations
        durations = [(ends[i] - starts[i]).days for i in range(len(starts))]

        return {
            'mean_duration': np.mean(durations),
            'max_duration': np.max(durations),
            'min_duration': np.min(durations),
            'count': len(durations)
        }