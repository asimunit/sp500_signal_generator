"""
Comprehensive Backtesting Engine for SP500 Trading Signals
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class Backtester:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.results = {}
        self.trades = pd.DataFrame()
        self.portfolio_values = pd.Series()
        self.positions = pd.Series()

    def run_backtest(self, signals: pd.Series,
                     prices: pd.Series,
                     position_sizes: Optional[pd.Series] = None,
                     transaction_cost: float = 0.001,
                     slippage: float = 0.0001,
                     max_position_size: float = 1.0,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Run comprehensive backtest with realistic trading constraints
        """
        logger.info("Starting backtest execution")

        # Align signals and prices
        common_index = signals.index.intersection(prices.index)
        signals = signals.reindex(common_index).fillna(0)
        prices = prices.reindex(common_index).fillna(method='ffill')

        if position_sizes is None:
            position_sizes = signals.abs() * 0.1  # Default 10% position size
        else:
            position_sizes = position_sizes.reindex(common_index).fillna(0)

        # Initialize tracking variables
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        position = 0.0
        position_price = 0.0

        portfolio_history = []
        cash_history = []
        position_history = []
        trades_list = []

        for i, (date, signal) in enumerate(signals.items()):
            current_price = prices.loc[date]

            # Skip if price data is missing
            if pd.isna(current_price):
                portfolio_history.append(portfolio_value)
                cash_history.append(cash)
                position_history.append(position)
                continue

            # Check stop loss and take profit
            if position != 0:
                position_return = (
                                              current_price - position_price) / position_price
                position_pnl = position_return if position > 0 else -position_return

                # Stop loss check
                if stop_loss and position_pnl < -stop_loss:
                    signal = -np.sign(position)  # Force exit
                    logger.debug(f"Stop loss triggered at {date}")

                # Take profit check
                elif take_profit and position_pnl > take_profit:
                    signal = -np.sign(position)  # Force exit
                    logger.debug(f"Take profit triggered at {date}")

            # Calculate target position
            if signal != 0:
                target_position_size = min(abs(position_sizes.loc[date]),
                                           max_position_size)
                target_position_value = target_position_size * portfolio_value
                target_position = np.sign(
                    signal) * target_position_value / current_price
            else:
                target_position = position  # No change

            # Execute trades
            if abs(target_position - position) > 1e-6:  # Avoid tiny trades
                trade_quantity = target_position - position
                trade_value = abs(trade_quantity) * current_price

                # Apply slippage
                execution_price = current_price * (
                            1 + np.sign(trade_quantity) * slippage)

                # Calculate transaction costs
                transaction_fee = trade_value * transaction_cost

                # Check if we have enough cash for the trade
                if trade_quantity > 0:  # Buying
                    required_cash = trade_value + transaction_fee
                    if required_cash <= cash:
                        # Execute buy
                        cash -= required_cash
                        position += trade_quantity
                        position_price = ((position_price * (
                                    position - trade_quantity)) +
                                          (
                                                      execution_price * trade_quantity)) / position if position != 0 else execution_price

                        trades_list.append({
                            'date': date,
                            'type': 'BUY',
                            'quantity': trade_quantity,
                            'price': execution_price,
                            'value': trade_value,
                            'transaction_cost': transaction_fee,
                            'cash_after': cash,
                            'position_after': position
                        })
                    else:
                        logger.warning(
                            f"Insufficient cash for trade at {date}")

                else:  # Selling
                    if abs(trade_quantity) <= abs(position):
                        # Execute sell
                        proceeds = trade_value - transaction_fee
                        cash += proceeds
                        position += trade_quantity  # trade_quantity is negative

                        # Update position price if closing partial position
                        if position == 0:
                            position_price = 0.0

                        trades_list.append({
                            'date': date,
                            'type': 'SELL',
                            'quantity': trade_quantity,
                            'price': execution_price,
                            'value': trade_value,
                            'transaction_cost': transaction_fee,
                            'cash_after': cash,
                            'position_after': position
                        })
                    else:
                        logger.warning(
                            f"Insufficient position for trade at {date}")

            # Calculate portfolio value
            position_value = position * current_price
            portfolio_value = cash + position_value

            # Store daily values
            portfolio_history.append(portfolio_value)
            cash_history.append(cash)
            position_history.append(position)

        # Create results DataFrames
        self.portfolio_values = pd.Series(portfolio_history,
                                          index=common_index)
        self.positions = pd.Series(position_history, index=common_index)
        self.trades = pd.DataFrame(trades_list)

        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(prices)

        logger.info(
            f"Backtest completed. Final portfolio value: ${portfolio_value:,.2f}")

        return {
            'portfolio_values': self.portfolio_values,
            'positions': self.positions,
            'trades': self.trades,
            'performance_metrics': performance_metrics,
            'final_value': portfolio_value,
            'total_return': (
                                        portfolio_value - self.initial_capital) / self.initial_capital,
            'trade_count': len(self.trades)
        }

    def calculate_performance_metrics(self, prices: pd.Series) -> Dict[
        str, float]:
        """
        Calculate comprehensive performance metrics
        """
        if len(self.portfolio_values) == 0:
            return {}

        # Portfolio returns
        portfolio_returns = self.portfolio_values.pct_change().dropna()
        benchmark_returns = prices.pct_change().dropna()

        # Align returns
        common_index = portfolio_returns.index.intersection(
            benchmark_returns.index)
        portfolio_returns = portfolio_returns.reindex(common_index)
        benchmark_returns = benchmark_returns.reindex(common_index)

        # Basic metrics
        total_return = (self.portfolio_values.iloc[
                            -1] - self.initial_capital) / self.initial_capital
        benchmark_total_return = (prices.iloc[-1] - prices.iloc[0]) / \
                                 prices.iloc[0]

        # Annualized metrics
        trading_days = len(portfolio_returns)
        years = trading_days / 252

        annualized_return = (1 + total_return) ** (
                    1 / years) - 1 if years > 0 else 0
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0

        # Calculate beta and alpha
        if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

            benchmark_annualized_return = (1 + benchmark_total_return) ** (
                        1 / years) - 1 if years > 0 else 0
            alpha = annualized_return - beta * benchmark_annualized_return
        else:
            beta = 0
            alpha = 0

        # Drawdown analysis
        running_max = self.portfolio_values.expanding().max()
        drawdowns = (self.portfolio_values - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Calculate drawdown duration
        drawdown_periods = self._calculate_drawdown_periods(drawdowns)
        avg_drawdown_duration = np.mean([dd['duration'] for dd in
                                         drawdown_periods]) if drawdown_periods else 0
        max_drawdown_duration = max([dd['duration'] for dd in
                                     drawdown_periods]) if drawdown_periods else 0

        # Win rate and trade analysis
        if len(self.trades) > 0:
            trade_metrics = self._calculate_trade_metrics()
        else:
            trade_metrics = {}

        # Calmar ratio
        calmar_ratio = annualized_return / abs(
            max_drawdown) if max_drawdown < 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(
            downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0

        # Information ratio
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(252) if len(
            active_returns) > 0 else 0
        information_ratio = alpha / tracking_error if tracking_error > 0 else 0

        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5) if len(
            portfolio_returns) > 0 else 0
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if len(
            portfolio_returns) > 0 else 0

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            'beta': beta,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'tracking_error': tracking_error,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'benchmark_return': benchmark_total_return,
            'benchmark_volatility': benchmark_volatility,
            'excess_return': total_return - benchmark_total_return,
            'trading_days': trading_days,
            **trade_metrics
        }

        return metrics

    def _calculate_drawdown_periods(self, drawdowns: pd.Series) -> List[
        Dict[str, Any]]:
        """
        Calculate drawdown periods and their characteristics
        """
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        peak_value = None

        for date, dd in drawdowns.items():
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_date = date
                peak_value = self.portfolio_values.loc[date] / (1 + dd)

            elif dd >= 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                duration = (date - start_date).days
                trough_value = self.portfolio_values.loc[date]
                drawdown_depth = (trough_value - peak_value) / peak_value

                drawdown_periods.append({
                    'start_date': start_date,
                    'end_date': date,
                    'duration': duration,
                    'depth': drawdown_depth,
                    'peak_value': peak_value,
                    'trough_value': trough_value
                })

        return drawdown_periods

    def _calculate_trade_metrics(self) -> Dict[str, float]:
        """
        Calculate trade-specific performance metrics
        """
        if len(self.trades) == 0:
            return {}

        # Group trades into round trips
        round_trips = self._identify_round_trips()

        if not round_trips:
            return {'trade_count': len(self.trades)}

        # Calculate P&L for each round trip
        round_trip_pnl = []
        for rt in round_trips:
            entry_price = rt['entry_price']
            exit_price = rt['exit_price']
            quantity = rt['quantity']
            direction = rt['direction']

            if direction == 'LONG':
                pnl = (exit_price - entry_price) * quantity
            else:  # SHORT
                pnl = (entry_price - exit_price) * quantity

            round_trip_pnl.append(pnl)

        round_trip_pnl = np.array(round_trip_pnl)

        # Trade metrics
        win_rate = np.sum(round_trip_pnl > 0) / len(round_trip_pnl)
        avg_win = np.mean(round_trip_pnl[round_trip_pnl > 0]) if np.any(
            round_trip_pnl > 0) else 0
        avg_loss = np.mean(round_trip_pnl[round_trip_pnl < 0]) if np.any(
            round_trip_pnl < 0) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

        # Trade frequency
        trading_period = (
                    self.trades['date'].max() - self.trades['date'].min()).days
        trades_per_month = len(round_trips) / (
                    trading_period / 30) if trading_period > 0 else 0

        return {
            'trade_count': len(self.trades),
            'round_trips': len(round_trips),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades_per_month': trades_per_month,
            'total_trade_pnl': np.sum(round_trip_pnl)
        }

    def _identify_round_trips(self) -> List[Dict[str, Any]]:
        """
        Identify complete round trip trades (entry + exit)
        """
        round_trips = []
        open_positions = []

        for _, trade in self.trades.iterrows():
            if trade['type'] == 'BUY':
                open_positions.append({
                    'entry_date': trade['date'],
                    'entry_price': trade['price'],
                    'quantity': trade['quantity'],
                    'direction': 'LONG'
                })

            elif trade['type'] == 'SELL':
                remaining_quantity = abs(trade['quantity'])

                while remaining_quantity > 0 and open_positions:
                    pos = open_positions[0]

                    if pos['quantity'] <= remaining_quantity:
                        # Close entire position
                        round_trips.append({
                            'entry_date': pos['entry_date'],
                            'exit_date': trade['date'],
                            'entry_price': pos['entry_price'],
                            'exit_price': trade['price'],
                            'quantity': pos['quantity'],
                            'direction': pos['direction'],
                            'holding_period': (
                                        trade['date'] - pos['entry_date']).days
                        })

                        remaining_quantity -= pos['quantity']
                        open_positions.pop(0)

                    else:
                        # Partial close
                        round_trips.append({
                            'entry_date': pos['entry_date'],
                            'exit_date': trade['date'],
                            'entry_price': pos['entry_price'],
                            'exit_price': trade['price'],
                            'quantity': remaining_quantity,
                            'direction': pos['direction'],
                            'holding_period': (
                                        trade['date'] - pos['entry_date']).days
                        })

                        pos['quantity'] -= remaining_quantity
                        remaining_quantity = 0

        return round_trips

    def rolling_backtest(self, signals: pd.Series,
                         prices: pd.Series,
                         window_size: int = 252,
                         step_size: int = 21) -> Dict[str, Any]:
        """
        Perform rolling window backtest to assess strategy stability
        """
        logger.info(
            f"Starting rolling backtest with {window_size}-day windows")

        rolling_results = []

        for start_idx in range(0, len(signals) - window_size, step_size):
            end_idx = start_idx + window_size

            # Get window data
            window_signals = signals.iloc[start_idx:end_idx]
            window_prices = prices.iloc[start_idx:end_idx]

            try:
                # Run backtest for this window
                temp_backtester = Backtester(self.initial_capital)
                window_results = temp_backtester.run_backtest(
                    window_signals, window_prices
                )

                rolling_results.append({
                    'start_date': window_signals.index[0],
                    'end_date': window_signals.index[-1],
                    'total_return': window_results['total_return'],
                    'sharpe_ratio': window_results['performance_metrics'].get(
                        'sharpe_ratio', 0),
                    'max_drawdown': window_results['performance_metrics'].get(
                        'max_drawdown', 0),
                    'win_rate': window_results['performance_metrics'].get(
                        'win_rate', 0),
                    'trade_count': window_results['trade_count']
                })

            except Exception as e:
                logger.warning(
                    f"Rolling backtest failed for window {start_idx}: {str(e)}")
                continue

        # Analyze rolling results
        rolling_df = pd.DataFrame(rolling_results)

        if len(rolling_df) > 0:
            stability_metrics = {
                'avg_return': rolling_df['total_return'].mean(),
                'return_volatility': rolling_df['total_return'].std(),
                'positive_periods': (rolling_df[
                                         'total_return'] > 0).sum() / len(
                    rolling_df),
                'avg_sharpe': rolling_df['sharpe_ratio'].mean(),
                'sharpe_volatility': rolling_df['sharpe_ratio'].std(),
                'avg_max_drawdown': rolling_df['max_drawdown'].mean(),
                'worst_period_return': rolling_df['total_return'].min(),
                'best_period_return': rolling_df['total_return'].max()
            }
        else:
            stability_metrics = {}

        return {
            'rolling_results': rolling_df,
            'stability_metrics': stability_metrics,
            'window_count': len(rolling_df)
        }

    def monte_carlo_simulation(self, signals: pd.Series,
                               prices: pd.Series,
                               n_simulations: int = 1000,
                               bootstrap_block_size: int = 21) -> Dict[
        str, Any]:
        """
        Perform Monte Carlo simulation using block bootstrap
        """
        logger.info(f"Running {n_simulations} Monte Carlo simulations")

        # Calculate returns
        returns = prices.pct_change().dropna()

        simulation_results = []

        for sim in range(n_simulations):
            try:
                # Bootstrap returns using block bootstrap to preserve serial correlation
                bootstrapped_returns = self._block_bootstrap(returns,
                                                             bootstrap_block_size)

                # Generate synthetic price series
                synthetic_prices = (1 + bootstrapped_returns).cumprod() * \
                                   prices.iloc[0]
                synthetic_prices.index = prices.index[:len(synthetic_prices)]

                # Align signals
                sim_signals = signals.reindex(synthetic_prices.index).fillna(0)

                # Run backtest
                temp_backtester = Backtester(self.initial_capital)
                sim_results = temp_backtester.run_backtest(sim_signals,
                                                           synthetic_prices)

                simulation_results.append({
                    'total_return': sim_results['total_return'],
                    'sharpe_ratio': sim_results['performance_metrics'].get(
                        'sharpe_ratio', 0),
                    'max_drawdown': sim_results['performance_metrics'].get(
                        'max_drawdown', 0),
                    'final_value': sim_results['final_value']
                })

            except Exception as e:
                logger.warning(
                    f"Monte Carlo simulation {sim} failed: {str(e)}")
                continue

        # Analyze simulation results
        sim_df = pd.DataFrame(simulation_results)

        if len(sim_df) > 0:
            mc_metrics = {
                'mean_return': sim_df['total_return'].mean(),
                'return_std': sim_df['total_return'].std(),
                'return_5th_percentile': sim_df['total_return'].quantile(0.05),
                'return_95th_percentile': sim_df['total_return'].quantile(
                    0.95),
                'probability_of_loss': (sim_df['total_return'] < 0).mean(),
                'mean_sharpe': sim_df['sharpe_ratio'].mean(),
                'sharpe_std': sim_df['sharpe_ratio'].std(),
                'worst_drawdown': sim_df['max_drawdown'].min(),
                'successful_simulations': len(sim_df)
            }
        else:
            mc_metrics = {}

        return {
            'simulation_results': sim_df,
            'monte_carlo_metrics': mc_metrics,
            'simulations_completed': len(sim_df)
        }

    def _block_bootstrap(self, returns: pd.Series,
                         block_size: int) -> pd.Series:
        """
        Perform block bootstrap to preserve serial correlation
        """
        n_blocks = len(returns) // block_size
        bootstrapped_returns = []

        for _ in range(n_blocks):
            # Randomly select a starting point
            start_idx = np.random.randint(0, len(returns) - block_size + 1)
            block = returns.iloc[start_idx:start_idx + block_size]
            bootstrapped_returns.extend(block.values)

        # Handle remaining observations
        remaining = len(returns) % block_size
        if remaining > 0:
            start_idx = np.random.randint(0, len(returns) - remaining + 1)
            block = returns.iloc[start_idx:start_idx + remaining]
            bootstrapped_returns.extend(block.values)

        return pd.Series(bootstrapped_returns[:len(returns)])

    def generate_backtest_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive backtest report
        """
        if len(self.portfolio_values) == 0:
            return {'error': 'No backtest results available'}

        report = {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_value': self.portfolio_values.iloc[-1],
                'total_return': (self.portfolio_values.iloc[
                                     -1] - self.initial_capital) / self.initial_capital,
                'period': {
                    'start': self.portfolio_values.index[0],
                    'end': self.portfolio_values.index[-1],
                    'days': len(self.portfolio_values)
                }
            },
            'portfolio_data': {
                'values': self.portfolio_values,
                'positions': self.positions,
                'trades': self.trades
            }
        }

        return report