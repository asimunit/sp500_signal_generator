"""
Backtesting page for SP500 Signal Generator
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import settings

# API Configuration
API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"


def show_page():
    """Main function to display the backtesting page"""
    st.title("📈 Strategy Backtesting")
    st.markdown(
        "Comprehensive backtesting and performance analysis of trading strategies.")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🚀 Run Backtest", "📊 Results Analysis", "🔄 Rolling Backtest",
         "🎲 Monte Carlo"])

    with tab1:
        show_backtesting_section()

    with tab2:
        show_results_analysis_section()

    with tab3:
        show_rolling_backtest_section()

    with tab4:
        show_monte_carlo_section()


def show_backtesting_section():
    """Display main backtesting interface"""
    st.header("🚀 Run Strategy Backtest")

    # Check for available models
    try:
        response = requests.get(f"{API_BASE_URL}/models/trained")
        if response.status_code != 200:
            st.error("⚠️ No trained models available for backtesting.")
            return

        models_data = response.json()
        trained_models = models_data.get("trained_models", {})

        if not trained_models:
            st.error("⚠️ No trained models available for backtesting.")
            return

    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎛️ Backtest Configuration")

        # Model selection
        model_keys = list(trained_models.keys())
        selected_model = st.selectbox(
            "Select Model for Backtesting",
            model_keys,
            format_func=lambda
                x: f"{trained_models[x]['model_type']} - {x[:25]}..."
        )

        if selected_model:
            model_info = trained_models[selected_model]
            st.info(f"""
            **Model Type:** {model_info['model_type']}  
            **Target:** {model_info['target_column']}  
            **Created:** {model_info['created_at'][:19]}
            """)

        # Backtest parameters
        st.subheader("💰 Trading Parameters")

        col_a, col_b = st.columns(2)

        with col_a:
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=10000
            )

            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=0.01,
                max_value=1.0,
                value=0.1,
                step=0.01,
                format="%.2f"
            )

        with col_b:
            transaction_cost = st.number_input(
                "Transaction Cost (%)",
                min_value=0.0001,
                max_value=0.01,
                value=0.001,
                step=0.0001,
                format="%.4f"
            )

            slippage = st.number_input(
                "Slippage (%)",
                min_value=0.0,
                max_value=0.01,
                value=0.0001,
                step=0.0001,
                format="%.4f"
            )

        # Risk management
        st.subheader("🛡️ Risk Management")

        col_x, col_y = st.columns(2)

        with col_x:
            use_stop_loss = st.checkbox("Enable Stop Loss")
            if use_stop_loss:
                stop_loss = st.slider("Stop Loss (%)", 0.01, 0.20, 0.05, 0.01)
            else:
                stop_loss = None

        with col_y:
            use_take_profit = st.checkbox("Enable Take Profit")
            if use_take_profit:
                take_profit = st.slider("Take Profit (%)", 0.02, 0.50, 0.10,
                                        0.01)
            else:
                take_profit = None

        # Date range
        st.subheader("📅 Backtest Period")

        use_custom_dates = st.checkbox("Use Custom Date Range")
        if use_custom_dates:
            col_date1, col_date2 = st.columns(2)

            with col_date1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365)
                )

            with col_date2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime.now() - timedelta(days=1)
                )
        else:
            start_date = None
            end_date = None

    with col2:
        st.subheader("🚀 Execute Backtest")

        # Single model backtest
        if st.button("📊 Run Backtest", type="primary",
                     use_container_width=True):
            run_single_backtest(
                selected_model, initial_capital, transaction_cost, slippage,
                max_position_size, stop_loss, take_profit, start_date, end_date
            )

        # Strategy comparison
        st.subheader("⚖️ Strategy Comparison")

        comparison_models = st.multiselect(
            "Select Models for Comparison",
            model_keys,
            default=model_keys[:min(3, len(model_keys))],
            format_func=lambda x: f"{trained_models[x]['model_type']}"
        )

        if st.button("⚖️ Compare Strategies", use_container_width=True):
            if len(comparison_models) < 2:
                st.warning("Please select at least 2 models for comparison")
            else:
                run_strategy_comparison(comparison_models, initial_capital,
                                        transaction_cost)

        # Quick backtest presets
        st.subheader("⚡ Quick Presets")

        if st.button("⚡ Conservative Backtest", use_container_width=True):
            run_preset_backtest(selected_model, "conservative")

        if st.button("⚡ Aggressive Backtest", use_container_width=True):
            run_preset_backtest(selected_model, "aggressive")

        if st.button("⚡ Balanced Backtest", use_container_width=True):
            run_preset_backtest(selected_model, "balanced")

        # Benchmark comparison
        st.subheader("📊 Benchmark")

        if 'latest_backtest_task' in st.session_state:
            if st.button("📈 Compare to Buy & Hold", use_container_width=True):
                compare_to_benchmark(st.session_state['latest_backtest_task'])


def run_single_backtest(model_key, initial_capital, transaction_cost, slippage,
                        max_position_size, stop_loss, take_profit, start_date,
                        end_date):
    """Run single model backtest"""
    with st.spinner("Running backtest..."):
        try:
            # Prepare backtest request
            backtest_request = {
                "model_key": model_key,
                "initial_capital": initial_capital,
                "transaction_cost": transaction_cost,
                "slippage": slippage,
                "max_position_size": max_position_size,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "start_date": str(start_date) if start_date else None,
                "end_date": str(end_date) if end_date else None
            }

            # Start backtest
            response = requests.post(
                f"{API_BASE_URL}/backtest/run/{model_key}",
                json=backtest_request
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    task_id = result["task_id"]
                    st.success(f"✅ Backtest started! Task ID: {task_id}")

                    # Store task ID for monitoring
                    st.session_state['latest_backtest_task'] = task_id
                    if "backtest_tasks" not in st.session_state:
                        st.session_state.backtest_tasks = []
                    st.session_state.backtest_tasks.append(task_id)

                    # Monitor progress
                    monitor_backtest_progress(task_id)

                else:
                    st.error("Failed to start backtest")
            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")


def run_preset_backtest(model_key, preset_type):
    """Run backtest with predefined parameters"""
    presets = {
        "conservative": {
            "initial_capital": 100000,
            "transaction_cost": 0.002,
            "max_position_size": 0.05,
            "stop_loss": 0.03,
            "take_profit": 0.08
        },
        "aggressive": {
            "initial_capital": 100000,
            "transaction_cost": 0.001,
            "max_position_size": 0.20,
            "stop_loss": None,
            "take_profit": None
        },
        "balanced": {
            "initial_capital": 100000,
            "transaction_cost": 0.001,
            "max_position_size": 0.10,
            "stop_loss": 0.05,
            "take_profit": 0.12
        }
    }

    params = presets.get(preset_type, presets["balanced"])

    with st.spinner(f"Running {preset_type} backtest..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/backtest/run/{model_key}",
                json=params
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    task_id = result["task_id"]
                    st.success(f"✅ {preset_type.title()} backtest started!")
                    st.session_state['latest_backtest_task'] = task_id
                    monitor_backtest_progress(task_id)
                else:
                    st.error(f"Failed to start {preset_type} backtest")
            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error(f"Error running {preset_type} backtest: {str(e)}")


def run_strategy_comparison(model_keys, initial_capital, transaction_cost):
    """Run strategy comparison backtest"""
    with st.spinner("Running strategy comparison..."):
        try:
            comparison_request = {
                "model_keys": model_keys,
                "initial_capital": initial_capital,
                "transaction_cost": transaction_cost
            }

            response = requests.post(
                f"{API_BASE_URL}/backtest/compare",
                json=comparison_request
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    task_id = result["task_id"]
                    st.success(f"✅ Strategy comparison started!")
                    st.session_state['latest_comparison_task'] = task_id
                    monitor_backtest_progress(task_id)
                else:
                    st.error("Failed to start strategy comparison")
            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error(f"Error running comparison: {str(e)}")


def monitor_backtest_progress(task_id):
    """Monitor backtest progress with real-time updates"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    max_attempts = 60  # Maximum 5 minutes
    attempt = 0

    while attempt < max_attempts:
        try:
            response = requests.get(
                f"{API_BASE_URL}/backtest/status/{task_id}")
            if response.status_code == 200:
                status_data = response.json()

                status = status_data.get("status", "unknown")
                progress = status_data.get("progress", 0)

                # Update progress bar
                progress_placeholder.progress(progress / 100)
                status_placeholder.info(
                    f"Status: {status.title()} - {progress}%")

                if status == "completed":
                    status_placeholder.success(
                        "✅ Backtest completed successfully!")

                    # Display quick results
                    final_value = status_data.get("final_value")
                    total_return = status_data.get("total_return")

                    if final_value and total_return:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Final Portfolio Value",
                                      f"${final_value:,.2f}")
                        with col2:
                            st.metric("Total Return", f"{total_return:.2%}")

                    break

                elif status == "failed":
                    error = status_data.get("error", "Unknown error")
                    status_placeholder.error(f"❌ Backtest failed: {error}")
                    break

                elif status in ["started", "running"]:
                    time.sleep(5)  # Wait 5 seconds before next check

                else:
                    status_placeholder.warning(f"⚠️ Unknown status: {status}")

            else:
                status_placeholder.error("❌ Could not get backtest status")
                break

        except Exception as e:
            status_placeholder.error(f"❌ Error monitoring progress: {str(e)}")
            break

        attempt += 1

    if attempt >= max_attempts:
        status_placeholder.warning(
            "⏰ Monitoring timeout. Check Results Analysis tab for updates.")


def compare_to_benchmark(task_id):
    """Compare backtest results to buy-and-hold benchmark"""
    try:
        response = requests.get(f"{API_BASE_URL}/backtest/benchmark/{task_id}")
        if response.status_code == 200:
            benchmark_data = response.json()

            if benchmark_data.get("success"):
                comparison = benchmark_data["benchmark_comparison"]

                st.subheader("📊 Benchmark Comparison Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🤖 Strategy Performance")
                    strategy = comparison["strategy"]

                    st.metric("Total Return",
                              f"{strategy['total_return']:.2%}")
                    st.metric("Sharpe Ratio",
                              f"{strategy['sharpe_ratio']:.2f}")
                    st.metric("Max Drawdown",
                              f"{strategy['max_drawdown']:.2%}")

                with col2:
                    st.markdown("### 📈 Buy & Hold Benchmark")
                    benchmark = comparison["benchmark"]

                    st.metric("Total Return",
                              f"{benchmark['total_return']:.2%}")
                    st.metric("Sharpe Ratio",
                              f"{benchmark['sharpe_ratio']:.2f}")
                    st.metric("Volatility", f"{benchmark['volatility']:.2%}")

                # Outperformance analysis
                outperformance = comparison["outperformance"]

                st.subheader("🏆 Outperformance Analysis")

                excess_return = outperformance["excess_return"]
                outperformed = outperformance["outperformed"]

                if outperformed:
                    st.success(
                        f"✅ Strategy outperformed benchmark by {excess_return:.2%}")
                else:
                    st.warning(
                        f"⚠️ Strategy underperformed benchmark by {abs(excess_return):.2%}")

            else:
                st.error("Failed to get benchmark comparison")
        else:
            st.error("Benchmark comparison not available")

    except Exception as e:
        st.error(f"Error comparing to benchmark: {str(e)}")


def show_results_analysis_section():
    """Display backtest results analysis"""
    st.header("📊 Backtest Results Analysis")

    # Get list of backtest results
    try:
        response = requests.get(f"{API_BASE_URL}/backtest/list")
        if response.status_code == 200:
            results_data = response.json()

            if results_data.get("success"):
                backtest_results = results_data.get("backtest_results", {})

                if not backtest_results:
                    st.info(
                        "No backtest results available. Run a backtest first.")
                    return

                # Select result to analyze
                selected_result = st.selectbox(
                    "Select Backtest Result to Analyze",
                    list(backtest_results.keys()),
                    format_func=lambda
                        x: f"{backtest_results[x].get('type', 'single')} - {x}"
                )

                if selected_result:
                    analyze_backtest_result(selected_result)

            else:
                st.error("Failed to load backtest results")
        else:
            st.warning("No backtest results available")

    except Exception as e:
        st.error(f"Error loading backtest results: {str(e)}")


def analyze_backtest_result(result_id):
    """Analyze specific backtest result"""
    try:
        response = requests.get(f"{API_BASE_URL}/backtest/results/{result_id}")
        if response.status_code == 200:
            result_data = response.json()

            if result_data.get("success"):
                backtest_results = result_data["backtest_results"]

                # Performance summary
                st.subheader("💰 Performance Summary")

                performance_summary = backtest_results["performance_summary"]
                performance_metrics = backtest_results["performance_metrics"]

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    final_value = performance_summary["final_value"]
                    st.metric("Final Value", f"${final_value:,.2f}")

                with col2:
                    total_return = performance_summary["total_return"]
                    st.metric("Total Return", f"{total_return:.2%}")

                with col3:
                    sharpe_ratio = performance_metrics.get("sharpe_ratio", 0)
                    st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

                with col4:
                    max_drawdown = performance_metrics.get("max_drawdown", 0)
                    st.metric("Max Drawdown", f"{max_drawdown:.2%}")

                # Detailed metrics
                st.subheader("📈 Detailed Performance Metrics")

                col_a, col_b = st.columns(2)

                with col_a:
                    st.markdown("### 📊 Return Metrics")

                    annualized_return = performance_metrics.get(
                        "annualized_return", 0)
                    st.metric("Annualized Return", f"{annualized_return:.2%}")

                    volatility = performance_metrics.get("volatility", 0)
                    st.metric("Volatility", f"{volatility:.2%}")

                    calmar_ratio = performance_metrics.get("calmar_ratio", 0)
                    st.metric("Calmar Ratio", f"{calmar_ratio:.2f}")

                with col_b:
                    st.markdown("### 🎯 Trade Metrics")

                    trade_count = performance_summary.get("trade_count", 0)
                    st.metric("Total Trades", trade_count)

                    win_rate = performance_metrics.get("win_rate", 0)
                    st.metric("Win Rate", f"{win_rate:.2%}")

                    profit_factor = performance_metrics.get("profit_factor", 0)
                    st.metric("Profit Factor", f"{profit_factor:.2f}")

                # Portfolio value chart
                st.subheader("📈 Portfolio Value Over Time")

                portfolio_data = backtest_results["portfolio_data"]
                dates = pd.to_datetime(portfolio_data["dates"])
                portfolio_values = portfolio_data["portfolio_values"]

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=portfolio_values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='blue', width=2)
                ))

                # Add benchmark line (buy and hold)
                initial_value = portfolio_values[0]

                fig.update_layout(
                    title="Portfolio Value Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Drawdown chart
                st.subheader("📉 Drawdown Analysis")

                # Calculate drawdowns
                portfolio_series = pd.Series(portfolio_values, index=dates)
                running_max = portfolio_series.expanding().max()
                drawdowns = (
                                        portfolio_series - running_max) / running_max * 100

                fig_dd = go.Figure()

                fig_dd.add_trace(go.Scatter(
                    x=dates,
                    y=drawdowns,
                    mode='lines',
                    name='Drawdown %',
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.3)',
                    line=dict(color='red')
                ))

                fig_dd.update_layout(
                    title="Portfolio Drawdown Over Time",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    height=400
                )

                st.plotly_chart(fig_dd, use_container_width=True)

                # Trade analysis
                trades = backtest_results.get("trades", [])
                if trades:
                    st.subheader("💼 Trade Analysis")

                    trades_df = pd.DataFrame(trades)

                    # Trade statistics
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 📊 Trade Statistics")
                        st.dataframe(trades_df.head(10),
                                     use_container_width=True)

                    with col2:
                        st.markdown("### 📈 Trade Distribution")

                        if "type" in trades_df.columns:
                            trade_counts = trades_df["type"].value_counts()

                            fig_pie = go.Figure(data=[go.Pie(
                                labels=trade_counts.index,
                                values=trade_counts.values,
                                hole=0.3
                            )])

                            fig_pie.update_layout(
                                title="Buy vs Sell Trades",
                                height=300
                            )

                            st.plotly_chart(fig_pie, use_container_width=True)

            else:
                st.error("Failed to load backtest results")
        else:
            st.error("Backtest results not found")

    except Exception as e:
        st.error(f"Error analyzing backtest result: {str(e)}")


def show_rolling_backtest_section():
    """Display rolling backtest interface"""
    st.header("🔄 Rolling Window Backtest")
    st.markdown(
        "Assess strategy stability across different market conditions using rolling window analysis.")

    # Check for available models
    try:
        response = requests.get(f"{API_BASE_URL}/models/trained")
        if response.status_code != 200:
            st.error("⚠️ No trained models available.")
            return

        models_data = response.json()
        trained_models = models_data.get("trained_models", {})

        if not trained_models:
            st.error("⚠️ No trained models available.")
            return

    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎛️ Rolling Backtest Configuration")

        # Model selection
        model_keys = list(trained_models.keys())
        selected_model = st.selectbox(
            "Select Model",
            model_keys,
            format_func=lambda
                x: f"{trained_models[x]['model_type']} - {x[:20]}..."
        )

        # Rolling parameters
        window_size = st.slider(
            "Window Size (days)",
            min_value=60,
            max_value=500,
            value=252,
            step=30,
            help="Size of each rolling window (252 ≈ 1 trading year)"
        )

        step_size = st.slider(
            "Step Size (days)",
            min_value=1,
            max_value=60,
            value=21,
            step=7,
            help="Number of days to move window forward"
        )

        # Trading parameters
        initial_capital = st.number_input("Initial Capital ($)", value=100000,
                                          step=10000)
        transaction_cost = st.number_input("Transaction Cost (%)", value=0.001,
                                           step=0.0001, format="%.4f")

    with col2:
        st.subheader("🚀 Execute Rolling Backtest")

        if st.button("🔄 Start Rolling Backtest", type="primary",
                     use_container_width=True):
            run_rolling_backtest(selected_model, window_size, step_size,
                                 initial_capital, transaction_cost)

        # Show rolling backtest benefits
        st.subheader("💡 Benefits of Rolling Backtest")
        st.info("""
        **Rolling window backtesting helps assess:**
        - Strategy consistency across time periods
        - Performance stability in different market conditions
        - Overfitting detection
        - Parameter sensitivity analysis
        - Risk-adjusted returns over time
        """)


def run_rolling_backtest(model_key, window_size, step_size, initial_capital,
                         transaction_cost):
    """Execute rolling backtest"""
    with st.spinner("Running rolling backtest analysis..."):
        try:
            rolling_request = {
                "model_key": model_key,
                "window_size": window_size,
                "step_size": step_size,
                "initial_capital": initial_capital,
                "transaction_cost": transaction_cost
            }

            response = requests.post(
                f"{API_BASE_URL}/backtest/rolling/{model_key}",
                json=rolling_request
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    task_id = result["task_id"]
                    st.success(
                        f"✅ Rolling backtest started! Task ID: {task_id}")
                    st.session_state['latest_rolling_task'] = task_id
                    monitor_backtest_progress(task_id)
                else:
                    st.error("Failed to start rolling backtest")
            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error(f"Error running rolling backtest: {str(e)}")


def show_monte_carlo_section():
    """Display Monte Carlo simulation interface"""
    st.header("🎲 Monte Carlo Simulation")
    st.markdown(
        "Test strategy robustness using Monte Carlo simulation with bootstrapped returns.")

    # Check for available models
    try:
        response = requests.get(f"{API_BASE_URL}/models/trained")
        if response.status_code != 200:
            st.error("⚠️ No trained models available.")
            return

        models_data = response.json()
        trained_models = models_data.get("trained_models", {})

        if not trained_models:
            st.error("⚠️ No trained models available.")
            return

    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎛️ Monte Carlo Configuration")

        # Model selection
        model_keys = list(trained_models.keys())
        selected_model = st.selectbox(
            "Select Model for Simulation",
            model_keys,
            format_func=lambda
                x: f"{trained_models[x]['model_type']} - {x[:20]}..."
        )

        # Simulation parameters
        n_simulations = st.selectbox(
            "Number of Simulations",
            [100, 500, 1000, 2000, 5000],
            index=2,
            help="More simulations = better statistical significance"
        )

        bootstrap_block_size = st.slider(
            "Bootstrap Block Size",
            min_value=5,
            max_value=50,
            value=21,
            help="Size of blocks for bootstrap sampling (preserves correlation)"
        )

        # Trading parameters
        initial_capital = st.number_input("Initial Capital ($)", value=100000,
                                          step=10000)

    with col2:
        st.subheader("🚀 Execute Simulation")

        if st.button("🎲 Start Monte Carlo Simulation", type="primary",
                     use_container_width=True):
            run_monte_carlo_simulation(
                selected_model, n_simulations, initial_capital,
                bootstrap_block_size
            )

        # Show Monte Carlo benefits
        st.subheader("💡 Monte Carlo Benefits")
        st.info("""
        **Monte Carlo simulation helps assess:**
        - Strategy performance distribution
        - Probability of losses/gains
        - Risk-adjusted performance metrics
        - Worst-case scenario analysis
        - Statistical significance of results
        """)

        # Expected simulation time
        expected_time = n_simulations * 0.5  # Rough estimate: 0.5 seconds per simulation
        st.metric("⏱️ Expected Runtime", f"~{expected_time / 60:.1f} minutes")


def run_monte_carlo_simulation(model_key, n_simulations, initial_capital,
                               bootstrap_block_size):
    """Execute Monte Carlo simulation"""
    with st.spinner(f"Running {n_simulations} Monte Carlo simulations..."):
        try:
            mc_request = {
                "model_key": model_key,
                "n_simulations": n_simulations,
                "initial_capital": initial_capital,
                "bootstrap_block_size": bootstrap_block_size
            }

            response = requests.post(
                f"{API_BASE_URL}/backtest/montecarlo/{model_key}",
                json=mc_request
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    task_id = result["task_id"]
                    st.success(
                        f"✅ Monte Carlo simulation started! Task ID: {task_id}")
                    st.session_state['latest_mc_task'] = task_id
                    monitor_backtest_progress(task_id)
                else:
                    st.error("Failed to start Monte Carlo simulation")
            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error(f"Error running Monte Carlo simulation: {str(e)}")


# Additional helper functions for active task monitoring
def show_active_tasks():
    """Show active backtesting tasks"""
    if "backtest_tasks" in st.session_state and st.session_state.backtest_tasks:
        st.subheader("🔄 Active Tasks")

        for task_id in st.session_state.backtest_tasks[:]:
            try:
                response = requests.get(
                    f"{API_BASE_URL}/backtest/status/{task_id}")
                if response.status_code == 200:
                    status_data = response.json()
                    status = status_data.get("status", "unknown")

                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.write(f"**Task:** {task_id[:20]}...")

                    with col2:
                        if status == "completed":
                            st.success("✅ Done")
                            st.session_state.backtest_tasks.remove(task_id)
                        elif status == "failed":
                            st.error("❌ Failed")
                            st.session_state.backtest_tasks.remove(task_id)
                        else:
                            st.info(f"🔄 {status}")

                    with col3:
                        progress = status_data.get("progress", 0)
                        st.progress(progress / 100)

            except Exception as e:
                st.warning(f"Could not get status for {task_id}")


# Initialize session state
if "backtest_tasks" not in st.session_state:
    st.session_state.backtest_tasks = []