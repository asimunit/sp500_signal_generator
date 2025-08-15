"""
Signal Generation page for SP500 Signal Generator
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import settings

# API Configuration
API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"


def show_page():
    """Main function to display the signal generation page"""
    st.title("🎯 Signal Generation")
    st.markdown(
        "Generate trading signals using trained models with dynamic thresholds and advanced filtering.")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🎯 Generate Signals", "📊 Signal Analysis", "🔮 Forecasts",
         "📈 Signal Performance"])

    with tab1:
        show_signal_generation_section()

    with tab2:
        show_signal_analysis_section()

    with tab3:
        show_forecasting_section()

    with tab4:
        show_signal_performance_section()


def show_signal_generation_section():
    """Display signal generation interface"""
    st.header("🎯 Generate Trading Signals")

    # Check for available models
    try:
        response = requests.get(f"{API_BASE_URL}/models/trained")
        if response.status_code != 200:
            st.error(
                "⚠️ No trained models available. Please train models first.")
            return

        models_data = response.json()
        if not models_data.get("success") or not models_data.get(
                "trained_models"):
            st.error(
                "⚠️ No trained models available. Please train models first.")
            return

        trained_models = models_data["trained_models"]

    except Exception as e:
        st.error(f"⚠️ Error loading models: {str(e)}")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎛️ Signal Configuration")

        # Model selection
        model_keys = list(trained_models.keys())
        selected_model = st.selectbox(
            "Select Model",
            model_keys,
            format_func=lambda
                x: f"{trained_models[x]['model_type']} - {x[:20]}..."
        )

        if selected_model:
            model_info = trained_models[selected_model]
            st.info(f"""
            **Model Type:** {model_info['model_type']}  
            **Target:** {model_info['target_column']}  
            **Created:** {model_info['created_at'][:19]}
            """)

        # Signal parameters
        st.subheader("⚙️ Signal Parameters")

        threshold_method = st.selectbox(
            "Threshold Method",
            ["adaptive", "percentile", "volatility_adjusted", "fixed"],
            help="Method for calculating signal thresholds"
        )

        smoothing_window = st.slider(
            "Smoothing Window",
            min_value=1,
            max_value=10,
            value=5,
            help="Window size for signal smoothing"
        )

        apply_filters = st.checkbox(
            "Apply Signal Filters",
            value=True,
            help="Apply momentum and volatility filters"
        )

        # Advanced parameters
        with st.expander("🔧 Advanced Parameters"):
            volatility_filter = st.checkbox("Volatility Filter", value=True)
            momentum_filter = st.checkbox("Momentum Filter", value=True)
            regime_filter = st.checkbox("Market Regime Filter", value=False)

            if threshold_method == "fixed":
                buy_threshold = st.number_input("Buy Threshold", value=0.01,
                                                step=0.001)
                sell_threshold = st.number_input("Sell Threshold", value=-0.01,
                                                 step=0.001)

    with col2:
        st.subheader("🚀 Generate Signals")

        if st.button("🎯 Generate Signals", type="primary",
                     use_container_width=True):
            with st.spinner("Generating trading signals..."):
                try:
                    # Prepare signal request
                    signal_request = {
                        "model_key": selected_model,
                        "threshold_method": threshold_method,
                        "smoothing_window": smoothing_window,
                        "apply_filters": apply_filters
                    }

                    # Generate signals
                    response = requests.post(
                        f"{API_BASE_URL}/predictions/signals/{selected_model}",
                        json=signal_request
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            st.success("✅ Signals generated successfully!")

                            # Store signals in session state
                            st.session_state['current_signals'] = result
                            st.session_state['selected_model'] = selected_model

                            # Display signal summary
                            signal_summary = result.get("signal_summary", {})
                            thresholds = result.get("thresholds", {})

                            st.subheader("📊 Signal Summary")

                            col_a, col_b, col_c = st.columns(3)

                            with col_a:
                                buy_signals = signal_summary.get("buy_signals",
                                                                 0)
                                st.metric("🟢 Buy Signals", buy_signals)

                            with col_b:
                                sell_signals = signal_summary.get(
                                    "sell_signals", 0)
                                st.metric("🔴 Sell Signals", sell_signals)

                            with col_c:
                                hold_periods = signal_summary.get(
                                    "hold_periods", 0)
                                st.metric("⚪ Hold Periods", hold_periods)

                            # Threshold information
                            st.subheader("🎚️ Signal Thresholds")

                            col_x, col_y = st.columns(2)
                            with col_x:
                                st.metric("📈 Buy Threshold",
                                          f"{thresholds.get('buy_threshold', 0):.4f}")
                            with col_y:
                                st.metric("📉 Sell Threshold",
                                          f"{thresholds.get('sell_threshold', 0):.4f}")

                            # Auto-refresh to show signals
                            st.rerun()

                        else:
                            st.error("Signal generation failed")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except Exception as e:
                    st.error(f"Error generating signals: {str(e)}")

        # Ensemble signal generation
        st.subheader("🎭 Ensemble Signals")

        ensemble_models = st.multiselect(
            "Select Models for Ensemble",
            model_keys,
            default=model_keys[:min(3, len(model_keys))],
            format_func=lambda x: f"{trained_models[x]['model_type']}"
        )

        ensemble_method = st.selectbox(
            "Ensemble Method",
            ["equal_weight", "performance_weight", "custom_weight"]
        )

        if st.button("🎭 Generate Ensemble Signals", use_container_width=True):
            if len(ensemble_models) < 2:
                st.warning(
                    "Please select at least 2 models for ensemble signals")
            else:
                generate_ensemble_signals(ensemble_models, ensemble_method)

        # Quick signal generation
        st.subheader("⚡ Quick Signal Generation")

        if st.button("⚡ Quick Conservative", use_container_width=True):
            quick_generate_signals(selected_model, "conservative")

        if st.button("⚡ Quick Aggressive", use_container_width=True):
            quick_generate_signals(selected_model, "aggressive")

        if st.button("⚡ Quick Balanced", use_container_width=True):
            quick_generate_signals(selected_model, "balanced")


def quick_generate_signals(model_key, style):
    """Quick signal generation with predefined parameters"""
    style_params = {
        "conservative": {
            "threshold_method": "percentile",
            "smoothing_window": 7,
            "apply_filters": True
        },
        "aggressive": {
            "threshold_method": "fixed",
            "smoothing_window": 3,
            "apply_filters": False
        },
        "balanced": {
            "threshold_method": "adaptive",
            "smoothing_window": 5,
            "apply_filters": True
        }
    }

    params = style_params.get(style, style_params["balanced"])

    try:
        response = requests.post(
            f"{API_BASE_URL}/predictions/signals/{model_key}",
            json=params
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                st.success(f"✅ {style.title()} signals generated!")
                st.session_state['current_signals'] = result
                st.session_state['selected_model'] = model_key
            else:
                st.error(f"Failed to generate {style} signals")
        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.error(f"Error generating {style} signals: {str(e)}")


def generate_ensemble_signals(ensemble_models, ensemble_method):
    """Generate ensemble signals from multiple models"""
    try:
        # First generate ensemble forecast
        ensemble_request = {
            "model_keys": ensemble_models,
            "steps": 20,
            "ensemble_method": ensemble_method
        }

        response = requests.post(
            f"{API_BASE_URL}/predictions/ensemble/forecast",
            json=ensemble_request
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                st.success("✅ Ensemble signals generated!")
                st.session_state['ensemble_forecast'] = result
            else:
                st.error("Failed to generate ensemble signals")
        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.error(f"Error generating ensemble signals: {str(e)}")


def show_signal_analysis_section():
    """Display signal analysis and visualization"""
    st.header("📊 Signal Analysis")

    # Check if signals are available
    if 'current_signals' not in st.session_state:
        st.warning("No signals available. Please generate signals first.")
        return

    signals_data = st.session_state['current_signals']
    signals_info = signals_data.get("signals", {})

    # Signal visualization
    st.subheader("📈 Signal Visualization")

    try:
        # Get price data for overlay
        response = requests.get(
            f"{API_BASE_URL}/data/price-data?columns=close")
        if response.status_code == 200:
            price_data = response.json()

            if price_data.get("success"):
                price_dict = price_data["data"]
                price_df = pd.DataFrame(price_dict["data"])
                price_df.index = pd.to_datetime(price_dict["dates"])

                # Get signal data
                signal_dates = pd.to_datetime(signals_info["dates"])
                signal_values = signals_info["signals"]

                # Align data
                signal_df = pd.DataFrame({
                    'signals': signal_values
                }, index=signal_dates)

                # Merge with price data
                combined_df = price_df.join(signal_df, how='outer')
                combined_df = combined_df.fillna(method='ffill').dropna()

                # Create signal visualization
                fig = go.Figure()

                # Price line
                fig.add_trace(go.Scatter(
                    x=combined_df.index,
                    y=combined_df['close'],
                    mode='lines',
                    name='S&P 500 Price',
                    line=dict(color='blue', width=1)
                ))

                # Buy signals
                buy_mask = combined_df['signals'] == 1
                if buy_mask.any():
                    fig.add_trace(go.Scatter(
                        x=combined_df.index[buy_mask],
                        y=combined_df['close'][buy_mask],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(color='green', size=10,
                                    symbol='triangle-up')
                    ))

                # Sell signals
                sell_mask = combined_df['signals'] == -1
                if sell_mask.any():
                    fig.add_trace(go.Scatter(
                        x=combined_df.index[sell_mask],
                        y=combined_df['close'][sell_mask],
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(color='red', size=10,
                                    symbol='triangle-down')
                    ))

                fig.update_layout(
                    title="Trading Signals on S&P 500 Price Chart",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=500,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

                # Signal distribution
                st.subheader("📊 Signal Distribution")

                signal_counts = pd.Series(signal_values).value_counts()
                signal_names = {-1: "Sell", 0: "Hold", 1: "Buy"}

                fig_pie = go.Figure(data=[go.Pie(
                    labels=[signal_names.get(k, str(k)) for k in
                            signal_counts.index],
                    values=signal_counts.values,
                    hole=0.3,
                    marker_colors=['red', 'gray', 'green']
                )])

                fig_pie.update_layout(
                    title="Signal Distribution",
                    height=400
                )

                st.plotly_chart(fig_pie, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating signal visualization: {str(e)}")

    # Signal statistics
    st.subheader("📋 Signal Statistics")

    signal_summary = signals_data.get("signal_summary", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_periods = signal_summary.get("total_periods", 0)
        st.metric("📊 Total Periods", total_periods)

    with col2:
        signal_frequency = signal_summary.get("signal_frequency", 0)
        st.metric("📈 Signal Frequency", f"{signal_frequency:.2%}")

    with col3:
        avg_holding = signal_summary.get("avg_holding_period", 0)
        st.metric("⏱️ Avg Holding Period", f"{avg_holding:.1f} days")

    with col4:
        activity_ratio = signal_summary.get("activity_ratio", 0)
        st.metric("🏃‍♂️ Activity Ratio", f"{activity_ratio:.2%}")

    # Detailed signal analysis
    with st.expander("🔍 Detailed Signal Analysis"):
        # Signal parameters used
        st.subheader("⚙️ Parameters Used")
        parameters = signals_data.get("parameters", {})

        for key, value in parameters.items():
            st.write(f"**{key.replace('_', ' ').title()}:** {value}")

        # Threshold information
        st.subheader("🎚️ Dynamic Thresholds")
        thresholds = signals_data.get("thresholds", {})

        for key, value in thresholds.items():
            if isinstance(value, (int, float)):
                st.write(f"**{key.replace('_', ' ').title()}:** {value:.4f}")

        # Signal quality metrics
        st.subheader("🏆 Signal Quality Metrics")
        st.info("Signal quality metrics will be calculated after backtesting.")


def show_forecasting_section():
    """Display forecasting results"""
    st.header("🔮 Forecasting Results")

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
        st.subheader("🎛️ Forecast Configuration")

        # Model selection
        model_keys = list(trained_models.keys())
        selected_model = st.selectbox(
            "Select Model for Forecasting",
            model_keys,
            format_func=lambda
                x: f"{trained_models[x]['model_type']} - {x[:20]}..."
        )

        # Forecast parameters
        forecast_steps = st.slider(
            "Forecast Horizon (days)",
            min_value=5,
            max_value=60,
            value=20,
            help="Number of days to forecast ahead"
        )

        confidence_level = st.selectbox(
            "Confidence Level",
            [0.05, 0.10, 0.20],
            index=0,
            format_func=lambda x: f"{int((1 - x) * 100)}%"
        )

    with col2:
        st.subheader("🚀 Generate Forecast")

        if st.button("🔮 Generate Forecast", type="primary",
                     use_container_width=True):
            with st.spinner("Generating forecast..."):
                try:
                    # Prepare forecast request
                    forecast_request = {
                        "model_key": selected_model,
                        "steps": forecast_steps,
                        "confidence_level": confidence_level
                    }

                    # Generate forecast
                    response = requests.post(
                        f"{API_BASE_URL}/predictions/forecast/{selected_model}",
                        json=forecast_request
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            st.success("✅ Forecast generated successfully!")

                            # Store forecast in session state
                            st.session_state['current_forecast'] = result

                            # Display forecast summary
                            forecast_data = result.get("forecast", {})
                            forecast_stats = result.get("forecast_stats", {})

                            st.subheader("📊 Forecast Summary")

                            col_a, col_b, col_c = st.columns(3)

                            with col_a:
                                mean_forecast = forecast_stats.get("mean", 0)
                                st.metric("📈 Mean Forecast",
                                          f"{mean_forecast:.4f}")

                            with col_b:
                                trend = forecast_stats.get("trend", "unknown")
                                st.metric("📊 Trend", trend.title())

                            with col_c:
                                std_forecast = forecast_stats.get("std", 0)
                                st.metric("📏 Volatility",
                                          f"{std_forecast:.4f}")

                            # Auto-refresh to show forecast
                            st.rerun()

                        else:
                            st.error("Forecast generation failed")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")

    # Display forecast if available
    if 'current_forecast' in st.session_state:
        show_forecast_visualization()

    # Ensemble forecasting
    st.subheader("🎭 Ensemble Forecasting")

    ensemble_models = st.multiselect(
        "Select Models for Ensemble Forecast",
        model_keys,
        default=model_keys[:min(3, len(model_keys))],
        format_func=lambda x: f"{trained_models[x]['model_type']}"
    )

    if st.button("🎭 Generate Ensemble Forecast") and len(ensemble_models) >= 2:
        generate_ensemble_forecast(ensemble_models, forecast_steps)


def show_forecast_visualization():
    """Display forecast visualization"""
    if 'current_forecast' not in st.session_state:
        return

    st.subheader("📈 Forecast Visualization")

    try:
        forecast_data = st.session_state['current_forecast']
        forecast_result = forecast_data.get("forecast", {})

        # Get historical price data
        response = requests.get(
            f"{API_BASE_URL}/data/price-data?columns=close")
        if response.status_code == 200:
            price_data = response.json()

            if price_data.get("success"):
                price_dict = price_data["data"]
                historical_df = pd.DataFrame(price_dict["data"])
                historical_df.index = pd.to_datetime(price_dict["dates"])

                # Get forecast data
                forecast_values = forecast_result.get("forecast_values", [])
                forecast_dates = pd.to_datetime(
                    forecast_result.get("forecast_dates", []))

                # Create visualization
                fig = go.Figure()

                # Historical data (last 60 days)
                historical_recent = historical_df.tail(60)
                fig.add_trace(go.Scatter(
                    x=historical_recent.index,
                    y=historical_recent['close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))

                # Forecast
                if forecast_values and len(forecast_dates) > 0:
                    fig.add_trace(go.Scatter(
                        x=forecast_dates,
                        y=forecast_values,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', width=2, dash='dash')
                    ))

                    # Confidence intervals if available
                    lower_bounds = forecast_result.get("lower_bounds", [])
                    upper_bounds = forecast_result.get("upper_bounds", [])

                    if lower_bounds and upper_bounds:
                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=upper_bounds,
                            mode='lines',
                            name='Upper CI',
                            line=dict(color='rgba(255,0,0,0.2)'),
                            showlegend=False
                        ))

                        fig.add_trace(go.Scatter(
                            x=forecast_dates,
                            y=lower_bounds,
                            mode='lines',
                            name='Lower CI',
                            line=dict(color='rgba(255,0,0,0.2)'),
                            fill='tonexty',
                            fillcolor='rgba(255,0,0,0.1)',
                            showlegend=False
                        ))

                fig.update_layout(
                    title="Price Forecast with Confidence Intervals",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=500,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating forecast visualization: {str(e)}")


def generate_ensemble_forecast(ensemble_models, forecast_steps):
    """Generate ensemble forecast"""
    try:
        ensemble_request = {
            "model_keys": ensemble_models,
            "steps": forecast_steps,
            "ensemble_method": "equal_weight"
        }

        response = requests.post(
            f"{API_BASE_URL}/predictions/ensemble/forecast",
            json=ensemble_request
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                st.success("✅ Ensemble forecast generated!")
                st.session_state['ensemble_forecast'] = result
            else:
                st.error("Failed to generate ensemble forecast")
        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.error(f"Error generating ensemble forecast: {str(e)}")


def show_signal_performance_section():
    """Display signal performance analysis"""
    st.header("📈 Signal Performance Analysis")

    # Check if signals are available
    if 'current_signals' not in st.session_state:
        st.warning("No signals available. Please generate signals first.")
        return

    selected_model = st.session_state.get('selected_model')

    if not selected_model:
        st.warning("No model selected for performance analysis.")
        return

    st.subheader("📊 Historical Performance Analysis")

    # Rolling forecast accuracy
    try:
        response = requests.get(
            f"{API_BASE_URL}/predictions/accuracy/{selected_model}")
        if response.status_code == 200:
            accuracy_data = response.json()

            if accuracy_data.get("success"):
                accuracy_metrics = accuracy_data.get("accuracy_metrics", {})

                # Display accuracy metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    mae = accuracy_metrics.get("mean_absolute_error", 0)
                    st.metric("📏 Mean Absolute Error", f"{mae:.4f}")

                with col2:
                    rmse = accuracy_metrics.get("root_mean_squared_error", 0)
                    st.metric("📐 Root Mean Squared Error", f"{rmse:.4f}")

                with col3:
                    dir_acc = accuracy_metrics.get("directional_accuracy", 0)
                    st.metric("🎯 Directional Accuracy", f"{dir_acc:.2%}")

                with col4:
                    correlation = accuracy_metrics.get("correlation", 0)
                    st.metric("🔗 Correlation", f"{correlation:.3f}")

                # Accuracy trend analysis
                if "recent_30d_mae" in accuracy_metrics and "older_period_mae" in accuracy_metrics:
                    st.subheader("📈 Accuracy Trend")

                    recent_mae = accuracy_metrics["recent_30d_mae"]
                    older_mae = accuracy_metrics["older_period_mae"]
                    trend = accuracy_metrics.get("accuracy_trend", "stable")

                    trend_color = "green" if trend == "improving" else "red" if trend == "declining" else "blue"

                    st.markdown(f"""
                    **Recent Period MAE:** {recent_mae:.4f}  
                    **Previous Period MAE:** {older_mae:.4f}  
                    **Trend:** <span style="color: {trend_color};">{trend.title()}</span>
                    """, unsafe_allow_html=True)

            else:
                st.warning("Could not calculate accuracy metrics")

        else:
            st.warning("Accuracy data not available")

    except Exception as e:
        st.error(f"Error loading accuracy data: {str(e)}")

    # Signal quality analysis
    st.subheader("🏆 Signal Quality Analysis")

    signals_data = st.session_state['current_signals']
    signal_summary = signals_data.get("signal_summary", {})

    # Signal frequency analysis
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Signal Frequency")

        buy_signals = signal_summary.get("buy_signals", 0)
        sell_signals = signal_summary.get("sell_signals", 0)
        total_signals = buy_signals + sell_signals
        total_periods = signal_summary.get("total_periods", 1)

        if total_periods > 0:
            signal_rate = (total_signals / total_periods) * 100

            st.metric("📈 Signal Rate", f"{signal_rate:.1f}%")
            st.metric("⚖️ Buy/Sell Ratio",
                      f"{buy_signals}/{sell_signals}" if sell_signals > 0 else f"{buy_signals}/0")

    with col2:
        st.markdown("### ⏱️ Timing Analysis")

        avg_holding = signal_summary.get("avg_holding_period", 0)
        activity_ratio = signal_summary.get("activity_ratio", 0)

        st.metric("📅 Avg Holding Period", f"{avg_holding:.1f} days")
        st.metric("🏃‍♂️ Activity Level", f"{activity_ratio:.1%}")

    # Performance recommendations
    st.subheader("💡 Performance Recommendations")

    # Analyze signal characteristics and provide recommendations
    recommendations = []

    if signal_summary.get("signal_frequency", 0) > 0.3:
        recommendations.append(
            "⚠️ High signal frequency detected. Consider increasing threshold sensitivity.")

    if signal_summary.get("activity_ratio", 0) < 0.1:
        recommendations.append(
            "📈 Low activity ratio. Consider decreasing threshold sensitivity for more signals.")

    avg_holding = signal_summary.get("avg_holding_period", 0)
    if avg_holding < 2:
        recommendations.append(
            "⚡ Very short holding periods. Consider increasing smoothing window.")
    elif avg_holding > 20:
        recommendations.append(
            "🐌 Long holding periods. Consider decreasing smoothing window for more responsiveness.")

    if not recommendations:
        recommendations.append("✅ Signal parameters appear well-calibrated.")

    for rec in recommendations:
        st.info(rec)

    # Export signals
    st.subheader("📤 Export Signals")

    if st.button("📥 Download Signal Data"):
        try:
            signals_info = signals_data.get("signals", {})

            # Create DataFrame
            signal_df = pd.DataFrame({
                'Date': signals_info.get("dates", []),
                'Signal': signals_info.get("signals", []),
                'Signal_Name': [
                    'SELL' if s == -1 else 'HOLD' if s == 0 else 'BUY'
                    for s in signals_info.get("signals", [])
                ]
            })

            # Convert to CSV
            csv = signal_df.to_csv(index=False)

            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"sp500_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"Error preparing download: {str(e)}")