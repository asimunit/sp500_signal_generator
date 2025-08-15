"""
Reusable chart components for SP500 Signal Generator
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import streamlit as st


def create_price_chart(data: pd.DataFrame,
                       price_column: str = 'close',
                       volume_column: Optional[str] = None,
                       signals: Optional[pd.Series] = None,
                       title: str = "Price Chart",
                       height: int = 500) -> go.Figure:
    """
    Create a comprehensive price chart with optional volume and signals
    """
    # Create subplots
    if volume_column and volume_column in data.columns:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(title, 'Volume'),
            row_width=[0.2, 0.7]
        )
        show_volume = True
    else:
        fig = go.Figure()
        show_volume = False

    # Price line/candlestick
    if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
        # Candlestick chart
        candlestick = go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        )

        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)

    else:
        # Line chart
        price_line = go.Scatter(
            x=data.index,
            y=data[price_column],
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        )

        if show_volume:
            fig.add_trace(price_line, row=1, col=1)
        else:
            fig.add_trace(price_line)

    # Add volume if available
    if show_volume:
        volume_bars = go.Bar(
            x=data.index,
            y=data[volume_column],
            name='Volume',
            marker_color='rgba(158,202,225,0.8)',
            yaxis='y2'
        )
        fig.add_trace(volume_bars, row=2, col=1)

    # Add trading signals if provided
    if signals is not None:
        # Buy signals
        buy_mask = signals == 1
        if buy_mask.any():
            buy_signals = go.Scatter(
                x=data.index[buy_mask],
                y=data[price_column][buy_mask],
                mode='markers',
                name='Buy Signal',
                marker=dict(
                    color='green',
                    size=12,
                    symbol='triangle-up',
                    line=dict(width=2, color='darkgreen')
                )
            )

            if show_volume:
                fig.add_trace(buy_signals, row=1, col=1)
            else:
                fig.add_trace(buy_signals)

        # Sell signals
        sell_mask = signals == -1
        if sell_mask.any():
            sell_signals = go.Scatter(
                x=data.index[sell_mask],
                y=data[price_column][sell_mask],
                mode='markers',
                name='Sell Signal',
                marker=dict(
                    color='red',
                    size=12,
                    symbol='triangle-down',
                    line=dict(width=2, color='darkred')
                )
            )

            if show_volume:
                fig.add_trace(sell_signals, row=1, col=1)
            else:
                fig.add_trace(sell_signals)

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=height,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def create_performance_chart(portfolio_values: pd.Series,
                             benchmark_values: Optional[pd.Series] = None,
                             title: str = "Portfolio Performance",
                             height: int = 400) -> go.Figure:
    """
    Create portfolio performance comparison chart
    """
    fig = go.Figure()

    # Portfolio performance
    fig.add_trace(go.Scatter(
        x=portfolio_values.index,
        y=portfolio_values,
        mode='lines',
        name='Strategy',
        line=dict(color='blue', width=2)
    ))

    # Benchmark if provided
    if benchmark_values is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_values.index,
            y=benchmark_values,
            mode='lines',
            name='Benchmark',
            line=dict(color='gray', width=2, dash='dash')
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=height,
        showlegend=True
    )

    return fig


def create_drawdown_chart(portfolio_values: pd.Series,
                          title: str = "Portfolio Drawdown",
                          height: int = 300) -> go.Figure:
    """
    Create drawdown chart
    """
    # Calculate drawdowns
    running_max = portfolio_values.expanding().max()
    drawdowns = (portfolio_values - running_max) / running_max * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdowns.index,
        y=drawdowns,
        mode='lines',
        name='Drawdown %',
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.3)',
        line=dict(color='red', width=1)
    ))

    fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=height,
        showlegend=False
    )

    return fig


def create_returns_distribution(returns: pd.Series,
                                title: str = "Returns Distribution",
                                height: int = 400) -> go.Figure:
    """
    Create returns distribution histogram with statistics
    """
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=returns * 100,  # Convert to percentage
        nbinsx=50,
        name='Returns',
        opacity=0.7,
        marker_color='skyblue'
    ))

    # Add vertical lines for statistics
    mean_return = returns.mean() * 100
    median_return = returns.median() * 100

    fig.add_vline(x=mean_return, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_return:.2f}%")
    fig.add_vline(x=median_return, line_dash="dot", line_color="green",
                  annotation_text=f"Median: {median_return:.2f}%")

    fig.update_layout(
        title=title,
        xaxis_title="Daily Returns (%)",
        yaxis_title="Frequency",
        height=height,
        showlegend=False
    )

    return fig


def create_correlation_heatmap(correlation_matrix: pd.DataFrame,
                               title: str = "Feature Correlation Matrix",
                               height: int = 500) -> go.Figure:
    """
    Create correlation heatmap
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale='RdBu',
        zmid=0,
        text=correlation_matrix.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Features",
        yaxis_title="Features"
    )

    return fig


def create_technical_indicators_chart(data: pd.DataFrame,
                                      price_column: str = 'close',
                                      indicators: List[str] = None,
                                      title: str = "Technical Indicators",
                                      height: int = 600) -> go.Figure:
    """
    Create technical indicators chart with multiple subplots
    """
    if indicators is None:
        indicators = ['sma_20', 'rsi', 'macd']

    # Filter available indicators
    available_indicators = [ind for ind in indicators if ind in data.columns]

    if not available_indicators:
        # Return simple price chart if no indicators available
        return create_price_chart(data, price_column, title=title,
                                  height=height)

    # Determine number of subplots
    n_subplots = 1 + len(available_indicators)  # Price + indicators

    fig = make_subplots(
        rows=n_subplots,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[title] + available_indicators
    )

    # Price chart
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[price_column],
        mode='lines',
        name='Price',
        line=dict(color='blue', width=2)
    ), row=1, col=1)

    # Add moving averages to price chart if available
    for ma in ['sma_20', 'sma_50', 'ema_20']:
        if ma in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[ma],
                mode='lines',
                name=ma.upper(),
                line=dict(width=1),
                opacity=0.7
            ), row=1, col=1)

    # Add other indicators to separate subplots
    row_idx = 2
    for indicator in available_indicators:
        if indicator not in ['sma_20', 'sma_50',
                             'ema_20']:  # Skip MAs (already on price chart)
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data[indicator],
                mode='lines',
                name=indicator.upper(),
                line=dict(color='orange', width=2)
            ), row=row_idx, col=1)

            # Add reference lines for specific indicators
            if indicator == 'rsi':
                fig.add_hline(y=70, line_dash="dash", line_color="red",
                              row=row_idx, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green",
                              row=row_idx, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray",
                              row=row_idx, col=1)
            elif indicator == 'macd':
                fig.add_hline(y=0, line_dash="solid", line_color="black",
                              row=row_idx, col=1)

            row_idx += 1

    fig.update_layout(
        height=height,
        showlegend=True,
        title_text=title
    )

    return fig


def create_signal_analysis_chart(signals: pd.Series,
                                 prices: pd.Series,
                                 title: str = "Signal Analysis",
                                 height: int = 500) -> go.Figure:
    """
    Create signal analysis chart showing signal distribution and timing
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=('Signals Over Time', 'Signal Distribution',
                        'Signal Timing', 'Cumulative Signals'),
        specs=[[{"secondary_y": True}, {"type": "pie"}],
               [{"colspan": 2}, None]],
        vertical_spacing=0.08
    )

    # 1. Signals over time with price
    fig.add_trace(go.Scatter(
        x=prices.index,
        y=prices,
        mode='lines',
        name='Price',
        line=dict(color='blue', width=1),
        yaxis='y2'
    ), row=1, col=1, secondary_y=True)

    fig.add_trace(go.Scatter(
        x=signals.index,
        y=signals,
        mode='markers',
        name='Signals',
        marker=dict(
            size=8,
            color=signals,
            colorscale=[[0, 'red'], [0.5, 'gray'], [1, 'green']],
            colorbar=dict(title="Signal", x=0.45)
        )
    ), row=1, col=1, secondary_y=False)

    # 2. Signal distribution pie chart
    signal_counts = signals.value_counts()
    signal_labels = {-1: 'Sell', 0: 'Hold', 1: 'Buy'}

    fig.add_trace(go.Pie(
        labels=[signal_labels.get(k, str(k)) for k in signal_counts.index],
        values=signal_counts.values,
        name="Distribution"
    ), row=1, col=2)

    # 3. Signal timing (gap between signals)
    signal_changes = signals[signals != 0]
    if len(signal_changes) > 1:
        signal_gaps = np.diff(signal_changes.index).astype(
            'timedelta64[D]').astype(int)

        fig.add_trace(go.Histogram(
            x=signal_gaps,
            nbinsx=20,
            name='Signal Gaps (Days)',
            marker_color='lightblue'
        ), row=2, col=1)

    fig.update_layout(
        title_text=title,
        height=height,
        showlegend=True
    )

    # Update y-axis labels
    fig.update_yaxes(title_text="Signal Value", row=1, col=1,
                     secondary_y=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Days Between Signals", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    return fig


def create_model_comparison_chart(model_results: Dict[str, Dict],
                                  metrics: List[str] = None,
                                  title: str = "Model Comparison",
                                  height: int = 500) -> go.Figure:
    """
    Create radar chart for model comparison
    """
    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Filter metrics that exist in the data
    available_metrics = []
    for metric in metrics:
        if any(metric in results for results in model_results.values()):
            available_metrics.append(metric)

    if not available_metrics:
        # Create bar chart instead
        return create_model_performance_bar_chart(model_results, title, height)

    fig = go.Figure()

    for model_name, results in model_results.items():
        values = []
        for metric in available_metrics:
            values.append(results.get(metric, 0))

        # Close the radar chart
        values.append(values[0])
        theta = available_metrics + [available_metrics[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=theta,
            fill='toself',
            name=model_name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title=title,
        height=height
    )

    return fig


def create_model_performance_bar_chart(model_results: Dict[str, Dict],
                                       title: str = "Model Performance",
                                       height: int = 400) -> go.Figure:
    """
    Create bar chart for model performance comparison
    """
    fig = go.Figure()

    model_names = list(model_results.keys())

    # Get common metrics
    all_metrics = set()
    for results in model_results.values():
        all_metrics.update(results.keys())

    for metric in sorted(all_metrics):
        values = [model_results[model].get(metric, 0) for model in model_names]

        fig.add_trace(go.Bar(
            name=metric,
            x=model_names,
            y=values
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Models",
        yaxis_title="Performance",
        height=height,
        barmode='group'
    )

    return fig


def create_forecast_chart(historical_data: pd.Series,
                          forecast_data: pd.Series,
                          confidence_intervals: Optional[pd.DataFrame] = None,
                          title: str = "Forecast Results",
                          height: int = 500) -> go.Figure:
    """
    Create forecast chart with historical data and predictions
    """
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data,
        mode='lines',
        name='Historical',
        line=dict(color='blue', width=2)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))

    # Confidence intervals
    if confidence_intervals is not None:
        fig.add_trace(go.Scatter(
            x=confidence_intervals.index,
            y=confidence_intervals['upper'],
            mode='lines',
            name='Upper CI',
            line=dict(color='rgba(255,0,0,0.2)'),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=confidence_intervals.index,
            y=confidence_intervals['lower'],
            mode='lines',
            name='Lower CI',
            line=dict(color='rgba(255,0,0,0.2)'),
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.1)',
            showlegend=True
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        height=height,
        showlegend=True
    )

    return fig


def create_risk_metrics_gauge(risk_metrics: Dict[str, float],
                              title: str = "Risk Metrics",
                              height: int = 300) -> go.Figure:
    """
    Create gauge chart for risk metrics
    """
    fig = make_subplots(
        rows=1,
        cols=len(risk_metrics),
        specs=[[{"type": "indicator"} for _ in risk_metrics]],
        subplot_titles=list(risk_metrics.keys())
    )

    for i, (metric, value) in enumerate(risk_metrics.items()):
        # Determine gauge parameters based on metric type
        if 'sharpe' in metric.lower():
            gauge_range = [0, 3]
            gauge_steps = [
                {'range': [0, 1], 'color': "lightgray"},
                {'range': [1, 2], 'color': "yellow"},
                {'range': [2, 3], 'color': "green"}
            ]
        elif 'drawdown' in metric.lower():
            gauge_range = [-50, 0]
            gauge_steps = [
                {'range': [-50, -20], 'color': "red"},
                {'range': [-20, -10], 'color': "yellow"},
                {'range': [-10, 0], 'color': "green"}
            ]
            value = value * 100  # Convert to percentage
        else:
            gauge_range = [0, 100]
            gauge_steps = [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ]

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [i / len(risk_metrics), (i + 1) / len(risk_metrics)],
                    'y': [0, 1]},
            gauge={
                'axis': {'range': gauge_range},
                'steps': gauge_steps,
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': gauge_range[1] * 0.9
                }
            }
        ), row=1, col=i + 1)

    fig.update_layout(
        title_text=title,
        height=height
    )

    return fig