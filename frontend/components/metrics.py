"""
Reusable metrics components for SP500 Signal Generator
"""
import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import plotly.graph_objects as go
import plotly.express as px


def display_performance_metrics(metrics: Dict[str, float],
                                layout: str = "columns",
                                precision: int = 4) -> None:
    """
    Display performance metrics in a structured layout
    """
    if not metrics:
        st.warning("No metrics available to display")
        return

    if layout == "columns":
        # Determine number of columns based on metrics count
        n_metrics = len(metrics)
        n_cols = min(4, n_metrics)
        cols = st.columns(n_cols)

        for i, (metric_name, value) in enumerate(metrics.items()):
            with cols[i % n_cols]:
                formatted_value = format_metric_value(metric_name, value,
                                                      precision)
                metric_label = format_metric_label(metric_name)
                delta = get_metric_delta(metric_name, value)

                st.metric(
                    label=metric_label,
                    value=formatted_value,
                    delta=delta,
                    help=get_metric_description(metric_name)
                )

    elif layout == "table":
        # Display as DataFrame
        metrics_df = pd.DataFrame([
            {
                "Metric": format_metric_label(name),
                "Value": format_metric_value(name, value, precision),
                "Description": get_metric_description(name)
            }
            for name, value in metrics.items()
        ])

        st.dataframe(metrics_df, use_container_width=True)

    elif layout == "cards":
        # Display as info cards
        for metric_name, value in metrics.items():
            with st.container():
                col1, col2 = st.columns([1, 3])

                with col1:
                    formatted_value = format_metric_value(metric_name, value,
                                                          precision)
                    st.markdown(f"### {formatted_value}")

                with col2:
                    metric_label = format_metric_label(metric_name)
                    description = get_metric_description(metric_name)

                    st.markdown(f"**{metric_label}**")
                    st.markdown(f"*{description}*")

                st.markdown("---")


def format_metric_value(metric_name: str, value: float,
                        precision: int = 4) -> str:
    """
    Format metric value based on its type
    """
    if pd.isna(value):
        return "N/A"

    metric_lower = metric_name.lower()

    # Percentage metrics
    if any(keyword in metric_lower for keyword in
           ['accuracy', 'return', 'drawdown', 'rate', 'ratio']):
        if 'ratio' in metric_lower and metric_lower not in ['sharpe_ratio',
                                                            'calmar_ratio',
                                                            'sortino_ratio']:
            return f"{value:.{precision}f}"
        else:
            return f"{value:.2%}"

    # Currency metrics
    elif any(keyword in metric_lower for keyword in
             ['value', 'capital', 'pnl', 'profit', 'loss']):
        if abs(value) >= 1000000:
            return f"${value / 1000000:.1f}M"
        elif abs(value) >= 1000:
            return f"${value / 1000:.1f}K"
        else:
            return f"${value:.2f}"

    # Count metrics
    elif any(keyword in metric_lower for keyword in
             ['count', 'trades', 'signals']):
        return f"{int(value):,}"

    # Ratio and statistical metrics
    elif any(keyword in metric_lower for keyword in
             ['sharpe', 'calmar', 'sortino', 'correlation', 'alpha', 'beta']):
        return f"{value:.3f}"

    # Error metrics
    elif any(keyword in metric_lower for keyword in
             ['mse', 'mae', 'rmse', 'error']):
        return f"{value:.{precision}f}"

    # Default formatting
    else:
        if abs(value) >= 100:
            return f"{value:.2f}"
        else:
            return f"{value:.{precision}f}"


def format_metric_label(metric_name: str) -> str:
    """
    Format metric name for display
    """
    # Common abbreviation expansions
    expansions = {
        'mse': 'Mean Squared Error',
        'mae': 'Mean Absolute Error',
        'rmse': 'Root Mean Squared Error',
        'mape': 'Mean Absolute Percentage Error',
        'aic': 'Akaike Information Criterion',
        'bic': 'Bayesian Information Criterion',
        'var': 'Value at Risk',
        'cvar': 'Conditional Value at Risk',
        'pnl': 'Profit & Loss',
        'roi': 'Return on Investment',
        'cagr': 'Compound Annual Growth Rate',
        'ytd': 'Year to Date'
    }

    metric_lower = metric_name.lower()

    # Check for exact matches
    if metric_lower in expansions:
        return expansions[metric_lower]

    # Check for partial matches
    for abbrev, expansion in expansions.items():
        if abbrev in metric_lower:
            return metric_lower.replace(abbrev, expansion).title()

    # Default: replace underscores and title case
    return metric_name.replace('_', ' ').title()


def get_metric_description(metric_name: str) -> str:
    """
    Get description for metric
    """
    descriptions = {
        'total_return': 'Total portfolio return over the period',
        'annualized_return': 'Annualized portfolio return',
        'sharpe_ratio': 'Risk-adjusted return (return per unit of risk)',
        'sortino_ratio': 'Return per unit of downside risk',
        'calmar_ratio': 'Annual return divided by maximum drawdown',
        'max_drawdown': 'Largest peak-to-trough decline',
        'volatility': 'Annualized standard deviation of returns',
        'win_rate': 'Percentage of profitable trades',
        'profit_factor': 'Ratio of gross profit to gross loss',
        'directional_accuracy': 'Percentage of correct direction predictions',
        'mse': 'Average squared prediction errors',
        'mae': 'Average absolute prediction errors',
        'rmse': 'Square root of mean squared errors',
        'correlation': 'Correlation between predictions and actual values',
        'alpha': 'Excess return over benchmark',
        'beta': 'Sensitivity to market movements',
        'information_ratio': 'Active return per unit of tracking error',
        'tracking_error': 'Standard deviation of excess returns',
        'var_95': '5% Value at Risk (worst expected loss)',
        'trade_count': 'Total number of executed trades',
        'avg_holding_period': 'Average time between buy and sell',
        'activity_ratio': 'Percentage of time with active positions'
    }

    return descriptions.get(metric_name.lower(), 'Performance metric')


def get_metric_delta(metric_name: str, value: float) -> Optional[str]:
    """
    Get delta indicator for metric (good/bad performance)
    """
    if pd.isna(value):
        return None

    metric_lower = metric_name.lower()

    # Good when higher
    if any(keyword in metric_lower for keyword in
           ['return', 'accuracy', 'sharpe', 'calmar', 'sortino', 'win_rate',
            'profit_factor', 'correlation', 'alpha']):
        if value > 0:
            return "👍" if value > 0.1 else None
        else:
            return "👎" if value < -0.1 else None

    # Good when lower
    elif any(keyword in metric_lower for keyword in
             ['drawdown', 'error', 'volatility', 'tracking_error']):
        if abs(value) < 0.05:
            return "👍"
        elif abs(value) > 0.2:
            return "👎"

    return None


def create_metrics_summary_card(metrics: Dict[str, float],
                                title: str = "Performance Summary",
                                key_metrics: Optional[
                                    List[str]] = None) -> None:
    """
    Create a summary card with key performance metrics
    """
    if key_metrics is None:
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown',
                       'win_rate']

    # Filter available metrics
    available_metrics = {k: v for k, v in metrics.items() if k in key_metrics}

    with st.container():
        st.markdown(f"### {title}")

        if not available_metrics:
            st.warning("No key metrics available")
            return

        cols = st.columns(len(available_metrics))

        for i, (metric_name, value) in enumerate(available_metrics.items()):
            with cols[i]:
                formatted_value = format_metric_value(metric_name, value)
                metric_label = format_metric_label(metric_name)

                # Color code based on performance
                if any(keyword in metric_name.lower() for keyword in
                       ['return', 'accuracy', 'sharpe']):
                    color = "green" if value > 0 else "red"
                elif 'drawdown' in metric_name.lower():
                    color = "green" if value > -0.1 else "red"
                else:
                    color = "blue"

                st.markdown(f"""
                <div style="text-align: center; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                    <h3 style="color: {color}; margin: 0;">{formatted_value}</h3>
                    <p style="margin: 5px 0; font-size: 0.8em; color: #666;">{metric_label}</p>
                </div>
                """, unsafe_allow_html=True)


def display_risk_metrics(metrics: Dict[str, float]) -> None:
    """
    Display risk-specific metrics with appropriate formatting
    """
    st.subheader("🛡️ Risk Metrics")

    risk_metrics = {
        k: v for k, v in metrics.items()
        if any(keyword in k.lower() for keyword in
               ['drawdown', 'volatility', 'var', 'risk', 'beta', 'tracking'])
    }

    if not risk_metrics:
        st.info("No risk metrics available")
        return

    display_performance_metrics(risk_metrics, layout="columns")

    # Risk level assessment
    max_dd = risk_metrics.get('max_drawdown', 0)
    volatility = risk_metrics.get('volatility', 0)

    risk_level = assess_risk_level(max_dd, volatility)

    risk_colors = {
        'Low': 'green',
        'Medium': 'orange',
        'High': 'red'
    }

    st.markdown(f"""
    <div style="padding: 10px; border: 2px solid {risk_colors.get(risk_level, 'blue')}; 
                border-radius: 5px; margin: 10px 0;">
        <strong>Risk Assessment: <span style="color: {risk_colors.get(risk_level, 'blue')}">{risk_level}</span></strong>
    </div>
    """, unsafe_allow_html=True)


def assess_risk_level(max_drawdown: float, volatility: float) -> str:
    """
    Assess overall risk level based on key metrics
    """
    if abs(max_drawdown) < 0.1 and volatility < 0.15:
        return "Low"
    elif abs(max_drawdown) < 0.2 and volatility < 0.25:
        return "Medium"
    else:
        return "High"


def display_trade_metrics(metrics: Dict[str, float]) -> None:
    """
    Display trading-specific metrics
    """
    st.subheader("💼 Trading Metrics")

    trade_metrics = {
        k: v for k, v in metrics.items()
        if any(keyword in k.lower() for keyword in
               ['trade', 'win', 'profit', 'avg_win', 'avg_loss', 'holding'])
    }

    if not trade_metrics:
        st.info("No trading metrics available")
        return

    display_performance_metrics(trade_metrics, layout="columns")

    # Trading efficiency assessment
    win_rate = trade_metrics.get('win_rate', 0)
    profit_factor = trade_metrics.get('profit_factor', 0)

    efficiency_score = assess_trading_efficiency(win_rate, profit_factor)

    st.markdown(f"""
    <div style="padding: 10px; background-color: #f0f2f6; border-radius: 5px; margin: 10px 0;">
        <strong>Trading Efficiency: {efficiency_score}/10</strong>
        <div style="background-color: #ddd; border-radius: 10px; height: 20px; margin-top: 5px;">
            <div style="background-color: #4CAF50; height: 100%; border-radius: 10px; 
                        width: {efficiency_score * 10}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def assess_trading_efficiency(win_rate: float, profit_factor: float) -> int:
    """
    Assess trading efficiency on a scale of 1-10
    """
    score = 0

    # Win rate contribution (0-5 points)
    if win_rate > 0.6:
        score += 5
    elif win_rate > 0.5:
        score += 4
    elif win_rate > 0.4:
        score += 3
    elif win_rate > 0.3:
        score += 2
    else:
        score += 1

    # Profit factor contribution (0-5 points)
    if profit_factor > 2.0:
        score += 5
    elif profit_factor > 1.5:
        score += 4
    elif profit_factor > 1.2:
        score += 3
    elif profit_factor > 1.0:
        score += 2
    else:
        score += 1

    return min(score, 10)


def create_performance_comparison_table(
        models_metrics: Dict[str, Dict[str, float]],
        key_metrics: Optional[List[str]] = None) -> None:
    """
    Create comparison table for multiple models
    """
    if key_metrics is None:
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown',
                       'win_rate', 'volatility']

    # Prepare data for comparison
    comparison_data = []

    for model_name, metrics in models_metrics.items():
        row = {'Model': model_name}
        for metric in key_metrics:
            if metric in metrics:
                row[format_metric_label(metric)] = format_metric_value(metric,
                                                                       metrics[
                                                                           metric])
            else:
                row[format_metric_label(metric)] = "N/A"
        comparison_data.append(row)

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)

        # Highlight best performers
        st.markdown("#### 🏆 Best Performers")

        for metric in key_metrics:
            metric_label = format_metric_label(metric)
            if metric_label in comparison_df.columns:
                # Find best performer for this metric
                numeric_values = {}
                for _, row in comparison_df.iterrows():
                    model = row['Model']
                    value = models_metrics[model].get(metric, None)
                    if value is not None and not pd.isna(value):
                        numeric_values[model] = value

                if numeric_values:
                    if any(keyword in metric.lower() for keyword in
                           ['return', 'accuracy', 'sharpe', 'win']):
                        best_model = max(numeric_values,
                                         key=numeric_values.get)
                        best_value = numeric_values[best_model]
                    else:
                        best_model = min(numeric_values,
                                         key=lambda x: abs(numeric_values[x]))
                        best_value = numeric_values[best_model]

                    st.info(
                        f"**{metric_label}:** {best_model} ({format_metric_value(metric, best_value)})")


def display_model_diagnostics(diagnostics: Dict[str, Any]) -> None:
    """
    Display model diagnostic information
    """
    st.subheader("🔍 Model Diagnostics")

    if not diagnostics:
        st.info("No diagnostic information available")
        return

    # Statistical tests
    if 'ljung_box' in diagnostics:
        ljung_box = diagnostics['ljung_box']
        st.markdown("**Ljung-Box Test (Residual Autocorrelation):**")

        p_value = ljung_box.get('p_value', 0)
        is_white_noise = ljung_box.get('residuals_are_white_noise', False)

        color = "green" if is_white_noise else "red"
        status = "✅ Pass" if is_white_noise else "❌ Fail"

        st.markdown(
            f"- Status: <span style='color: {color}'>{status}</span> (p-value: {p_value:.4f})",
            unsafe_allow_html=True)

    if 'jarque_bera' in diagnostics:
        jarque_bera = diagnostics['jarque_bera']
        st.markdown("**Jarque-Bera Test (Normality):**")

        p_value = jarque_bera.get('p_value', 0)
        is_normal = jarque_bera.get('residuals_are_normal', False)

        color = "green" if is_normal else "red"
        status = "✅ Normal" if is_normal else "❌ Non-normal"

        st.markdown(
            f"- Status: <span style='color: {color}'>{status}</span> (p-value: {p_value:.4f})",
            unsafe_allow_html=True)

    if 'heteroskedasticity' in diagnostics:
        hetero = diagnostics['heteroskedasticity']
        st.markdown("**Heteroskedasticity Test:**")

        p_value = hetero.get('p_value', 0)
        is_homoskedastic = hetero.get('homoskedastic', False)

        color = "green" if is_homoskedastic else "red"
        status = "✅ Homoskedastic" if is_homoskedastic else "❌ Heteroskedastic"

        st.markdown(
            f"- Status: <span style='color: {color}'>{status}</span> (p-value: {p_value:.4f})",
            unsafe_allow_html=True)


def create_quick_performance_overview(metrics: Dict[str, float]) -> None:
    """
    Create a quick performance overview with key insights
    """
    st.markdown("### 📊 Quick Performance Overview")

    # Extract key metrics
    total_return = metrics.get('total_return', 0)
    sharpe_ratio = metrics.get('sharpe_ratio', 0)
    max_drawdown = metrics.get('max_drawdown', 0)
    win_rate = metrics.get('win_rate', 0)

    # Generate insights
    insights = []

    # Return assessment
    if total_return > 0.15:
        insights.append("🚀 **Strong Returns:** Excellent portfolio growth")
    elif total_return > 0.05:
        insights.append("📈 **Positive Returns:** Solid performance")
    elif total_return > 0:
        insights.append(
            "📊 **Modest Returns:** Positive but moderate performance")
    else:
        insights.append("📉 **Negative Returns:** Portfolio declined")

    # Risk assessment
    if abs(max_drawdown) < 0.1:
        insights.append("🛡️ **Low Risk:** Well-controlled downside risk")
    elif abs(max_drawdown) < 0.2:
        insights.append("⚖️ **Moderate Risk:** Balanced risk profile")
    else:
        insights.append("⚠️ **High Risk:** Significant drawdowns observed")

    # Risk-adjusted performance
    if sharpe_ratio > 1.5:
        insights.append(
            "⭐ **Excellent Risk-Adjusted Returns:** High Sharpe ratio")
    elif sharpe_ratio > 1.0:
        insights.append(
            "✅ **Good Risk-Adjusted Returns:** Decent Sharpe ratio")
    elif sharpe_ratio > 0.5:
        insights.append(
            "📊 **Fair Risk-Adjusted Returns:** Below average Sharpe")
    else:
        insights.append("❌ **Poor Risk-Adjusted Returns:** Low Sharpe ratio")

    # Display insights
    for insight in insights:
        st.markdown(insight)

    # Overall grade
    grade = calculate_overall_grade(total_return, sharpe_ratio, max_drawdown,
                                    win_rate)
    grade_colors = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red',
                    'F': 'darkred'}

    st.markdown(f"""
    <div style="text-align: center; padding: 20px; margin: 20px 0; 
                border: 3px solid {grade_colors.get(grade, 'gray')}; border-radius: 10px;">
        <h2 style="color: {grade_colors.get(grade, 'gray')}; margin: 0;">Overall Grade: {grade}</h2>
    </div>
    """, unsafe_allow_html=True)


def calculate_overall_grade(total_return: float, sharpe_ratio: float,
                            max_drawdown: float, win_rate: float) -> str:
    """
    Calculate overall performance grade
    """
    score = 0

    # Return score (30%)
    if total_return > 0.20:
        score += 30
    elif total_return > 0.15:
        score += 25
    elif total_return > 0.10:
        score += 20
    elif total_return > 0.05:
        score += 15
    elif total_return > 0:
        score += 10

    # Sharpe ratio score (30%)
    if sharpe_ratio > 2.0:
        score += 30
    elif sharpe_ratio > 1.5:
        score += 25
    elif sharpe_ratio > 1.0:
        score += 20
    elif sharpe_ratio > 0.5:
        score += 15
    elif sharpe_ratio > 0:
        score += 10

    # Max drawdown score (25%)
    if abs(max_drawdown) < 0.05:
        score += 25
    elif abs(max_drawdown) < 0.10:
        score += 20
    elif abs(max_drawdown) < 0.15:
        score += 15
    elif abs(max_drawdown) < 0.25:
        score += 10
    elif abs(max_drawdown) < 0.35:
        score += 5

    # Win rate score (15%)
    if win_rate > 0.7:
        score += 15
    elif win_rate > 0.6:
        score += 12
    elif win_rate > 0.5:
        score += 10
    elif win_rate > 0.4:
        score += 7
    elif win_rate > 0.3:
        score += 5

    # Convert to letter grade
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'