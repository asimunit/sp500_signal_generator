"""
Data Overview page for SP500 Signal Generator
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import settings
from components import charts, metrics

# API Configuration
API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"


def show_page():
    """Main function to display the data overview page"""
    st.title("📊 Data Overview")
    st.markdown(
        "Monitor and analyze S&P 500 data quality, statistics, and trends.")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Data Loading", "📊 Statistics", "🔍 Quality Analysis",
         "📉 Visualizations"])

    with tab1:
        show_data_loading_section()

    with tab2:
        show_statistics_section()

    with tab3:
        show_quality_analysis_section()

    with tab4:
        show_visualizations_section()


def show_data_loading_section():
    """Display data loading and fetching section"""
    st.header("📈 Data Loading & Management")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Fetch S&P 500 Data")

        # Data fetching parameters
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            symbol = st.selectbox(
                "Symbol",
                ["^GSPC", "^DJI", "^IXIC", "SPY", "QQQ"],
                index=0,
                help="Stock symbol to fetch data for"
            )

        with col_b:
            period = st.selectbox(
                "Period",
                ["1y", "2y", "5y", "10y", "max"],
                index=2,
                help="Historical data period"
            )

        with col_c:
            interval = st.selectbox(
                "Interval",
                ["1d", "1wk", "1mo"],
                index=0,
                help="Data frequency"
            )

        # Fetch data button
        if st.button("🔄 Fetch Data", type="primary"):
            with st.spinner("Fetching data from Yahoo Finance..."):
                try:
                    # Call API to fetch data
                    payload = {
                        "symbol": symbol,
                        "period": period,
                        "interval": interval
                    }

                    response = requests.post(f"{API_BASE_URL}/data/fetch",
                                             json=payload)

                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            st.success(
                                f"✅ Successfully fetched {result['data_shape'][0]} records!")

                            # Display fetch summary
                            st.info(f"""
                            **Data Summary:**
                            - Records: {result['data_shape'][0]:,}
                            - Features: {result['data_shape'][1]:,}  
                            - Date Range: {result['date_range']['start']} to {result['date_range']['end']}
                            - Quality Score: {result['quality_score']:.1f}%
                            """)

                            # Auto-refresh other sections
                            st.rerun()
                        else:
                            st.error("Failed to fetch data")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")

    with col2:
        st.subheader("Current Data Status")

        # Get current data summary
        try:
            response = requests.get(f"{API_BASE_URL}/data/summary")
            if response.status_code == 200:
                data_summary = response.json()

                # Display data metrics
                shape = data_summary.get("shape", [0, 0])
                quality_score = data_summary.get("quality_metrics", {}).get(
                    "quality_score", 0)

                st.metric("📊 Data Points", f"{shape[0]:,}")
                st.metric("🏷️ Features", f"{shape[1]:,}")
                st.metric("🎯 Quality Score", f"{quality_score:.1f}%")

                # Date range
                date_range = data_summary.get("quality_metrics", {}).get(
                    "date_range", {})
                if date_range:
                    st.metric("📅 Start Date",
                              str(date_range.get("start", "N/A"))[:10])
                    st.metric("📅 End Date",
                              str(date_range.get("end", "N/A"))[:10])
            else:
                st.warning("No data loaded")

        except Exception as e:
            st.error(f"Error loading data status: {str(e)}")


def show_statistics_section():
    """Display data statistics section"""
    st.header("📊 Data Statistics")

    try:
        # Get data statistics
        response = requests.get(f"{API_BASE_URL}/data/statistics")
        if response.status_code == 200:
            stats_data = response.json()

            if stats_data.get("success"):
                summary_stats = stats_data.get("summary_statistics", {})

                if summary_stats:
                    # Create statistics DataFrame
                    stats_df = pd.DataFrame(summary_stats).round(4)

                    # Display key price statistics
                    st.subheader("💰 Price Statistics")

                    price_cols = ["close", "open", "high", "low"]
                    available_price_cols = [col for col in price_cols if
                                            col in stats_df.columns]

                    if available_price_cols:
                        price_stats = stats_df[available_price_cols]
                        st.dataframe(price_stats, use_container_width=True)

                    # Display volume statistics
                    st.subheader("📈 Volume Statistics")
                    if "volume" in stats_df.columns:
                        volume_stats = stats_df[["volume"]]
                        st.dataframe(volume_stats, use_container_width=True)

                    # Display technical indicator statistics
                    st.subheader("🔧 Technical Indicators")
                    tech_indicators = [col for col in stats_df.columns if
                                       any(x in col for x in
                                           ["sma", "ema", "rsi", "macd",
                                            "volatility", "returns"])]

                    if tech_indicators:
                        # Limit to first 10 indicators for display
                        display_indicators = tech_indicators[:10]
                        tech_stats = stats_df[display_indicators]
                        st.dataframe(tech_stats, use_container_width=True)

                        if len(tech_indicators) > 10:
                            st.info(
                                f"Showing 10 of {len(tech_indicators)} technical indicators")

                    # Key metrics summary
                    st.subheader("📋 Key Metrics")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if "close" in summary_stats:
                            current_price = summary_stats["close"]["mean"]
                            st.metric("💲 Average Price",
                                      f"${current_price:.2f}")

                            price_std = summary_stats["close"]["std"]
                            st.metric("📊 Price Volatility",
                                      f"${price_std:.2f}")

                    with col2:
                        if "volume" in summary_stats:
                            avg_volume = summary_stats["volume"]["mean"]
                            st.metric("📈 Average Volume", f"{avg_volume:,.0f}")

                    with col3:
                        if "returns" in summary_stats:
                            avg_return = summary_stats["returns"][
                                             "mean"] * 252 * 100  # Annualized
                            st.metric("📈 Annualized Return",
                                      f"{avg_return:.2f}%")

                else:
                    st.warning("No statistics available")
            else:
                st.error("Failed to load statistics")
        else:
            st.warning("No data available for statistics")

    except Exception as e:
        st.error(f"Error loading statistics: {str(e)}")


def show_quality_analysis_section():
    """Display data quality analysis section"""
    st.header("🔍 Data Quality Analysis")

    try:
        # Get data summary with quality metrics
        response = requests.get(f"{API_BASE_URL}/data/summary")
        if response.status_code == 200:
            data_summary = response.json()
            quality_metrics = data_summary.get("quality_metrics", {})

            # Overall quality score
            col1, col2 = st.columns([1, 2])

            with col1:
                quality_score = quality_metrics.get("quality_score", 0)

                # Quality score gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=quality_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Data Quality Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Quality issues
                st.subheader("📝 Quality Issues")
                issues = quality_metrics.get("issues", [])

                if issues:
                    for issue in issues:
                        st.warning(f"⚠️ {issue}")
                else:
                    st.success("✅ No quality issues detected!")

                # Missing data analysis
                st.subheader("🕳️ Missing Data Analysis")
                missing_data = quality_metrics.get("missing_data", {})

                if missing_data:
                    missing_df = pd.DataFrame([
                        {
                            "Column": col,
                            "Missing Count": info["count"],
                            "Missing %": f"{info['percentage']:.2f}%"
                        }
                        for col, info in missing_data.items()
                        if info["count"] > 0
                    ])

                    if not missing_df.empty:
                        st.dataframe(missing_df, use_container_width=True)
                    else:
                        st.success("✅ No missing data found!")

            # Data completeness over time
            st.subheader("📈 Data Completeness Over Time")

            # Get price data for completeness analysis
            try:
                price_response = requests.get(
                    f"{API_BASE_URL}/data/price-data?columns=close")
                if price_response.status_code == 200:
                    price_data = price_response.json()

                    if price_data.get("success"):
                        data_dict = price_data["data"]
                        dates = pd.to_datetime(data_dict["dates"])

                        # Calculate daily data availability
                        date_range = pd.date_range(start=dates.min(),
                                                   end=dates.max(), freq='D')
                        availability = []

                        for date in date_range:
                            available = 1 if date in dates.values else 0
                            availability.append(available)

                        # Plot availability
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=date_range,
                            y=availability,
                            mode='markers',
                            name='Data Available',
                            marker=dict(color='green', size=3)
                        ))

                        fig.update_layout(
                            title="Data Availability Over Time",
                            xaxis_title="Date",
                            yaxis_title="Data Available (1=Yes, 0=No)",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Availability statistics
                        availability_pct = (sum(availability) / len(
                            availability)) * 100
                        st.metric("📊 Data Availability",
                                  f"{availability_pct:.1f}%")

            except Exception as e:
                st.warning(f"Could not analyze data completeness: {str(e)}")

        else:
            st.warning("No data quality information available")

    except Exception as e:
        st.error(f"Error loading quality analysis: {str(e)}")


def show_visualizations_section():
    """Display data visualizations section"""
    st.header("📉 Data Visualizations")

    # Visualization controls
    col1, col2, col3 = st.columns(3)

    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["Price Chart", "Volume Chart", "Technical Indicators",
             "Returns Distribution", "Correlation Matrix"]
        )

    with col2:
        date_range = st.selectbox(
            "Date Range",
            ["Last 30 Days", "Last 90 Days", "Last 1 Year", "All Data"]
        )

    with col3:
        chart_style = st.selectbox(
            "Chart Style",
            ["Default", "Dark", "White"]
        )

    # Generate visualizations based on selection
    if chart_type == "Price Chart":
        show_price_chart(date_range, chart_style)
    elif chart_type == "Volume Chart":
        show_volume_chart(date_range, chart_style)
    elif chart_type == "Technical Indicators":
        show_technical_indicators_chart(date_range, chart_style)
    elif chart_type == "Returns Distribution":
        show_returns_distribution(chart_style)
    elif chart_type == "Correlation Matrix":
        show_correlation_matrix(chart_style)


def show_price_chart(date_range, chart_style):
    """Display price chart"""
    try:
        # Get price data
        response = requests.get(
            f"{API_BASE_URL}/data/price-data?columns=close,high,low,open,volume")
        if response.status_code == 200:
            price_data = response.json()

            if price_data.get("success"):
                data_dict = price_data["data"]
                df = pd.DataFrame(data_dict["data"])
                df.index = pd.to_datetime(data_dict["dates"])

                # Filter by date range
                df = filter_by_date_range(df, date_range)

                # Create candlestick chart
                fig = go.Figure(data=go.Candlestick(
                    x=df.index,
                    open=df.get('open', df['close']),
                    high=df.get('high', df['close']),
                    low=df.get('low', df['close']),
                    close=df['close'],
                    name="S&P 500"
                ))

                fig.update_layout(
                    title=f"S&P 500 Price Chart ({date_range})",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    template=f"plotly_{chart_style.lower()}" if chart_style != "Default" else "plotly",
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

                # Price statistics for the period
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("💰 Current Price",
                              f"${df['close'].iloc[-1]:.2f}")

                with col2:
                    price_change = df['close'].iloc[-1] - df['close'].iloc[0]
                    price_change_pct = (price_change / df['close'].iloc[
                        0]) * 100
                    st.metric("📈 Period Change", f"${price_change:.2f}",
                              f"{price_change_pct:.2f}%")

                with col3:
                    st.metric("📊 High", f"${df['close'].max():.2f}")

                with col4:
                    st.metric("📉 Low", f"${df['close'].min():.2f}")

            else:
                st.error("Failed to load price data")
        else:
            st.warning("No price data available")

    except Exception as e:
        st.error(f"Error displaying price chart: {str(e)}")


def show_volume_chart(date_range, chart_style):
    """Display volume chart"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/data/price-data?columns=volume,close")
        if response.status_code == 200:
            price_data = response.json()

            if price_data.get("success"):
                data_dict = price_data["data"]
                df = pd.DataFrame(data_dict["data"])
                df.index = pd.to_datetime(data_dict["dates"])

                df = filter_by_date_range(df, date_range)

                # Create volume chart
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name='Volume',
                    marker_color='rgba(0,100,200,0.7)'
                ))

                fig.update_layout(
                    title=f"S&P 500 Trading Volume ({date_range})",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    template=f"plotly_{chart_style.lower()}" if chart_style != "Default" else "plotly",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Volume statistics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("📊 Average Volume",
                              f"{df['volume'].mean():,.0f}")

                with col2:
                    st.metric("📈 Max Volume", f"{df['volume'].max():,.0f}")

                with col3:
                    st.metric("📉 Min Volume", f"{df['volume'].min():,.0f}")

    except Exception as e:
        st.error(f"Error displaying volume chart: {str(e)}")


def show_technical_indicators_chart(date_range, chart_style):
    """Display technical indicators chart"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/data/technical-indicators?indicators=sma_20,sma_50,rsi,macd")
        if response.status_code == 200:
            tech_data = response.json()

            if tech_data.get("success"):
                data_dict = tech_data["data"]
                df = pd.DataFrame(data_dict["indicators"])
                df.index = pd.to_datetime(data_dict["dates"])

                df = filter_by_date_range(df, date_range)

                # Create subplots for different indicators
                from plotly.subplots import make_subplots

                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Moving Averages', 'RSI', 'MACD'),
                    vertical_spacing=0.1
                )

                # Moving averages
                if 'sma_20' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20'),
                        row=1, col=1)
                if 'sma_50' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50'),
                        row=1, col=1)

                # RSI
                if 'rsi' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['rsi'], name='RSI'), row=2,
                        col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red",
                                  row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green",
                                  row=2, col=1)

                # MACD
                if 'macd' in df.columns:
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['macd'], name='MACD'),
                        row=3, col=1)

                fig.update_layout(
                    title=f"Technical Indicators ({date_range})",
                    template=f"plotly_{chart_style.lower()}" if chart_style != "Default" else "plotly",
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error displaying technical indicators: {str(e)}")


def show_returns_distribution(chart_style):
    """Display returns distribution"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/data/statistics?column=returns")
        if response.status_code == 200:
            stats_data = response.json()

            # Also get the actual returns data
            price_response = requests.get(
                f"{API_BASE_URL}/data/price-data?columns=close")
            if price_response.status_code == 200:
                price_data = price_response.json()
                data_dict = price_data["data"]
                df = pd.DataFrame(data_dict["data"])

                # Calculate returns
                returns = df['close'].pct_change().dropna() * 100

                # Create distribution plot
                fig = go.Figure()

                fig.add_trace(go.Histogram(
                    x=returns,
                    nbinsx=50,
                    name='Returns Distribution',
                    opacity=0.7
                ))

                fig.update_layout(
                    title="Daily Returns Distribution",
                    xaxis_title="Daily Returns (%)",
                    yaxis_title="Frequency",
                    template=f"plotly_{chart_style.lower()}" if chart_style != "Default" else "plotly",
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

                # Returns statistics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📊 Mean Return", f"{returns.mean():.3f}%")

                with col2:
                    st.metric("📈 Std Deviation", f"{returns.std():.3f}%")

                with col3:
                    st.metric("📉 Min Return", f"{returns.min():.2f}%")

                with col4:
                    st.metric("📈 Max Return", f"{returns.max():.2f}%")

    except Exception as e:
        st.error(f"Error displaying returns distribution: {str(e)}")


def show_correlation_matrix(chart_style):
    """Display correlation matrix"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/data/correlation?method=pearson")
        if response.status_code == 200:
            corr_data = response.json()

            if corr_data.get("success"):
                corr_matrix = corr_data["correlation_matrix"]

                # Convert to DataFrame
                corr_df = pd.DataFrame(corr_matrix)

                # Limit to first 15 features for readability
                if len(corr_df) > 15:
                    corr_df = corr_df.iloc[:15, :15]

                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=corr_df.values,
                    x=corr_df.columns,
                    y=corr_df.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_df.round(2).values,
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ))

                fig.update_layout(
                    title="Feature Correlation Matrix",
                    template=f"plotly_{chart_style.lower()}" if chart_style != "Default" else "plotly",
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)

                # Correlation insights
                st.subheader("🔍 Correlation Insights")

                # Find highest correlations
                corr_values = corr_df.values
                np.fill_diagonal(corr_values, 0)  # Remove self-correlations

                max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_values)),
                                                corr_values.shape)
                max_corr = corr_values[max_corr_idx]

                st.info(f"""
                **Strongest Correlation:** {corr_df.index[max_corr_idx[0]]} ↔ {corr_df.columns[max_corr_idx[1]]}  
                **Correlation Value:** {max_corr:.3f}
                """)

    except Exception as e:
        st.error(f"Error displaying correlation matrix: {str(e)}")


def filter_by_date_range(df, date_range):
    """Filter DataFrame by date range selection"""
    if date_range == "Last 30 Days":
        return df.tail(30)
    elif date_range == "Last 90 Days":
        return df.tail(90)
    elif date_range == "Last 1 Year":
        return df.tail(252)  # Approximately 1 trading year
    else:  # All Data
        return df