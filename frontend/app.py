"""
Main Streamlit application for SP500 Signal Generator
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import page modules
from pages import data_overview, model_training, signal_generation, backtesting
from components import charts, metrics
from config.settings import settings

# Page configuration
st.set_page_config(
    page_title="SP500 Signal Generator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-alert {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .warning-alert {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .error-alert {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"


def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def display_header():
    """Display the main header"""
    st.markdown("""
    <div class="main-header">
        <h1>📈 SP500 Statistical Signal Generator</h1>
        <p>Advanced Time Series Forecasting & Trading Signal Generation</p>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display the sidebar navigation"""
    st.sidebar.markdown("## 🧭 Navigation")

    # Page selection
    pages = {
        "📊 Data Overview": "data_overview",
        "🤖 Model Training": "model_training",
        "🎯 Signal Generation": "signal_generation",
        "📈 Backtesting": "backtesting"
    }

    selected_page = st.sidebar.selectbox(
        "Select Page",
        list(pages.keys()),
        index=0
    )

    st.sidebar.markdown("---")

    # API Status
    st.sidebar.markdown("## 🔧 System Status")

    if check_api_health():
        st.sidebar.markdown('<div class="success-alert">✅ API Connected</div>',
                            unsafe_allow_html=True)

        # Get system status
        try:
            response = requests.get(f"{API_BASE_URL}/status")
            if response.status_code == 200:
                status_data = response.json()

                st.sidebar.markdown("### Service Status")
                services = status_data.get("services", {})

                for service_name, service_info in services.items():
                    status = service_info.get("status", "unknown")
                    if status == "active":
                        st.sidebar.markdown(
                            f"✅ {service_name.replace('_', ' ').title()}")
                    else:
                        st.sidebar.markdown(
                            f"❌ {service_name.replace('_', ' ').title()}")
        except:
            pass
    else:
        st.sidebar.markdown(
            '<div class="error-alert">❌ API Disconnected</div>',
            unsafe_allow_html=True)
        st.sidebar.markdown("Please ensure the backend is running:")
        st.sidebar.code("python run_backend.py")

    st.sidebar.markdown("---")

    # Quick Actions
    st.sidebar.markdown("## ⚡ Quick Actions")

    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()

    if st.sidebar.button("🧹 Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared!")

    # Settings
    st.sidebar.markdown("## ⚙️ Settings")

    st.sidebar.markdown("### Display Options")
    theme = st.sidebar.selectbox("Chart Theme",
                                 ["plotly", "plotly_white", "plotly_dark"])
    st.session_state['chart_theme'] = theme

    auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False)
    if auto_refresh:
        st.session_state['auto_refresh'] = True
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 10,
                                             300, 60)
        st.session_state['refresh_interval'] = refresh_interval

    # Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.markdown("""
    **SP500 Signal Generator** provides advanced statistical modeling 
    and backtesting capabilities for S&P 500 trading strategies.

    **Features:**
    - Multiple time series models
    - Dynamic signal generation  
    - Comprehensive backtesting
    - Performance analytics
    """)

    return pages[selected_page]


def display_notifications():
    """Display system notifications"""
    if not check_api_health():
        st.error("""
        🚨 **API Connection Error**

        The backend API is not accessible. Please ensure it's running:
        ```bash
        python run_backend.py
        ```
        """)
        st.stop()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_data_summary():
    """Fetch data summary from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/data/summary")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


@st.cache_data(ttl=300)
def fetch_trained_models():
    """Fetch trained models from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/models/trained")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def display_dashboard_overview():
    """Display dashboard overview on main page"""
    st.markdown("## 📊 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    # Data Status
    with col1:
        data_summary = fetch_data_summary()
        if data_summary:
            data_shape = data_summary.get("shape", [0, 0])
            quality_score = data_summary.get("quality_metrics", {}).get(
                "quality_score", 0)

            st.metric(
                "📈 Data Points",
                f"{data_shape[0]:,}",
                help="Number of data points loaded"
            )
            st.metric(
                "🎯 Data Quality",
                f"{quality_score:.1f}%",
                help="Overall data quality score"
            )
        else:
            st.metric("📈 Data Points", "No data",
                      help="Please load data first")

    # Models Status
    with col2:
        models_data = fetch_trained_models()
        if models_data and models_data.get("success"):
            model_count = models_data.get("count", 0)
            st.metric(
                "🤖 Trained Models",
                model_count,
                help="Number of trained models"
            )

            if model_count > 0:
                # Show best performing model
                models = models_data.get("trained_models", {})
                best_model = None
                best_accuracy = 0

                for model_key, model_info in models.items():
                    accuracy = model_info.get("performance", {}).get(
                        "directional_accuracy", 0)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model_key

                if best_model:
                    st.metric(
                        "🏆 Best Accuracy",
                        f"{best_accuracy:.1%}",
                        help=f"Best directional accuracy: {best_model}"
                    )
        else:
            st.metric("🤖 Trained Models", "0", help="No models trained yet")

    # Predictions Status
    with col3:
        try:
            response = requests.get(f"{API_BASE_URL}/predictions/cache")
            if response.status_code == 200:
                cache_data = response.json()
                prediction_count = cache_data.get("count", 0)
                st.metric(
                    "🔮 Cached Predictions",
                    prediction_count,
                    help="Number of cached predictions"
                )
            else:
                st.metric("🔮 Cached Predictions", "0")
        except:
            st.metric("🔮 Cached Predictions", "Error")

    # Backtest Status
    with col4:
        try:
            response = requests.get(f"{API_BASE_URL}/backtest/list")
            if response.status_code == 200:
                backtest_data = response.json()
                backtest_count = backtest_data.get("count", 0)
                st.metric(
                    "📊 Backtest Results",
                    backtest_count,
                    help="Number of completed backtests"
                )
            else:
                st.metric("📊 Backtest Results", "0")
        except:
            st.metric("📊 Backtest Results", "Error")


def display_quick_start():
    """Display quick start guide"""
    st.markdown("## 🚀 Quick Start Guide")

    with st.expander("📋 Getting Started", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 1️⃣ **Load Data**
            - Navigate to **Data Overview**
            - Click **Fetch SP500 Data**  
            - Review data quality metrics

            ### 2️⃣ **Train Models**
            - Go to **Model Training**
            - Select model types (ARIMA, LSTM, Prophet, etc.)
            - Configure hyperparameters
            - Start training process
            """)

        with col2:
            st.markdown("""
            ### 3️⃣ **Generate Signals**
            - Visit **Signal Generation**
            - Choose trained model
            - Configure signal parameters
            - Generate trading signals

            ### 4️⃣ **Backtest Strategy**
            - Access **Backtesting**
            - Run comprehensive backtests
            - Analyze performance metrics
            - Compare strategies
            """)


def main():
    """Main application function"""
    # Display header
    display_header()

    # Display notifications
    display_notifications()

    # Get selected page from sidebar
    selected_page = display_sidebar()

    # Auto-refresh logic
    if st.session_state.get('auto_refresh', False):
        time.sleep(st.session_state.get('refresh_interval', 60))
        st.rerun()

    # Route to selected page
    if selected_page == "data_overview":
        data_overview.show_page()
    elif selected_page == "model_training":
        model_training.show_page()
    elif selected_page == "signal_generation":
        signal_generation.show_page()
    elif selected_page == "backtesting":
        backtesting.show_page()
    else:
        # Default dashboard page
        display_dashboard_overview()
        st.markdown("---")
        display_quick_start()

        # Recent Activity
        st.markdown("## 📈 Recent Activity")

        # Show recent models, predictions, etc.
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🤖 Recent Models")
            models_data = fetch_trained_models()
            if models_data and models_data.get("success"):
                models = models_data.get("trained_models", {})
                if models:
                    recent_models = sorted(
                        models.items(),
                        key=lambda x: x[1].get("created_at", ""),
                        reverse=True
                    )[:5]

                    for model_key, model_info in recent_models:
                        model_type = model_info.get("model_type", "Unknown")
                        created_at = model_info.get("created_at", "")

                        st.markdown(f"**{model_type}** - {model_key}")
                        st.caption(f"Created: {created_at}")
                else:
                    st.info("No models trained yet")
            else:
                st.warning("Unable to load model data")

        with col2:
            st.markdown("### 📊 System Health")

            # Show system metrics
            try:
                response = requests.get(f"{API_BASE_URL}/status")
                if response.status_code == 200:
                    status_data = response.json()

                    memory_info = status_data.get("memory_info", {})
                    if "used_percent" in memory_info:
                        st.metric(
                            "Memory Usage",
                            f"{memory_info['used_percent']:.1f}%",
                            help="System memory utilization"
                        )

                    disk_info = status_data.get("disk_info", {})
                    if "used_percent" in disk_info:
                        st.metric(
                            "Disk Usage",
                            f"{disk_info['used_percent']:.1f}%",
                            help="Disk space utilization"
                        )
                else:
                    st.warning("Unable to load system status")
            except:
                st.error("Error fetching system status")


if __name__ == "__main__":
    main()