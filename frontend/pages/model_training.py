"""
Model Training page for SP500 Signal Generator
"""
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import settings, MODEL_CONFIGS

# API Configuration
API_BASE_URL = f"http://{settings.API_HOST}:{settings.API_PORT}"

def show_page():
    """Main function to display the model training page"""
    st.title("🤖 Model Training")
    st.markdown("Train and manage various time series forecasting models for S&P 500 signal generation.")

    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 Train Models", "📊 Model Status", "🏆 Model Comparison", "⚙️ Model Management"])

    with tab1:
        show_model_training_section()

    with tab2:
        show_model_status_section()

    with tab3:
        show_model_comparison_section()

    with tab4:
        show_model_management_section()

def show_model_training_section():
    """Display model training interface"""
    st.header("🚀 Train New Models")

    # Check if data is available
    try:
        response = requests.get(f"{API_BASE_URL}/data/summary")
        if response.status_code != 200:
            st.error("⚠️ No data available. Please load data first in the Data Overview section.")
            return
    except:
        st.error("⚠️ Cannot connect to data service. Please ensure the backend is running.")
        return

    # Model selection and configuration
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Model Configuration")

        # Model type selection
        model_type = st.selectbox(
            "Select Model Type",
            list(MODEL_CONFIGS.keys()),
            format_func=lambda x: f"{MODEL_CONFIGS[x]['name']} - {MODEL_CONFIGS[x]['description']}"
        )

        # Display model information
        model_info = MODEL_CONFIGS[model_type]
        st.info(f"""
        **{model_info['name']}**  
        {model_info['description']}
        
        **Key Parameters:** {', '.join(model_info['hyperparameters'])}
        """)

        # Target variable selection
        target_options = [
            "returns_1d", "returns_5d", "returns_10d", "returns_20d",
            "close", "volatility", "direction_target_1d"
        ]

        target_column = st.selectbox(
            "Target Variable",
            target_options,
            help="Variable to predict"
        )

        # Training parameters
        st.subheader("🎛️ Training Parameters")

        col_a, col_b = st.columns(2)

        with col_a:
            hyperparameter_tune = st.checkbox(
                "Enable Hyperparameter Tuning",
                value=True,
                help="Automatically optimize model parameters"
            )

            train_size = st.slider(
                "Training Data Ratio",
                min_value=0.6,
                max_value=0.9,
                value=0.8,
                step=0.05,
                help="Proportion of data used for training"
            )

        with col_b:
            if model_type in ["lstm", "gru", "cnn_lstm", "attention_lstm"]:
                epochs = st.number_input("Training Epochs", min_value=10, max_value=200, value=100)
                batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)

                custom_params = {
                    "epochs": epochs,
                    "batch_size": batch_size
                }
            elif model_type == "arima":
                max_p = st.number_input("Max AR Order (p)", min_value=1, max_value=10, value=5)
                max_d = st.number_input("Max Differencing (d)", min_value=0, max_value=3, value=2)
                max_q = st.number_input("Max MA Order (q)", min_value=1, max_value=10, value=5)

                custom_params = {
                    "max_p": max_p,
                    "max_d": max_d,
                    "max_q": max_q
                }
            elif model_type == "garch":
                garch_type = st.selectbox("GARCH Type", ["GARCH", "EGARCH", "GJR-GARCH"])
                max_p = st.number_input("Max GARCH Order (p)", min_value=1, max_value=5, value=3)
                max_q = st.number_input("Max ARCH Order (q)", min_value=1, max_value=5, value=3)

                custom_params = {
                    "garch_type": garch_type,
                    "max_p": max_p,
                    "max_q": max_q
                }
            else:
                custom_params = {}

    with col2:
        st.subheader("🎯 Training Actions")

        # Training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            with st.spinner("Initializing model training..."):
                try:
                    # Prepare training request
                    training_request = {
                        "model_type": model_type,
                        "target_column": target_column,
                        "hyperparameter_tune": hyperparameter_tune,
                        "train_size": train_size,
                        "custom_params": custom_params if custom_params else None
                    }

                    # Start training
                    response = requests.post(
                        f"{API_BASE_URL}/models/train/{model_type}",
                        json=training_request
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            task_id = result["task_id"]
                            st.success(f"✅ Training started! Task ID: {task_id}")

                            # Store task ID in session state for monitoring
                            if "training_tasks" not in st.session_state:
                                st.session_state.training_tasks = []
                            st.session_state.training_tasks.append(task_id)

                            # Auto-refresh to show progress
                            st.rerun()
                        else:
                            st.error("Failed to start training")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except Exception as e:
                    st.error(f"Error starting training: {str(e)}")

        # Ensemble training
        st.subheader("🎭 Ensemble Training")

        ensemble_models = st.multiselect(
            "Select Models for Ensemble",
            list(MODEL_CONFIGS.keys()),
            default=["arima", "lstm", "prophet"],
            format_func=lambda x: MODEL_CONFIGS[x]['name']
        )

        if st.button("🎭 Train Ensemble", use_container_width=True):
            if len(ensemble_models) < 2:
                st.warning("Please select at least 2 models for ensemble training")
            else:
                with st.spinner("Starting ensemble training..."):
                    try:
                        ensemble_request = {
                            "model_types": ensemble_models,
                            "target_column": target_column
                        }

                        response = requests.post(
                            f"{API_BASE_URL}/models/ensemble/train",
                            json=ensemble_request
                        )

                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                task_id = result["task_id"]
                                st.success(f"✅ Ensemble training started! Task ID: {task_id}")

                                if "training_tasks" not in st.session_state:
                                    st.session_state.training_tasks = []
                                st.session_state.training_tasks.append(task_id)

                                st.rerun()
                            else:
                                st.error("Failed to start ensemble training")
                        else:
                            st.error(f"API Error: {response.status_code}")

                    except Exception as e:
                        st.error(f"Error starting ensemble training: {str(e)}")

        # Quick training presets
        st.subheader("⚡ Quick Training Presets")

        if st.button("🏃‍♂️ Quick ARIMA", use_container_width=True):
            quick_train_model("arima", target_column, hyperparameter_tune=False)

        if st.button("🧠 Quick LSTM", use_container_width=True):
            quick_train_model("lstm", target_column, hyperparameter_tune=False)

        if st.button("🔮 Quick Prophet", use_container_width=True):
            quick_train_model("prophet", target_column, hyperparameter_tune=False)

def quick_train_model(model_type, target_column, hyperparameter_tune=False):
    """Quick training function with default parameters"""
    try:
        training_request = {
            "model_type": model_type,
            "target_column": target_column,
            "hyperparameter_tune": hyperparameter_tune,
            "train_size": 0.8
        }

        response = requests.post(
            f"{API_BASE_URL}/models/train/{model_type}",
            json=training_request
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                task_id = result["task_id"]
                st.success(f"✅ Quick {model_type.upper()} training started!")

                if "training_tasks" not in st.session_state:
                    st.session_state.training_tasks = []
                st.session_state.training_tasks.append(task_id)
            else:
                st.error(f"Failed to start {model_type} training")
        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.error(f"Error starting quick training: {str(e)}")

def show_model_status_section():
    """Display model training status and progress"""
    st.header("📊 Model Training Status")

    # Check for active training tasks
    if "training_tasks" in st.session_state and st.session_state.training_tasks:
        st.subheader("🔄 Active Training Tasks")

        for task_id in st.session_state.training_tasks[:]:
            try:
                response = requests.get(f"{API_BASE_URL}/models/train/status/{task_id}")
                if response.status_code == 200:
                    status_data = response.json()

                    with st.container():
                        col1, col2, col3 = st.columns([2, 1, 1])

                        with col1:
                            st.write(f"**Task:** {task_id}")
                            st.write(f"**Model:** {status_data.get('model_type', 'Unknown')}")

                        with col2:
                            status = status_data.get("status", "unknown")
                            if status == "completed":
                                st.success("✅ Completed")
                                # Remove from active tasks
                                st.session_state.training_tasks.remove(task_id)
                            elif status == "failed":
                                st.error("❌ Failed")
                                st.session_state.training_tasks.remove(task_id)
                            elif status in ["started", "training"]:
                                st.info("🔄 Training...")
                            else:
                                st.warning(f"❓ {status}")

                        with col3:
                            progress = status_data.get("progress", 0)
                            st.progress(progress / 100)
                            st.write(f"{progress}%")

                        # Show additional info for completed/failed tasks
                        if status == "completed":
                            st.success(f"Model training completed successfully!")
                            if "results" in status_data:
                                results = status_data["results"]
                                if "directional_accuracy" in results:
                                    st.metric("🎯 Directional Accuracy", f"{results['directional_accuracy']:.2%}")

                        elif status == "failed":
                            error = status_data.get("error", "Unknown error")
                            st.error(f"Training failed: {error}")

                        st.markdown("---")

            except Exception as e:
                st.warning(f"Could not get status for task {task_id}: {str(e)}")

        # Auto-refresh for active tasks
        if any(True for task_id in st.session_state.training_tasks):
            time.sleep(5)
            st.rerun()

    else:
        st.info("No active training tasks. Start training in the 'Train Models' tab.")

    # Show training history
    st.subheader("📈 Training History")

    try:
        response = requests.get(f"{API_BASE_URL}/models/trained")
        if response.status_code == 200:
            models_data = response.json()

            if models_data.get("success"):
                trained_models = models_data.get("trained_models", {})

                if trained_models:
                    # Create training history DataFrame
                    history_data = []
                    for model_key, model_info in trained_models.items():
                        history_data.append({
                            "Model Key": model_key,
                            "Type": model_info.get("model_type", "Unknown"),
                            "Target": model_info.get("target_column", "Unknown"),
                            "Created": model_info.get("created_at", "Unknown"),
                            "Data Shape": f"{model_info.get('train_data_shape', [0, 0])[0]} x {model_info.get('train_data_shape', [0, 0])[1]}",
                            "Performance": format_performance_metrics(model_info.get("performance", {}))
                        })

                    history_df = pd.DataFrame(history_data)
                    st.dataframe(history_df, use_container_width=True)

                    # Training timeline chart
                    if len(history_data) > 1:
                        st.subheader("📊 Training Timeline")

                        # Parse dates and create timeline
                        dates = []
                        model_types = []

                        for item in history_data:
                            try:
                                date = pd.to_datetime(item["Created"])
                                dates.append(date)
                                model_types.append(item["Type"])
                            except:
                                continue

                        if dates:
                            fig = px.scatter(
                                x=dates,
                                y=model_types,
                                title="Model Training Timeline",
                                labels={"x": "Date", "y": "Model Type"}
                            )

                            fig.update_traces(marker_size=10)
                            fig.update_layout(height=400)

                            st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No models have been trained yet.")
            else:
                st.error("Failed to load training history")
        else:
            st.warning("Could not load training history")

    except Exception as e:
        st.error(f"Error loading training history: {str(e)}")

def format_performance_metrics(performance):
    """Format performance metrics for display"""
    if not performance:
        return "N/A"

    metrics = []
    if "directional_accuracy" in performance:
        metrics.append(f"Acc: {performance['directional_accuracy']:.2%}")
    if "mse" in performance:
        metrics.append(f"MSE: {performance['mse']:.4f}")
    if "sharpe_ratio" in performance:
        metrics.append(f"Sharpe: {performance['sharpe_ratio']:.2f}")

    return " | ".join(metrics) if metrics else "N/A"

def show_model_comparison_section():
    """Display model comparison interface"""
    st.header("🏆 Model Comparison")

    # Get list of trained models
    try:
        response = requests.get(f"{API_BASE_URL}/models/trained")
        if response.status_code == 200:
            models_data = response.json()

            if models_data.get("success"):
                trained_models = models_data.get("trained_models", {})

                if len(trained_models) < 2:
                    st.warning("Need at least 2 trained models for comparison.")
                    return

                # Model selection for comparison
                model_keys = list(trained_models.keys())

                selected_models = st.multiselect(
                    "Select Models to Compare",
                    model_keys,
                    default=model_keys[:min(3, len(model_keys))],
                    format_func=lambda x: f"{trained_models[x]['model_type']} - {x}"
                )

                if len(selected_models) >= 2 and st.button("🔍 Compare Models"):
                    compare_models(selected_models)

                # Show comparison matrix if models are selected
                if len(selected_models) >= 2:
                    show_comparison_matrix(selected_models, trained_models)

            else:
                st.error("Failed to load trained models")
        else:
            st.warning("No trained models available")

    except Exception as e:
        st.error(f"Error loading models for comparison: {str(e)}")

def compare_models(selected_models):
    """Compare selected models"""
    try:
        model_keys_str = ",".join(selected_models)
        response = requests.get(f"{API_BASE_URL}/models/compare?model_keys={model_keys_str}")

        if response.status_code == 200:
            comparison_data = response.json()

            if comparison_data.get("success"):
                comparison = comparison_data.get("comparison", {})
                best_models = comparison_data.get("best_models", {})

                # Display comparison results
                st.subheader("📊 Comparison Results")

                # Best models summary
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🏆 Best Performers")
                    for metric, best_info in best_models.items():
                        model_name = best_info["model"]
                        value = best_info["value"]

                        if metric in ["mse", "mae", "rmse"]:
                            st.metric(f"Best {metric.upper()}", model_name, f"{value:.4f}")
                        elif "accuracy" in metric or "sharpe" in metric:
                            st.metric(f"Best {metric.replace('_', ' ').title()}", model_name, f"{value:.3f}")

                with col2:
                    st.markdown("### 📈 Performance Metrics")

                    # Create comparison DataFrame
                    comparison_df = []
                    for model_key, model_data in comparison.items():
                        row = {"Model": model_key, "Type": model_data["model_type"]}
                        row.update(model_data["metrics"])
                        comparison_df.append(row)

                    comparison_df = pd.DataFrame(comparison_df)
                    st.dataframe(comparison_df, use_container_width=True)

                # Visualization
                st.subheader("📊 Visual Comparison")

                if len(comparison_df) > 0:
                    # Performance radar chart
                    metrics_to_plot = ["directional_accuracy", "sharpe_ratio", "mse"]
                    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]

                    if available_metrics:
                        fig = go.Figure()

                        for _, row in comparison_df.iterrows():
                            values = []
                            for metric in available_metrics:
                                val = row[metric]
                                # Normalize MSE (lower is better, so invert)
                                if metric == "mse":
                                    val = 1 / (1 + val) if val > 0 else 0
                                values.append(val)

                            fig.add_trace(go.Scatterpolar(
                                r=values,
                                theta=available_metrics,
                                fill='toself',
                                name=row["Model"]
                            ))

                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 1]
                                )),
                            showlegend=True,
                            title="Model Performance Comparison"
                        )

                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("Model comparison failed")
        else:
            st.error(f"API Error: {response.status_code}")

    except Exception as e:
        st.error(f"Error comparing models: {str(e)}")

def show_comparison_matrix(selected_models, trained_models):
    """Show comparison matrix for selected models"""
    st.subheader("📋 Model Details Matrix")

    matrix_data = []
    for model_key in selected_models:
        model_info = trained_models[model_key]

        matrix_data.append({
            "Model Key": model_key,
            "Type": model_info.get("model_type", "Unknown"),
            "Target": model_info.get("target_column", "Unknown"),
            "Created": model_info.get("created_at", "Unknown")[:19],  # Remove microseconds
            "Train Size": f"{model_info.get('train_data_shape', [0, 0])[0]:,}",
            "Features": f"{model_info.get('train_data_shape', [0, 0])[1]:,}"
        })

    matrix_df = pd.DataFrame(matrix_data)
    st.dataframe(matrix_df, use_container_width=True)

def show_model_management_section():
    """Display model management interface"""
    st.header("⚙️ Model Management")

    try:
        response = requests.get(f"{API_BASE_URL}/models/trained")
        if response.status_code == 200:
            models_data = response.json()

            if models_data.get("success"):
                trained_models = models_data.get("trained_models", {})

                if trained_models:
                    # Model selection
                    selected_model = st.selectbox(
                        "Select Model to Manage",
                        list(trained_models.keys()),
                        format_func=lambda x: f"{trained_models[x]['model_type']} - {x}"
                    )

                    if selected_model:
                        model_info = trained_models[selected_model]

                        col1, col2 = st.columns([2, 1])

                        with col1:
                            st.subheader("📋 Model Details")

                            # Display detailed model information
                            details_dict = {
                                "Model Key": selected_model,
                                "Model Type": model_info.get("model_type", "Unknown"),
                                "Target Variable": model_info.get("target_column", "Unknown"),
                                "Created At": model_info.get("created_at", "Unknown"),
                                "Training Data Shape": f"{model_info.get('train_data_shape', [0, 0])[0]} x {model_info.get('train_data_shape', [0, 0])[1]}"
                            }

                            for key, value in details_dict.items():
                                st.write(f"**{key}:** {value}")

                            # Performance metrics
                            st.subheader("📊 Performance Metrics")
                            performance = model_info.get("performance", {})

                            if performance:
                                metrics_col1, metrics_col2 = st.columns(2)

                                with metrics_col1:
                                    for key, value in list(performance.items())[:len(performance)//2]:
                                        if isinstance(value, (int, float)):
                                            if "accuracy" in key.lower():
                                                st.metric(key.replace("_", " ").title(), f"{value:.2%}")
                                            elif key.lower() in ["mse", "mae", "rmse"]:
                                                st.metric(key.upper(), f"{value:.4f}")
                                            else:
                                                st.metric(key.replace("_", " ").title(), f"{value:.3f}")

                                with metrics_col2:
                                    for key, value in list(performance.items())[len(performance)//2:]:
                                        if isinstance(value, (int, float)):
                                            if "accuracy" in key.lower():
                                                st.metric(key.replace("_", " ").title(), f"{value:.2%}")
                                            elif key.lower() in ["mse", "mae", "rmse"]:
                                                st.metric(key.upper(), f"{value:.4f}")
                                            else:
                                                st.metric(key.replace("_", " ").title(), f"{value:.3f}")

                            else:
                                st.info("No performance metrics available")

                        with col2:
                            st.subheader("🛠️ Actions")

                            # Get detailed model info
                            if st.button("📊 View Detailed Info", use_container_width=True):
                                try:
                                    detail_response = requests.get(f"{API_BASE_URL}/models/trained/{selected_model}/details")
                                    if detail_response.status_code == 200:
                                        detail_data = detail_response.json()

                                        with st.expander("📋 Detailed Model Information", expanded=True):
                                            st.json(detail_data)
                                    else:
                                        st.error("Failed to get detailed model info")
                                except Exception as e:
                                    st.error(f"Error getting model details: {str(e)}")

                            # Delete model
                            st.subheader("🗑️ Danger Zone")
                            if st.button("🗑️ Delete Model", type="secondary", use_container_width=True):
                                if st.checkbox("I confirm I want to delete this model"):
                                    try:
                                        delete_response = requests.delete(f"{API_BASE_URL}/models/trained/{selected_model}")
                                        if delete_response.status_code == 200:
                                            st.success("✅ Model deleted successfully!")
                                            st.rerun()
                                        else:
                                            st.error("Failed to delete model")
                                    except Exception as e:
                                        st.error(f"Error deleting model: {str(e)}")

                else:
                    st.info("No trained models available for management.")

            else:
                st.error("Failed to load trained models")

        else:
            st.warning("Could not load trained models")

    except Exception as e:
        st.error(f"Error in model management: {str(e)}")

    # Model export/import section
    st.subheader("📦 Model Export/Import")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📤 Export Models")
        st.info("Export functionality will save models to local storage for backup or sharing.")
        if st.button("📤 Export All Models"):
            st.info("Export functionality coming soon!")

    with col2:
        st.markdown("### 📥 Import Models")
        st.info("Import previously exported models or pre-trained models.")
        if st.button("📥 Import Models"):
            st.info("Import functionality coming soon!")