"""
Machine Learning Models for SP500 Forecasting (LSTM, GRU, CNN-LSTM)
"""
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, \
    MaxPooling1D, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, Optional, List
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class MLModels:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.history = {}
        self.feature_importance = {}

    def prepare_sequences(self, data: pd.DataFrame,
                          feature_columns: List[str],
                          target_column: str,
                          sequence_length: int = 60,
                          forecast_horizon: int = 1) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequences for time series modeling
        """
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(data[feature_columns])

        # Store scaler for later use
        self.scalers[target_column] = scaler

        X, y = [], []

        for i in range(sequence_length, len(data) - forecast_horizon + 1):
            # Input sequence
            X.append(scaled_features[i - sequence_length:i])
            # Target (future value)
            y.append(data[target_column].iloc[i + forecast_horizon - 1])

        X = np.array(X)
        y = np.array(y)

        # Split into train/validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(
            f"Prepared sequences - Train: {X_train.shape}, Val: {X_val.shape}")

        return (X_train, y_train), (X_val, y_val), scaler

    def build_lstm_model(self, input_shape: Tuple[int, int],
                         units: List[int] = [50, 50],
                         dropout_rate: float = 0.2,
                         learning_rate: float = 0.001) -> tf.keras.Model:
        """
        Build LSTM model architecture
        """
        model = Sequential()

        # First LSTM layer
        model.add(
            LSTM(units[0], return_sequences=True if len(units) > 1 else False,
                 input_shape=input_shape))
        model.add(Dropout(dropout_rate))

        # Additional LSTM layers
        for i, unit in enumerate(units[1:]):
            return_seq = i < len(units) - 2
            model.add(LSTM(unit, return_sequences=return_seq))
            model.add(Dropout(dropout_rate))

        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def build_gru_model(self, input_shape: Tuple[int, int],
                        units: List[int] = [50, 50],
                        dropout_rate: float = 0.2,
                        learning_rate: float = 0.001) -> tf.keras.Model:
        """
        Build GRU model architecture
        """
        model = Sequential()

        # First GRU layer
        model.add(
            GRU(units[0], return_sequences=True if len(units) > 1 else False,
                input_shape=input_shape))
        model.add(Dropout(dropout_rate))

        # Additional GRU layers
        for i, unit in enumerate(units[1:]):
            return_seq = i < len(units) - 2
            model.add(GRU(unit, return_sequences=return_seq))
            model.add(Dropout(dropout_rate))

        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def build_cnn_lstm_model(self, input_shape: Tuple[int, int],
                             conv_filters: int = 64,
                             kernel_size: int = 3,
                             lstm_units: int = 50,
                             dropout_rate: float = 0.2,
                             learning_rate: float = 0.001) -> tf.keras.Model:
        """
        Build CNN-LSTM hybrid model
        """
        model = Sequential()

        # CNN layers for feature extraction
        model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size,
                         activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_rate))

        # LSTM layers for sequence modeling
        model.add(LSTM(lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))

        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def build_attention_lstm_model(self, input_shape: Tuple[int, int],
                                   lstm_units: int = 50,
                                   dropout_rate: float = 0.2,
                                   learning_rate: float = 0.001) -> tf.keras.Model:
        """
        Build LSTM model with attention mechanism
        """
        from tensorflow.keras.layers import MultiHeadAttention, \
            LayerNormalization

        inputs = Input(shape=input_shape)

        # LSTM layers
        lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
        lstm_out = Dropout(dropout_rate)(lstm_out)

        # Attention layer
        attention_out = MultiHeadAttention(num_heads=4,
                                           key_dim=lstm_units // 4)(lstm_out,
                                                                    lstm_out)
        attention_out = LayerNormalization()(attention_out + lstm_out)

        # Global average pooling
        pooled = tf.keras.layers.GlobalAveragePooling1D()(attention_out)

        # Dense layers
        dense_out = Dense(25, activation='relu')(pooled)
        dense_out = Dropout(dropout_rate)(dense_out)
        output = Dense(1)(dense_out)

        model = Model(inputs=inputs, outputs=output)

        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              model_type: str = 'lstm') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using random search
        """
        logger.info(f"Starting hyperparameter tuning for {model_type}")

        # Define hyperparameter search space
        param_grid = {
            'units': [[32], [50], [64], [32, 32], [50, 50], [64, 64],
                      [100, 50]],
            'dropout_rate': [0.1, 0.2, 0.3],
            'learning_rate': [0.001, 0.01, 0.0001],
            'batch_size': [16, 32, 64]
        }

        best_score = float('inf')
        best_params = None
        results = []

        # Random search (simplified version)
        n_iterations = 10

        for iteration in range(n_iterations):
            # Sample random parameters
            params = {
                'units': np.random.choice(param_grid['units']),
                'dropout_rate': np.random.choice(param_grid['dropout_rate']),
                'learning_rate': np.random.choice(param_grid['learning_rate']),
                'batch_size': np.random.choice(param_grid['batch_size'])
            }

            try:
                # Build model
                input_shape = (X_train.shape[1], X_train.shape[2])

                if model_type == 'lstm':
                    model = self.build_lstm_model(input_shape,
                                                  **{k: v for k, v in
                                                     params.items() if
                                                     k != 'batch_size'})
                elif model_type == 'gru':
                    model = self.build_gru_model(input_shape,
                                                 **{k: v for k, v in
                                                    params.items() if
                                                    k != 'batch_size'})
                elif model_type == 'cnn_lstm':
                    model = self.build_cnn_lstm_model(input_shape,
                                                      **{k: v for k, v in
                                                         params.items() if
                                                         k != 'batch_size'})
                else:
                    continue

                # Train model
                early_stopping = EarlyStopping(patience=10,
                                               restore_best_weights=True)

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=params['batch_size'],
                    callbacks=[early_stopping],
                    verbose=0
                )

                # Evaluate
                val_loss = min(history.history['val_loss'])

                results.append({
                    'params': params,
                    'val_loss': val_loss,
                    'epochs_trained': len(history.history['loss'])
                })

                if val_loss < best_score:
                    best_score = val_loss
                    best_params = params

                logger.info(
                    f"Iteration {iteration + 1}: val_loss = {val_loss:.6f}")

            except Exception as e:
                logger.warning(
                    f"Hyperparameter iteration {iteration + 1} failed: {str(e)}")
                continue

        logger.info(
            f"Hyperparameter tuning completed. Best val_loss: {best_score:.6f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }

    def train_neural_network(self, data: pd.DataFrame,
                             feature_columns: List[str],
                             target_column: str,
                             model_type: str = 'lstm',
                             sequence_length: int = 60,
                             forecast_horizon: int = 1,
                             hyperparameter_tune: bool = True) -> Dict[
        str, Any]:
        """
        Train neural network model (LSTM, GRU, CNN-LSTM)
        """
        logger.info(f"Training {model_type} model for {target_column}")

        try:
            # Prepare data
            (X_train, y_train), (X_val,
                                 y_val), scaler = self.prepare_sequences(
                data, feature_columns, target_column, sequence_length,
                forecast_horizon
            )

            input_shape = (X_train.shape[1], X_train.shape[2])

            # Hyperparameter tuning
            if hyperparameter_tune:
                tuning_results = self.hyperparameter_tuning(X_train, y_train,
                                                            X_val, y_val,
                                                            model_type)
                best_params = tuning_results['best_params']
            else:
                best_params = {
                    'units': [50, 50],
                    'dropout_rate': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 32
                }

            # Build final model
            if model_type == 'lstm':
                model = self.build_lstm_model(input_shape, **{k: v for k, v in
                                                              best_params.items()
                                                              if
                                                              k != 'batch_size'})
            elif model_type == 'gru':
                model = self.build_gru_model(input_shape, **{k: v for k, v in
                                                             best_params.items()
                                                             if
                                                             k != 'batch_size'})
            elif model_type == 'cnn_lstm':
                model = self.build_cnn_lstm_model(input_shape,
                                                  **{k: v for k, v in
                                                     best_params.items() if
                                                     k != 'batch_size'})
            elif model_type == 'attention_lstm':
                model = self.build_attention_lstm_model(input_shape,
                                                        **{k: v for k, v in
                                                           best_params.items()
                                                           if
                                                           k != 'batch_size'})
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Callbacks
            callbacks = [
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
            ]

            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=best_params['batch_size'],
                callbacks=callbacks,
                verbose=1
            )

            # Evaluate model
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)

            # Calculate metrics
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            train_mae = mean_absolute_error(y_train, train_pred)
            val_mae = mean_absolute_error(y_val, val_pred)

            # Directional accuracy
            train_direction_acc = self._calculate_directional_accuracy(y_train,
                                                                       train_pred.flatten())
            val_direction_acc = self._calculate_directional_accuracy(y_val,
                                                                     val_pred.flatten())

            # Store model and results
            model_key = f"{model_type}_{target_column}"
            self.models[model_key] = model
            self.history[model_key] = history.history

            logger.info(
                f"{model_type} training completed. Val MSE: {val_mse:.6f}")

            return {
                'success': True,
                'model_type': model_type,
                'target_column': target_column,
                'best_params': best_params,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_direction_accuracy': train_direction_acc,
                'val_direction_accuracy': val_direction_acc,
                'epochs_trained': len(history.history['loss']),
                'model_key': model_key,
                'input_shape': input_shape
            }

        except Exception as e:
            logger.error(f"Error training {model_type}: {str(e)}")
            return {'success': False, 'error': str(e)}

    def train_ensemble_models(self, data: pd.DataFrame,
                              feature_columns: List[str],
                              target_column: str) -> Dict[str, Any]:
        """
        Train ensemble models (Random Forest, Gradient Boosting, SVR)
        """
        logger.info(f"Training ensemble models for {target_column}")

        try:
            # Prepare data (no sequences for traditional ML)
            X = data[feature_columns].fillna(0)
            y = data[target_column]

            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'svr': SVR(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale'
                )
            }

            results = {}

            for model_name, model in models.items():
                logger.info(f"Training {model_name}")

                # Train model
                if model_name == 'svr':
                    model.fit(X_train_scaled, y_train)
                    train_pred = model.predict(X_train_scaled)
                    test_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    train_pred = model.predict(X_train)
                    test_pred = model.predict(X_test)

                # Calculate metrics
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)

                # Directional accuracy
                train_direction_acc = self._calculate_directional_accuracy(
                    y_train.values, train_pred)
                test_direction_acc = self._calculate_directional_accuracy(
                    y_test.values, test_pred)

                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_importance = dict(
                        zip(feature_columns, model.feature_importances_))
                    self.feature_importance[
                        f"{model_name}_{target_column}"] = feature_importance
                else:
                    feature_importance = {}

                # Store model
                model_key = f"{model_name}_{target_column}"
                self.models[model_key] = model
                if model_name == 'svr':
                    self.scalers[model_key] = scaler

                results[model_name] = {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_direction_accuracy': train_direction_acc,
                    'test_direction_accuracy': test_direction_acc,
                    'feature_importance': feature_importance,
                    'model_key': model_key
                }

                logger.info(
                    f"{model_name} - Test MSE: {test_mse:.6f}, Test R²: {test_r2:.4f}")

            return {
                'success': True,
                'results': results,
                'target_column': target_column
            }

        except Exception as e:
            logger.error(f"Error training ensemble models: {str(e)}")
            return {'success': False, 'error': str(e)}

    def forecast(self, model_key: str, data: pd.DataFrame,
                 feature_columns: List[str], steps: int = 20) -> Dict[
        str, Any]:
        """
        Generate forecasts using trained model
        """
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found")

        model = self.models[model_key]

        try:
            if 'lstm' in model_key or 'gru' in model_key or 'cnn' in model_key or 'attention' in model_key:
                # Neural network models
                return self._forecast_neural_network(model, model_key, data,
                                                     feature_columns, steps)
            else:
                # Traditional ML models
                return self._forecast_traditional_ml(model, model_key, data,
                                                     feature_columns, steps)

        except Exception as e:
            logger.error(
                f"Error generating forecast with {model_key}: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _forecast_neural_network(self, model, model_key: str,
                                 data: pd.DataFrame,
                                 feature_columns: List[str], steps: int) -> \
    Dict[str, Any]:
        """
        Forecast using neural network models
        """
        # Use the scaler from training
        target_column = model_key.split('_', 1)[1]
        scaler = self.scalers.get(target_column)

        if scaler is None:
            raise ValueError(f"Scaler not found for {target_column}")

        # Get the sequence length from model input shape
        sequence_length = model.input_shape[1]

        # Prepare last sequence
        scaled_features = scaler.transform(
            data[feature_columns].tail(sequence_length))
        last_sequence = scaled_features.reshape(1, sequence_length,
                                                len(feature_columns))

        forecasts = []

        for step in range(steps):
            # Predict next value
            pred = model.predict(last_sequence, verbose=0)[0, 0]
            forecasts.append(pred)

            # Update sequence for next prediction (simplified approach)
            # In practice, you'd want to incorporate the prediction back into features
            last_sequence = np.roll(last_sequence, -1, axis=1)
            # For simplicity, just repeat the last feature vector
            last_sequence[0, -1, :] = last_sequence[0, -2, :]

        # Create forecast dates
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=steps, freq='D')

        forecast_df = pd.DataFrame({
            'forecast': forecasts
        }, index=forecast_dates)

        return {
            'success': True,
            'forecasts': forecasts,
            'forecast_dates': forecast_dates.tolist(),
            'forecast_df': forecast_df
        }

    def _forecast_traditional_ml(self, model, model_key: str,
                                 data: pd.DataFrame,
                                 feature_columns: List[str], steps: int) -> \
    Dict[str, Any]:
        """
        Forecast using traditional ML models
        """
        # For traditional ML, we'll use the last known features and predict iteratively
        last_features = data[feature_columns].iloc[-1:].fillna(0)

        # Scale if needed (for SVR)
        if model_key in self.scalers:
            scaler = self.scalers[model_key]
            last_features_scaled = scaler.transform(last_features)
            use_scaled = True
        else:
            last_features_scaled = last_features.values
            use_scaled = False

        forecasts = []

        for step in range(steps):
            if use_scaled:
                pred = model.predict(last_features_scaled)[0]
            else:
                pred = model.predict(last_features)[0]

            forecasts.append(pred)

            # Simple approach: keep features constant
            # In practice, you'd update relevant features based on the prediction

        # Create forecast dates
        last_date = data.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=steps, freq='D')

        forecast_df = pd.DataFrame({
            'forecast': forecasts
        }, index=forecast_dates)

        return {
            'success': True,
            'forecasts': forecasts,
            'forecast_dates': forecast_dates.tolist(),
            'forecast_df': forecast_df
        }

    def _calculate_directional_accuracy(self, actual: np.ndarray,
                                        predicted: np.ndarray) -> float:
        """
        Calculate directional accuracy (percentage of correct direction predictions)
        """
        if len(actual) < 2:
            return 0.0

        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))

        return np.mean(actual_direction == predicted_direction)

    def get_model_summary(self, model_key: str) -> Dict[str, Any]:
        """
        Get summary information about a trained model
        """
        if model_key not in self.models:
            return {}

        model = self.models[model_key]
        summary = {'model_key': model_key}

        if hasattr(model, 'summary'):
            # Neural network model
            summary['type'] = 'neural_network'
            summary['trainable_params'] = model.count_params()

            if model_key in self.history:
                history = self.history[model_key]
                summary['training_history'] = {
                    'final_train_loss': history['loss'][-1],
                    'final_val_loss': history['val_loss'][-1],
                    'epochs_trained': len(history['loss']),
                    'best_val_loss': min(history['val_loss'])
                }
        else:
            # Traditional ML model
            summary['type'] = 'traditional_ml'

            if hasattr(model, 'n_estimators'):
                summary['n_estimators'] = model.n_estimators

            if model_key in self.feature_importance:
                importance = self.feature_importance[model_key]
                top_features = sorted(importance.items(), key=lambda x: x[1],
                                      reverse=True)[:10]
                summary['top_features'] = top_features

        return summary