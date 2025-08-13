"""
Random Forest model implementation for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, List, Optional
import json
import os
import pickle
from benchmarking_pipeline.models.base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Random Forest model with given configuration.
        
        Args:
            config: Configuration dictionary for RandomForestRegressor parameters.
                    e.g., {'n_estimators': 100, 'max_depth': 10, 'random_state': 42, 'lookback_window': 10}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        self.lookback_window = self.config.get('lookback_window', 10)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        self._build_model()
        
    def _build_model(self):
        """
        Build the RandomForestRegressor model instance from the configuration.
        Supports both univariate and multivariate data dynamically.
        """
        # Get hyperparameters from config, excluding model-level parameters
        model_params = {}
        for key, value in self.config.items():
            if key not in ['lookback_window', 'forecast_horizon', 'target_cols']:
                model_params[key] = value

        # Filter out keys that are not valid for RandomForestRegressor
        valid_keys = set(RandomForestRegressor().get_params().keys())
        filtered_params = {k: v for k, v in model_params.items() if k in valid_keys}

        if 'random_state' not in filtered_params:
            filtered_params['random_state'] = 42

        if 'n_jobs' not in filtered_params:
            filtered_params['n_jobs'] = -1
            
        # Create single RandomForestRegressor that handles both univariate and multivariate
        self.model = RandomForestRegressor(**filtered_params)
        self.is_fitted = False
        
        # Set instance attributes for model-level parameters
        self.lookback_window = self.config.get('lookback_window', 10)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        
        # Handle target columns - support both univariate and multivariate
        if 'target_cols' in self.config:
            self.target_cols = self.config['target_cols']
        else:
            self.target_cols = ['y']  # Default for univariate
            
        # Ensure target_cols is always a list
        if isinstance(self.target_cols, str):
            self.target_cols = [self.target_cols]
            
        self.n_targets = len(self.target_cols)
        
        # Set target_col for compatibility with hyperparameter tuner
        self.target_col = self.target_cols[0] if len(self.target_cols) == 1 else 'y'

    def _create_features(self, y_series: np.ndarray, x_series: np.ndarray = None, timestamps: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series features for Random Forest with timestamp features.
        Supports both univariate and multivariate data.
        
        Args:
            y_series: Target time series (1D for univariate, 2D for multivariate)
            x_series: Exogenous variables (optional)
            timestamps: Timestamp features (optional)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets
        """
        # Handle multivariate data
        if y_series.ndim == 1:
            # Univariate case - reshape to 2D
            y_series = y_series.reshape(-1, 1)
            n_targets = 1
        else:
            # Multivariate case
            n_targets = y_series.shape[1]
            
        n_samples = len(y_series) - self.lookback_window - self.forecast_horizon + 1
        
        if n_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {self.lookback_window + self.forecast_horizon} samples.")
        
        features = []
        targets = []
        
        for i in range(n_samples):
            # Create lag features for all targets
            lag_features = []
            for target_idx in range(n_targets):
                target_lags = y_series[i:i + self.lookback_window, target_idx]
                lag_features.extend(target_lags)
            
            # Create rolling statistics for all targets
            rolling_stats = []
            for target_idx in range(n_targets):
                target_window = y_series[i:i + self.lookback_window, target_idx]
                rolling_mean = np.mean(target_window)
                rolling_std = np.std(target_window)
                rolling_min = np.min(target_window)
                rolling_max = np.max(target_window)
                trend = np.polyfit(range(self.lookback_window), target_window, 1)[0]
                rolling_stats.extend([rolling_mean, rolling_std, rolling_min, rolling_max, trend])
            
            # Combine all features
            sample_features = lag_features + rolling_stats
            
            # Add timestamp features if available
            if timestamps is not None and len(timestamps) >= i + self.lookback_window + self.forecast_horizon:
                # Add current timestamp and future timestamps as features
                current_timestamp = timestamps[i + self.lookback_window - 1]
                future_timestamps = timestamps[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon]
                
                # Convert timestamps to numerical features
                if isinstance(current_timestamp, (pd.Timestamp, np.datetime64)):
                    current_time_features = [
                        current_timestamp.year,
                        current_timestamp.month,
                        current_timestamp.day,
                        current_timestamp.hour,
                        current_timestamp.dayofweek,
                        current_timestamp.dayofyear
                    ]
                else:
                    # If timestamps are already numerical, use as is
                    current_time_features = [current_timestamp]
                
                # Add future timestamp features
                future_time_features = []
                for ts in future_timestamps:
                    if isinstance(ts, (pd.Timestamp, np.datetime64)):
                        future_time_features.extend([
                            ts.year, ts.month, ts.day, ts.hour, ts.dayofweek, ts.dayofyear
                        ])
                    else:
                        future_time_features.append(ts)
                
                sample_features.extend(current_time_features + future_time_features)
            
            # Add exogenous features if available
            if x_series is not None and len(x_series) >= i + self.lookback_window:
                x_lags = x_series[i:i + self.lookback_window]
                sample_features.extend(x_lags.flatten())
            
            features.append(sample_features)
            
            # Create multi-output targets: flatten to (forecast_horizon * n_targets)
            target_values = []
            for step in range(self.forecast_horizon):
                for target_idx in range(n_targets):
                    target_values.append(y_series[i + self.lookback_window + step, target_idx])
            targets.append(target_values)
        
        return np.array(features), np.array(targets)

    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None, y_start_date: pd.Timestamp = None, x_start_date: pd.Timestamp = None, y_context_timestamps: np.ndarray = None, y_target_timestamps: np.ndarray = None, **kwargs) -> 'RandomForestModel':
        """
        Train the Random Forest model on given data.
        
        TECHNIQUE: Single Model with Timestamp Features
        - Creates lag features from historical target values
        - Adds rolling statistics (mean, std, min, max)
        - Includes trend features using linear regression
        - Incorporates timestamp features for time-aware splits
        - Uses a single model to predict all forecast horizon steps
        
        Args:
            y_context: Past target values (time series) - used for training
            y_target: Future target values (optional, for validation) - NOT used for feature creation
            x_context: Past exogenous variables (optional)
            x_target: Future exogenous variables (optional)
            y_start_date: The start date timestamp for y_context and y_target in string form
            x_start_date: The start date timestamp for x_context and x_target in string form
            y_context_timestamps: Timestamps for y_context (optional)
            y_target_timestamps: Timestamps for y_target (optional)
        
        Returns:
            self: The fitted model instance.
        """
        if self.model is None:
            self._build_model()
        
        # Convert inputs to numpy arrays
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_data = y_context.values
        else:
            y_data = y_context
            
        if isinstance(x_context, (pd.Series, pd.DataFrame)):
            x_data = x_context.values
        else:
            x_data = x_context
        
        # Handle multivariate data - don't flatten, let _create_features handle it
        # Ensure y_data is 2D for multivariate processing
        if y_data.ndim == 1:
            y_data = y_data.reshape(-1, 1)
        
        # Use ONLY training data (y_context) for feature creation to avoid data leakage
        # Do NOT combine with y_target during training
        training_y_data = y_data
        
        # Use only training timestamps for feature creation
        training_timestamps = y_context_timestamps
        
        # Create features and targets using ONLY training data
        X, y = self._create_features(training_y_data, x_data, training_timestamps)
        
        # y is shape (n_samples, forecast_horizon * n_targets)
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, y_context: Union[pd.Series, np.ndarray] = None, y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None, y_context_timestamps: np.ndarray = None, y_target_timestamps: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Make predictions using the trained Random Forest model.
        
        TECHNIQUE: Single Model with Timestamp Features
        - Uses last lookback_window values to create features
        - Incorporates timestamp features for time-aware prediction
        - Predicts all forecast horizon steps with a single model
        
        Args:
            y_context: Past target values (time series) - used for prediction
            y_target: Future target values - used to determine prediction length
            x_context: Past exogenous variables (optional)
            x_target: Future exogenous variables (optional)
            y_context_timestamps: Timestamps for y_context (optional)
            y_target_timestamps: Timestamps for y_target (optional)
            
        Returns:
            np.ndarray: Model predictions with shape (1, forecast_steps)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")
        
        forecast_steps = len(y_target)
        
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_data = y_context.values
        else:
            y_data = y_context
        
        if isinstance(x_context, (pd.Series, pd.DataFrame)):
            x_data = x_context.values
        else:
            x_data = x_context
        
        # Handle multivariate data - ensure 2D
        if y_data.ndim == 1:
            y_data = y_data.reshape(-1, 1)
        
        # Use the last available context to create a single feature row
        # For prediction, we want to predict forecast_horizon steps ahead (as the model was trained)
        # So we need to create a feature row for the current context and the next forecast_horizon timestamps
        if y_context_timestamps is not None and y_target_timestamps is not None:
            # Use the last lookback_window context and the actual target timestamps
            context_timestamps = y_context_timestamps[-self.lookback_window:]
            # Create dummy zeros with correct shape for multivariate - use forecast_horizon, not forecast_steps
            dummy_targets = np.zeros((self.forecast_horizon, y_data.shape[1]))
            feature_row, _ = self._create_features(
                np.concatenate([y_data, dummy_targets], axis=0),
                x_data,
                np.concatenate([y_context_timestamps, y_target_timestamps])
            )
            X_last = feature_row[-1:].reshape(1, -1)
        else:
            # Fallback: no timestamps
            dummy_targets = np.zeros((self.forecast_horizon, y_data.shape[1]))
            feature_row, _ = self._create_features(
                np.concatenate([y_data, dummy_targets], axis=0),
                x_data,
                None
            )
            X_last = feature_row[-1:].reshape(1, -1)
        
        # Predict all steps at once
        preds = self.model.predict(X_last)
        
        # Reshape predictions to (forecast_horizon, n_targets) and return only requested steps
        preds_reshaped = preds.reshape(self.forecast_horizon, y_data.shape[1])
        return preds_reshaped[:forecast_steps]
    
    def rolling_predict(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, pd.DataFrame, np.ndarray], x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None, y_context_timestamps: np.ndarray = None, y_target_timestamps: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Generate full-length predictions for the test set using a rolling window approach.
        This method repeatedly calls the model's predict method, each time advancing the context by forecast_horizon steps,
        until the total number of predictions matches the length of y_target.
        Supports both univariate and multivariate data.

        Args:
            y_context: Initial context (typically last lookback_window values from train set)
            y_target: The full test set (used to determine total prediction length)
            x_context, x_target: Exogenous variables (optional)
            y_context_timestamps, y_target_timestamps: Timestamps (optional)
            **kwargs: Additional keyword arguments
        Returns:
            np.ndarray: Full-length predictions matching the length of y_target
        """
        # Handle multivariate data
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            context = y_context.values
        else:
            context = np.copy(y_context)
            
        if context.ndim == 1:
            context = context.reshape(-1, 1)
            
        # Determine number of targets from context
        n_targets = context.shape[1]
        
        preds = []
        steps_remaining = len(y_target)
        idx = 0
        while steps_remaining > 0:
            steps = min(self.forecast_horizon, steps_remaining)
            # Create dummy targets with correct shape for multivariate
            dummy_y_target = np.zeros((steps, n_targets))
            pred = self.predict(
                y_context=context,
                y_target=dummy_y_target,
                x_context=x_context,
                x_target=x_target,
                y_context_timestamps=y_context_timestamps,
                y_target_timestamps=y_target_timestamps,
                **kwargs
            )
            # pred is now already in the correct shape (steps, n_targets)
            preds.extend(pred)
            # Update context with the predictions
            context = np.concatenate([context, pred], axis=0)
            steps_remaining -= steps
            idx += steps
        return np.array(preds)

    def _create_step_features(self, y_data: np.ndarray, x_data: np.ndarray = None, context_timestamps: np.ndarray = None, target_timestamp: np.ndarray = None) -> np.ndarray:
        """
        Create features for a single prediction step.
        
        Args:
            y_data: Historical target values
            x_data: Historical exogenous variables (optional)
            context_timestamps: Timestamps for context data (optional)
            target_timestamp: Timestamp for the target step (optional)
            
        Returns:
            np.ndarray: Feature vector for the prediction step
        """
        # Use the last lookback_window values
        lag_features = y_data[-self.lookback_window:]
        
        # Create rolling statistics
        rolling_mean = np.mean(lag_features)
        rolling_std = np.std(lag_features)
        rolling_min = np.min(lag_features)
        rolling_max = np.max(lag_features)
        
        # Create trend features
        trend = np.polyfit(range(self.lookback_window), lag_features, 1)[0]
        
        # Combine all features
        sample_features = list(lag_features) + [rolling_mean, rolling_std, rolling_min, rolling_max, trend]
        
        # Add timestamp features if available
        if context_timestamps is not None and target_timestamp is not None:
            current_timestamp = context_timestamps[-1]
            
            # Convert timestamps to numerical features
            if isinstance(current_timestamp, (pd.Timestamp, np.datetime64)):
                current_time_features = [
                    current_timestamp.year,
                    current_timestamp.month,
                    current_timestamp.day,
                    current_timestamp.hour,
                    current_timestamp.dayofweek,
                    current_timestamp.dayofyear
                ]
            else:
                current_time_features = [current_timestamp]
            
            # Add target timestamp features
            if isinstance(target_timestamp, (pd.Timestamp, np.datetime64)):
                target_time_features = [
                    target_timestamp.year,
                    target_timestamp.month,
                    target_timestamp.day,
                    target_timestamp.hour,
                    target_timestamp.dayofweek,
                    target_timestamp.dayofyear
                ]
            else:
                target_time_features = [target_timestamp]
            
            sample_features.extend(current_time_features + target_time_features)
        
        # Add exogenous features if available
        if x_data is not None:
            x_lags = x_data[-self.lookback_window:]
            sample_features.extend(x_lags.flatten())
        
        return np.array(sample_features)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        """
        params = {}
        
        # Add model-specific parameters
        params['lookback_window'] = self.lookback_window
        params['forecast_horizon'] = self.forecast_horizon
        params['target_cols'] = self.target_cols
        params['n_targets'] = self.n_targets
        
        # Add RandomForestRegressor parameters
        if self.model:
            rf_params = self.model.get_params()
            params.update(rf_params)
            
        return params

    def set_params(self, **params: Dict[str, Any]) -> 'RandomForestModel':
        """
        Set parameters for the Random Forest model.
        Rebuilds the model if target_cols changes to ensure proper feature dimensions.
        """
        # Check if target_cols is being changed
        target_cols_changed = 'target_cols' in params and params['target_cols'] != getattr(self, 'target_cols', None)
        
        # Update config with new parameters
        self.config.update(params)
        
        # If target_cols changed, we need to rebuild the model completely
        if target_cols_changed:
            self._build_model()
        else:
            # For other parameters, just update the model's parameters
            model_params = {}
            for key, value in params.items():
                if key not in ['lookback_window', 'forecast_horizon', 'target_cols']:
                    model_params[key] = value
            
            # Filter out keys that are not valid for RandomForestRegressor
            valid_keys = set(RandomForestRegressor().get_params().keys())
            filtered_params = {k: v for k, v in model_params.items() if k in valid_keys}
            
            if filtered_params:
                self.model.set_params(**filtered_params)
        
        # Reset fitted state when parameters change
        self.is_fitted = False
        
        return self

    def save(self, path: str) -> None:
        """
        Save the trained scikit-learn model to disk using pickle.
        
        Args:
            path: Path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
        
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        model_state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'lookback_window': self.lookback_window,
            'forecast_horizon': self.forecast_horizon,
            'target_cols': self.target_cols,
            'n_targets': self.n_targets,
            'model': self.model
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
            
    def load(self, path: str) -> 'RandomForestModel':
        """
        Load a trained scikit-learn model from disk.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        self.config = model_state['config']
        self.is_fitted = model_state['is_fitted']
        self.lookback_window = model_state['lookback_window']
        self.forecast_horizon = model_state['forecast_horizon']
        self.target_cols = model_state['target_cols']
        self.n_targets = model_state['n_targets']
        self.model = model_state['model']
        
        return self