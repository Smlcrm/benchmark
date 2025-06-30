"""
XGBoost model implementation for time series forecasting.
"""
import os
import pickle
from typing import Dict, Any, Union, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from benchmarking_pipeline.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize XGBoost model with a given configuration.
        
        Args:
            config: Configuration dictionary for XGBoostRegressor parameters.
                    e.g., {'n_estimators': 100, 'learning_rate': 0.1, 'lookback_window': 10}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        self.lookback_window = self.config.get('lookback_window', 10)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        self._build_model()
        
    def _build_model(self):
        """
        Build the XGBRegressor model instance from the configuration.
        """
        # Get hyperparameters from config.
        model_params = self.config.get('model_params', {})
        
        # Ensure random_state for reproducibility if not provided
        if 'random_state' not in model_params:
            model_params['random_state'] = 42
            
        self.model = XGBRegressor(**model_params)
        self.is_fitted = False

    def _create_features(self, y_series: np.ndarray, x_series: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series features for XGBoost.
        
        Args:
            y_series: Target time series
            x_series: Exogenous variables (optional)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets
        """
        n_samples = len(y_series) - self.lookback_window - self.forecast_horizon + 1
        
        if n_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {self.lookback_window + self.forecast_horizon} samples.")
        
        features = []
        targets = []
        
        for i in range(n_samples):
            # Create lag features
            lag_features = y_series[i:i + self.lookback_window]
            
            # Create rolling statistics
            rolling_mean = np.mean(lag_features)
            rolling_std = np.std(lag_features)
            rolling_min = np.min(lag_features)
            rolling_max = np.max(lag_features)
            
            # Create trend features
            trend = np.polyfit(range(self.lookback_window), lag_features, 1)[0]
            
            # Combine all features
            sample_features = list(lag_features) + [rolling_mean, rolling_std, rolling_min, rolling_max, trend]
            
            # Add current exogenous features if available
            if x_series is not None and len(x_series) > i + self.lookback_window:
                current_x = x_series[i + self.lookback_window]
                sample_features.extend(current_x.flatten())
            
            features.append(sample_features)
            targets.append(y_series[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon])
        
        return np.array(features), np.array(targets)

    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None) -> 'XGBoostModel':
        """
        Train the XGBoost model on given data.
        
        TECHNIQUE: Lag-based Feature Engineering with Gradient Boosting
        - Creates lag features from historical target values
        - Adds rolling statistics (mean, std, min, max)
        - Includes trend features using linear regression
        - Incorporates current exogenous variables if available
        - Uses XGBoost's gradient boosting for non-linear pattern learning
        
        Args:
            y_context: Past target values (time series) - used for training
            y_target: Future target values (optional, for validation)
            x_context: Past exogenous variables (optional)
            x_target: Future exogenous variables (optional)
        
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
        
        # Ensure y_data is 1D
        if y_data.ndim > 1:
            y_data = y_data.flatten()
        
        # Create features and targets
        X, y = self._create_features(y_data, x_data)
        
        print(f"Training XGBoost model with {X.shape[0]} samples and {X.shape[1]} features...")
        
        # Train single model for multi-output regression
        self.model.fit(X, y)
        
        self.is_fitted = True
        print("Training complete.")
        return self
        
    def predict(self, y_context: Union[pd.Series, np.ndarray] = None, y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the trained XGBoost model.
        
        TECHNIQUE: Autoregressive Rolling Window Multi-step Forecasting
        - Uses last lookback_window values to create features
        - Predicts forecast_horizon steps ahead using single model
        - Uses its own predictions to update the window and predict further steps
        - Repeats until forecast_steps are reached
        
        Args:
            y_context: Past target values (time series) - used for prediction
            y_target: Future target values - used to determine prediction length
            x_context: Past exogenous variables (optional)
            x_target: Future exogenous variables (optional)
            
        Returns:
            np.ndarray: Model predictions with shape (1, forecast_steps)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")
        
        forecast_steps = len(y_target)
        
        # Convert inputs to numpy arrays
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_data = y_context.values
        else:
            y_data = y_context
        if y_data.ndim > 1:
            y_data = y_data.flatten()
        
        if x_context is not None and isinstance(x_context, (pd.Series, pd.DataFrame)):
            x_data = x_context.values
        else:
            x_data = x_context
        
        predictions = []
        window = list(y_data[-self.lookback_window:])
        x_covariates = None
        if x_data is not None:
            x_covariates = x_data[-self.lookback_window:] if len(x_data) >= self.lookback_window else None
        
        steps_remaining = forecast_steps
        while len(predictions) < forecast_steps:
            # Create features for the current window
            lag_features = np.array(window)
            rolling_mean = np.mean(lag_features)
            rolling_std = np.std(lag_features)
            rolling_min = np.min(lag_features)
            rolling_max = np.max(lag_features)
            trend = np.polyfit(range(self.lookback_window), lag_features, 1)[0]
            sample_features = list(lag_features) + [rolling_mean, rolling_std, rolling_min, rolling_max, trend]
            # Add current exogenous features if available
            if x_covariates is not None and len(x_covariates) == self.lookback_window:
                current_x = x_covariates[-1]
                sample_features.extend(current_x.flatten())
            X_last = np.array(sample_features).reshape(1, -1)
            # Predict up to forecast_horizon steps
            pred = self.model.predict(X_last)[0]
            # If pred is a scalar, make it a list
            if np.isscalar(pred):
                pred = [pred]
            # Only take as many steps as needed
            steps_to_take = min(self.forecast_horizon, steps_remaining)
            predictions.extend(pred[:steps_to_take])
            # Update window for next prediction
            window = window[steps_to_take:] + list(pred[:steps_to_take])
            if x_covariates is not None:
                # For exogenous, just repeat the last value (or you could advance if you have x_target)
                x_covariates = np.vstack([x_covariates[steps_to_take:], x_covariates[-1:]]) if x_covariates.shape[0] > 0 else x_covariates
            steps_remaining -= steps_to_take
        return np.array(predictions[:forecast_steps]).reshape(1, -1)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the underlying xgboost model.
        """
        if self.model:
            return self.model.get_params()
        return self.config.get('model_params', {})

    def set_params(self, **params: Dict[str, Any]) -> 'XGBoostModel':
        """
        Set model parameters. This will rebuild the model instance.
        """
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        self.config['model_params'].update(params)
        
        if self.model:
            self.model.set_params(**params)
        
        return self

    def save(self, path: str) -> None:
        """
        Save the trained xgboost model to disk using pickle.
        
        Args:
            path: Path to save the model.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        model_state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'lookback_window': self.lookback_window,
            'forecast_horizon': self.forecast_horizon,
            'model': self.model
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
            
    def load(self, path: str) -> 'XGBoostModel':
        """
        Load a trained xgboost model from disk.
        
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
        self.model = model_state['model']
        
        return self