"""
Random Forest model implementation for time series forecasting.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, List
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
        """
        # Get hyperparameters from config, with sensible defaults from sklearn
        model_params = self.config.get('model_params', {})

        # Filter out keys that are not valid for RandomForestRegressor
        valid_keys = set(RandomForestRegressor().get_params().keys())
        filtered_params = {k: v for k, v in model_params.items() if k in valid_keys}

        if 'random_state' not in filtered_params:
            filtered_params['random_state'] = 42

        if 'n_jobs' not in filtered_params:
            filtered_params['n_jobs'] = -1
            
        self.model = RandomForestRegressor(**filtered_params)
        self.is_fitted = False

    def _create_features(self, y_series: np.ndarray, x_series: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time series features for Random Forest.
        
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
            
            # Add exogenous features if available
            if x_series is not None and len(x_series) >= i + self.lookback_window:
                x_lags = x_series[i:i + self.lookback_window]
                sample_features.extend(x_lags.flatten())
            
            features.append(sample_features)
            targets.append(y_series[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon])
        
        return np.array(features), np.array(targets)

    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None, y_start_date: pd.Timestamp = None, x_start_date: pd.Timestamp = None) -> 'RandomForestModel':
        """
        Train the Random Forest model on given data.
        
        TECHNIQUE: Lag-based Feature Engineering
        - Creates lag features from historical target values
        - Adds rolling statistics (mean, std, min, max)
        - Includes trend features using linear regression
        - Incorporates exogenous variables if available
        - Trains separate model for each forecast horizon step
        
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
        
        # For multi-step forecasting, train separate models
        if self.forecast_horizon > 1:
            self.models = []
            print(f"Training {self.forecast_horizon} Random Forest models for multi-step forecasting...")
            for i in range(self.forecast_horizon):
                model = RandomForestRegressor(**self.config.get('model_params', {}))
                model.fit(X, y[:, i])
                self.models.append(model)
            print(f"Training complete. Models can predict up to {self.forecast_horizon} steps directly, or more using iterative prediction.")
        else:
            # Single-step forecasting
            self.model.fit(X, y.flatten())
        
        self.is_fitted = True
        return self
        
    def predict(self, y_context: Union[pd.Series, np.ndarray] = None, y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the trained Random Forest model.
        
        TECHNIQUE: Multi-step Forecasting with Lag Features
        - Uses last lookback_window values to create features
        - Predicts forecast_horizon steps ahead
        - For multi-step: uses separate models for each step
        - For single-step: uses single model
        
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
            
        if isinstance(x_context, (pd.Series, pd.DataFrame)):
            x_data = x_context.values
        else:
            x_data = x_context
        
        # Ensure y_data is 1D
        if y_data.ndim > 1:
            y_data = y_data.flatten()
        
        # Create features for prediction
        X_pred, _ = self._create_features(y_data, x_data)
        
        if len(X_pred) == 0:
            raise ValueError("Not enough data for prediction. Need at least lookback_window samples.")
        
        # Use the last sample for prediction
        X_last = X_pred[-1:].reshape(1, -1)
        
        # Make predictions
        if hasattr(self, 'models') and self.forecast_horizon > 1:
            # Multi-step forecasting
            predictions = []
            
            if forecast_steps > self.forecast_horizon:
                print(f"Warning: Requesting {forecast_steps} predictions but forecast_horizon is {self.forecast_horizon}. Using true iterative prediction.")
                # Prepare context for iterative prediction
                y_context_iter = y_data.copy()
                x_context_iter = x_data.copy() if x_data is not None else None
                for step in range(forecast_steps):
                    model_idx = step % self.forecast_horizon
                    # Create features for the current context
                    X_step, _ = self._create_features(y_context_iter, x_context_iter)
                    X_last_step = X_step[-1:].reshape(1, -1)
                    pred = self.models[model_idx].predict(X_last_step)[0]
                    predictions.append(pred)
                    # Update context with new prediction
                    y_context_iter = np.append(y_context_iter, pred)
                    if x_context_iter is not None and x_target is not None:
                        # If exogenous variables, append the next x_target row
                        if step < len(x_target):
                            x_context_iter = np.vstack([x_context_iter, x_target[step]])
                
            else:
                # Standard multi-step prediction
                X_last = X_pred[-1:].reshape(1, -1)
                for i in range(forecast_steps):
                    pred = self.models[i].predict(X_last)[0]
                    predictions.append(pred)
                
        else:
            # Single-step forecasting
            predictions = [self.model.predict(X_last)[0]] * forecast_steps
        
        return np.array(predictions).reshape(1, -1)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        """
        if hasattr(self, 'models') and self.forecast_horizon > 1:
            return {f'model_{i}': model.get_params() for i, model in enumerate(self.models)}
        elif self.model:
            return self.model.get_params()
        return self.config.get('model_params', {})

    def set_params(self, **params: Dict[str, Any]) -> 'RandomForestModel':
        """
        Set model parameters. This will rebuild the model.
        """
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        
        # Handle nested parameter names like 'model_params__n_estimators'
        for key, value in params.items():
            if '__' in key:
                # Split on '__' to get the nested structure
                parts = key.split('__')
                if parts[0] == 'model_params':
                    # This is a model parameter
                    self.config['model_params'][parts[1]] = value
                else:
                    # This is a top-level parameter
                    self.config[parts[0]] = value
            else:
                # This is a top-level parameter
                self.config[key] = value
        
        # Rebuild the model with new parameters
        self._build_model()
        
        # If we have trained models, update them too
        if hasattr(self, 'models') and self.forecast_horizon > 1 and self.is_fitted:
            self.models = []
            filtered_params = self._get_filtered_model_params()
            for i in range(self.forecast_horizon):
                model = RandomForestRegressor(**filtered_params)
                self.models.append(model)
        
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
            'forecast_horizon': self.forecast_horizon
        }
        
        if hasattr(self, 'models'):
            model_state['models'] = self.models
        else:
            model_state['model'] = self.model
        
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
        
        if 'models' in model_state:
            self.models = model_state['models']
        else:
            self.model = model_state['model']
        
        return self