"""
Exponential Smoothing model implementation.

TO BE CHANGED: This model needs to be updated to match the new interface with y_context, x_context, y_target, x_target parameters.
"""

import os
import pickle
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from benchmarking_pipeline.models.base_model import BaseModel


class ExponentialSmoothingModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Exponential Smoothing model with a given configuration.
        
        Args:
            config: Configuration dictionary for model parameters.
                    e.g., {'model_params': {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12}}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        self.trend = self.config.get('trend', None)
        self.seasonal = self.config.get('seasonal', None)
        self.seasonal_periods = self.config.get('seasonal_periods', None)
        self.damped_trend = self.config.get('damped_trend', False)
        self.model_ = None

    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, np.ndarray] = None, x_target: Union[pd.Series, np.ndarray] = None) -> 'ExponentialSmoothingModel':
        """
        Train the Exponential Smoothing model on given data.

        Args:
            y_context: Past target values (time series) - used for training
        
        Returns:
            self: The fitted model instance
        """
        if isinstance(y_context, pd.Series):
            endog = y_context.values
        else:
            endog = y_context
        # Optionally, could use y_target for validation/early stopping in future
        self.model_ = ExponentialSmoothing(
            endog,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend
        ).fit()
        self.is_fitted = True
        return self

    def predict(self, y_context: Union[pd.Series, np.ndarray] = None, y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the trained Exponential Smoothing model.

        Returns:
            np.ndarray: Model predictions with shape (1, forecast_horizon)
        """
        if not self.is_fitted:
            raise ValueError("Model not initialized. Call train first.")
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")
        forecast_steps = len(y_target)
        forecast = self.model_.forecast(steps=forecast_steps)
        return forecast.reshape(1, -1)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the configuration.
        """
        return self.config.get('model_params', {})

    def set_params(self, **params: Dict[str, Any]) -> 'ExponentialSmoothingModel':
        """
        Set model parameters by updating the configuration.
        The model will be rebuilt with these new parameters on the next .train() call.
        """
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        self.config['model_params'].update(params)
        
        self.is_fitted = False
        self.model_ = None
        
        return self

    def save(self, path: str) -> None:
        """
        Save the trained statsmodels ETS model to disk using pickle.
        
        Args:
            path: Path to save the model.
        """
        if not self.is_fitted or self.model_ is None:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # The fitted results object from statsmodels can be pickled
        with open(path, 'wb') as f:
            pickle.dump(self.model_, f)
            
    def load(self, path: str) -> 'ExponentialSmoothingModel':
        """
        Load a trained statsmodels ETS model from disk.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            self.model_ = pickle.load(f)
        self.is_fitted = True
        return self