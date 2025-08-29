"""
Exponential Smoothing model implementation.
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
                    e.g., {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 12, ...}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        def _cast_param(key, value):
            if key == 'seasonal_periods':
                return int(value) if value is not None else None
            if key == 'damped_trend':
                if isinstance(value, str):
                    return value.lower() == 'true'
                return bool(value)
            if key == 'forecast_horizon':
                return int(value) if value is not None else 1
            if key in ['trend', 'seasonal']:
                if isinstance(value, str) and value.lower() == 'none':
                    return None
                return value
            return value
        self.trend = _cast_param('trend', self.config.get('trend', None))
        self.seasonal = _cast_param('seasonal', self.config.get('seasonal', None))
        self.seasonal_periods = _cast_param('seasonal_periods', self.config.get('seasonal_periods', None))
        self.damped_trend = _cast_param('damped_trend', self.config.get('damped_trend', False))
        self.model_ = None
        self.is_fitted = False
        self.training_loss = self.config.get('training_loss', 'mae')
        self.forecast_horizon = _cast_param('forecast_horizon', self.config.get('forecast_horizon', 1))

    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, np.ndarray] = None, x_target: Union[pd.Series, np.ndarray] = None, **kwargs) -> 'ExponentialSmoothingModel':
        print(f"[ExponentialSmoothing train] y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}")
        # Ensure correct types for model parameters
        trend = self.trend
        seasonal = self.seasonal
        if isinstance(trend, str) and trend.lower() == 'none':
            trend = None
        if isinstance(seasonal, str) and seasonal.lower() == 'none':
            seasonal = None
        seasonal_periods = int(self.seasonal_periods) if self.seasonal_periods is not None else None
        damped_trend = bool(self.damped_trend)
        if isinstance(damped_trend, str):
            damped_trend = damped_trend.lower() == 'true'
        # Only allow damped_trend if trend is not None
        if trend is None:
            damped_trend = None
        if isinstance(y_context, pd.Series):
            endog = y_context.values
        else:
            endog = y_context
        self.model_ = ExponentialSmoothing(
            endog,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend
        ).fit()
        self.is_fitted = True
        return self

    def predict(self, y_context: Union[pd.Series, np.ndarray] = None, y_target: Union[pd.Series, np.ndarray] = None, **kwargs) -> np.ndarray:
        print(f"[ExponentialSmoothing predict] y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}")
        print(f"[ExponentialSmoothing predict] y_target type: {type(y_target)}, shape: {getattr(y_target, 'shape', 'N/A')}")
        if not self.is_fitted:
            raise ValueError("Model not initialized. Call train first.")
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")
        forecast_steps = len(y_target)
        forecast = self.model_.forecast(steps=forecast_steps)
        # Ensure numpy array for consistent reshaping
        if hasattr(forecast, 'values'):
            forecast_array = forecast.values
        else:
            forecast_array = np.array(forecast)
        return forecast_array.reshape(1, -1)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the configuration and instance attributes.
        """
        return {
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'damped_trend': self.damped_trend,
            'training_loss': self.training_loss,
            'forecast_horizon': self.forecast_horizon,
            'is_fitted': self.is_fitted
        }

    def set_params(self, **params: Dict[str, Any]) -> 'ExponentialSmoothingModel':
        """
        Set model parameters by updating the configuration and instance attributes.
        The model will be rebuilt with these new parameters on the next .train() call.
        """
        def _cast_param(key, value):
            if key == 'seasonal_periods':
                return int(value) if value is not None else None
            if key == 'damped_trend':
                if isinstance(value, str):
                    return value.lower() == 'true'
                return bool(value)
            if key == 'forecast_horizon':
                return int(value) if value is not None else 1
            if key in ['trend', 'seasonal']:
                if isinstance(value, str) and value.lower() == 'none':
                    return None
                return value
            return value
        for key, value in params.items():
            casted_value = _cast_param(key, value)
            setattr(self, key, casted_value)
            self.config[key] = casted_value
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