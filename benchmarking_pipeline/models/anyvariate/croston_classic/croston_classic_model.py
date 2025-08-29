"""
Croston's Classic Model implementation for intermittent demand forecasting.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Union
import pickle
import os


from benchmarking_pipeline.models.base_model import BaseModel

class CrostonClassicModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the Croston's Classic model with a given configuration.
        
        Args:
            config: Configuration dictionary containing model parameters.
                - alpha: float, smoothing parameter for demand level (0 < alpha < 1)
                - gamma: float, smoothing parameter for interval level (0 < gamma < 1)
                - phi: float, trend damping parameter (0 < phi < 1)
                - forecast_horizon: int, number of steps to forecast ahead
                - training_loss: str, primary loss function for training
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        # Smoothing parameter, with a common default of 0.1
        self.alpha = self.config.get('alpha', 0.4)
        self.beta = self.config.get('beta', 0.1)
        self.gamma = self.config.get('gamma', 0.1)
        self.phi = self.config.get('phi', 0.9)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        
        self.is_fitted = False
        
        # Fitted parameters, initialized to None
        self.demand_level_ = None
        self.interval_level_ = None
        
    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, **kwargs) -> 'CrostonClassicModel':
        print(f"[Croston train] y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}")
        """
        Train the Croston's Classic model on the given time series data.
        
        The method decomposes the series into non-zero demand values and the
        time intervals between them, then applies Simple Exponential Smoothing
        to both series.
        
        Args:
            y_context: Past target values - the training data for the time series.
            y_target: Not used by the Croston's Classic model, but included for compatibility with the base class.
            
        Returns:
            self: The fitted model instance.
        """
        # Convert y_context to a 1D numpy array of numbers
        if isinstance(y_context, pd.Series):
            series = y_context.values
        elif isinstance(y_context, np.ndarray):
            series = np.asarray(y_context).flatten()
    
        series = np.atleast_1d(series)

        # Find indices of non-zero demand
        non_zero_indices = np.nonzero(series)

        # If there's no demand or only one demand point we cannot forecast
        if len(non_zero_indices[0]) < 2:
            # Set levels to a default state (e.g., average of the whole series)
            self.demand_level_ = np.mean(series) if len(series) > 0 else 0
            
            self.interval_level_ = len(series) if len(series) > 0 else 1
            self.is_fitted = True
            return self

        # Get demand values and intervals
        demands = series[non_zero_indices]
        intervals = np.diff(non_zero_indices[0])
        
        # Demand level starts with the first non-zero demand
        current_demand_level = demands[0]
        # Interval level starts with the first interval
        current_interval_level = intervals[0]
        
        # Apply SES to the rest of the demands and intervals
        for i in range(1, len(demands)):
            current_demand_level = self.alpha * demands[i] + (1 - self.alpha) * current_demand_level
        
        for i in range(1, len(intervals)):
            current_interval_level = self.alpha * intervals[i] + (1 - self.alpha) * current_interval_level
            
        self.demand_level_ = current_demand_level
        self.interval_level_ = current_interval_level
        self.is_fitted = True
        
        return self

    def predict(self, y_context, y_target=None, y_context_timestamps=None, y_target_timestamps=None, **kwargs):
        print(f"[Croston predict] y_context type: {type(y_context)}, shape: {getattr(y_context, 'shape', 'N/A')}")
        print(f"[Croston predict] y_target type: {type(y_target)}, shape: {getattr(y_target, 'shape', 'N/A')}")
        """
        Make predictions using the trained Croston's Classic model.
        
        Args:
            y_target: Used to determine the number of steps to forecast.
            y_context, x_context, x_target: Not used.
            
        Returns:
            np.ndarray: Model predictions with shape (1, forecast_horizon).
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")
        
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")
        
        forecast_steps = len(y_target)
        
        # Avoid division by zero
        if self.interval_level_ is None or self.interval_level_ == 0:
            forecast_value = 0
        else:
            forecast_value = self.demand_level_ / self.interval_level_
            
        # Croston's forecast is a constant value for all future steps
        forecast = np.full(forecast_steps, forecast_value)
        
        return forecast.reshape(1, -1) # Reshape to (1, forecast_steps)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters.
        """
        return {
            'alpha': self.alpha,
            'training_loss': self.training_loss,
            'forecast_horizon': self.forecast_horizon,
            'is_fitted': self.is_fitted
        }

    def set_params(self, **params: Dict[str, Any]) -> 'CrostonClassicModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set.
            
        Returns:
            self: The model instance with updated parameters.
        """
        model_param_changed = False

        for key, value in params.items():
            if hasattr(self, key):
                if key == 'alpha' and getattr(self, key) != value:
                    model_param_changed = True
                setattr(self, key, value)
            else:
                self.config[key] = value

        if model_param_changed and self.is_fitted:
            # If a core model parameter changes, the model needs to be retrained
            self.is_fitted = False
            self.demand_level_ = None
            self.interval_level_ = None
            
        return self

    def save(self, path: str) -> None:
        """
        Save the Croston's Classic model to disk.
        
        Args:
            path: Path to save the model file.
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model. Call train() first.")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'params': self.get_params(),
            'demand_level_': self.demand_level_,
            'interval_level_': self.interval_level_,
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(model_state, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {path}: {str(e)}")

    def load(self, path: str) -> None:
        """
        Load a Croston's Classic model from disk.
        
        Args:
            path: Path to the saved model file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")

        try:
            with open(path, 'rb') as f:
                model_state = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}")
            
        self.config = model_state['config']
        self.is_fitted = model_state['is_fitted']
        self.set_params(**model_state['params'])
        self.demand_level_ = model_state['demand_level_']
        self.interval_level_ = model_state['interval_level_']

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the Croston's Classic model's properties.
        
        Returns:
            Dict[str, Any]: A dictionary containing model summary information.
        """
        summary = {
            'model_type': 'CrostonClassic',
            'alpha': self.alpha,
            'is_fitted': self.is_fitted,
            'forecast_horizon': self.forecast_horizon,
        }
        
        if self.is_fitted:
            summary.update({
                'fitted_demand_level': self.demand_level_,
                'fitted_interval_level': self.interval_level_
            })
            
        return summary