"""
ARIMA model implementation.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, Any, Union, Tuple, Optional
import pickle
import os
from benchmarking_pipeline.models.base_model import BaseModel


class ARIMAModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize ARIMA model with given configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
                - p: int, AR order
                - d: int, differencing order
                - q: int, MA order
                - s: int, seasonality
                - target_col: str, name of target column
                - loss_functions: List[str], list of loss function names to use
                - primary_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, config_file)
        self.p = self.config.get('p', 1)
        self.d = self.config.get('d', 1)
        self.q = self.config.get('q', 1)
        self.s = self.config.get('s', 1)
        self.target_col = self.config.get('target_col', 'y')
        self.model_ = None  # Use model_ consistently
        self.is_fitted = False  # Explicitly initialize
        self.loss_functions = self.config.get('loss_functions', ['mae'])
        self.primary_loss = self.config.get('primary_loss', self.loss_functions[0])
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        
    def train(self, 
              y_context: Union[pd.Series, np.ndarray], 
              y_target: Union[pd.Series, np.ndarray] = None, 
              x_context: Union[pd.Series, np.ndarray] = None, 
              x_target: Union[pd.Series, np.ndarray] = None, 
              y_start_date: Optional[str] = None,
              x_start_date: Optional[str] = None,
              **kwargs
    ) -> 'ARIMAModel':
        """
        Train the ARIMA model on given data.
        
        Args:
            y_context: Past target values - training data
            x_context: Past exogenous variables - used during training
        Returns:
            self: The fitted model instance
        """
        # Use y_context as the endogenous variable (training data)
        if isinstance(y_context, pd.Series):
            endog = y_context.values
        else:
            endog = y_context
        
        # Handle exogenous variables if provided (only x_context)
        exog = None
        if x_context is not None:
            if isinstance(x_context, (pd.Series, pd.DataFrame)):
                exog = x_context.values
            else:
                exog = x_context
        
        model = ARIMA(endog=endog, seasonal_order=(self.p, self.d, self.q, self.s), exog=exog)
        self.model_ = model.fit()
        self.is_fitted = True
        return self
        
    def predict(self, y_context, y_target=None, x_context=None, x_target=None, y_start_date=None, x_start_date=None, **kwargs):
        """
        Make predictions using the trained ARIMA model.
        
        Args:
            y_target: Used to determine the number of steps to forecast
            x_target: Future exogenous variables for prediction (optional)
        
        Returns:
            np.ndarray: Model predictions with shape (1, forecast_horizon)
        """
        if not self.is_fitted:
            raise ValueError("Model not initialized. Call train first.")
        
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")
        
        # Handle exogenous variables for prediction (only x_target)
        exog = None
        if x_target is not None:
            if isinstance(x_target, (pd.Series, pd.DataFrame)):
                exog = x_target.values
            else:
                exog = x_target
        
        forecast_steps = len(y_target)
        forecast = self.model_.forecast(steps=forecast_steps, exog=exog)
        return forecast.reshape(1, -1)  # Return as (1, forecast_steps) to make it consistent with other models
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return {
            'p': self.p,
            'd': self.d,
            'q': self.q,
            'target_col': self.target_col,
            'loss_functions': self.loss_functions,
            'primary_loss': self.primary_loss,
            'forecast_horizon': self.forecast_horizon,
            'is_fitted': self.is_fitted
        }
        
    def set_params(self, **params: Dict[str, Any]) -> 'ARIMAModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        # Track if model parameters that affect fitting are changed
        model_params_changed = False
        
        for key, value in params.items():
            if hasattr(self, key):
                # Check if this is a model parameter that requires refitting
                if key in ['p', 'd', 'q', 's'] and getattr(self, key) != value:
                    model_params_changed = True
                setattr(self, key, value)
            else:
                # Update config if parameter not found in instance attributes
                self.config[key] = value
        
        # If model parameters changed, reset the fitted model
        if model_params_changed and self.is_fitted:
            self.model_ = None
            self.is_fitted = False
            
        return self
        
    def save(self, path: str) -> None:
        """
        Save the ARIMA model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model. Call train() first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        model_state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'params': self.get_params(),
            'model': self.model_
        }
        
        try:
            # Save model state to file
            with open(path, 'wb') as f:
                pickle.dump(model_state, f)
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {path}: {str(e)}")
        
    def load(self, path: str) -> None:
        """
        Load the ARIMA model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        try:
            # Load model state from file
            with open(path, 'rb') as f:
                model_state = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {str(e)}")
            
        # Restore model state
        self.config = model_state['config']
        self.is_fitted = model_state['is_fitted']
        self.set_params(**model_state['params'])
        self.model_ = model_state['model']
        
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the ARIMA model's properties and performance.
        
        Returns:
            Dict[str, Any]: Dictionary containing model summary information
        """
        summary = {
            'model_type': 'ARIMA',
            'order': (self.p, self.d, self.q),
            'is_fitted': self.is_fitted,
            'forecast_horizon': self.forecast_horizon,
            'target_col': self.target_col,
            'loss_functions': self.loss_functions,
            'primary_loss': self.primary_loss
        }
        
        if self.is_fitted and self.model_ is not None:
            # Add fitted model information
            summary.update({
                'aic': getattr(self.model_, 'aic', None),
                'bic': getattr(self.model_, 'bic', None),
                'hqic': getattr(self.model_, 'hqic', None),
                'llf': getattr(self.model_, 'llf', None),  # Log-likelihood
                'nobs': getattr(self.model_, 'nobs', None)  # Number of observations
            })
            
        return summary 