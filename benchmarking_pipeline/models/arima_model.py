"""
ARIMA model implementation.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Dict, Any, Union, Tuple
import itertools
import pickle
import os
from .base_model import BaseModel


class ARIMAModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize ARIMA model with given configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
                - p: int, AR order
                - d: int, differencing order
                - q: int, MA order
                - target_col: str, name of target column
                - exog_cols: list of str, names of exogenous variables
                - loss_functions: List[str], list of loss function names to use
                - primary_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, config_file)
        self.p = self.config.get('p', 1)
        self.d = self.config.get('d', 1)
        self.q = self.config.get('q', 1)
        self.target_col = self.config.get('target_col', 'y')
        self.exog_cols = self.config.get('exog_cols', None)
        self.model = None
        
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'ARIMAModel':
        """
        Train the ARIMA model on given data.
        
        Args:
            X: Training features (exogenous variables)
            y: Target values (time series)
            
        Returns:
            self: The fitted model instance
        """
        # Convert inputs to appropriate format
        if isinstance(y, pd.Series):
            endog = y
        else:
            endog = pd.Series(y)
            
        if isinstance(X, pd.DataFrame):
            exog = X
        elif X is not None:
            exog = pd.DataFrame(X)
        else:
            exog = None
            
        # Fit ARIMA model
        self.model = ARIMA(endog, order=(self.p, self.d, self.q), exog=exog)
        self.model = self.model.fit()
        self.is_fitted = True
        
        return self
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained ARIMA model.
        
        Args:
            X: Input data for prediction (exogenous variables)
            
        Returns:
            np.ndarray: Model predictions with shape (n_samples, forecast_horizon)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train first.")
            
        # Convert input to appropriate format
        if isinstance(X, pd.DataFrame):
            exog = X
        elif X is not None:
            exog = pd.DataFrame(X)
        else:
            exog = None
            
        # Get predictions for each step in the forecast horizon
        predictions = np.zeros((len(X), self.forecast_horizon))
        for i in range(len(X)):
            # Get forecast for this point
            forecast = self.model.forecast(steps=self.forecast_horizon, exog=exog.iloc[i:i+1] if exog is not None else None)
            predictions[i] = forecast.values
            
        return predictions
        
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
            'exog_cols': self.exog_cols,
            'loss_functions': self.loss_functions,
            'primary_loss': self.primary_loss,
            'forecast_horizon': self.forecast_horizon
        }
        
    def set_params(self, **params: Dict[str, Any]) -> 'ARIMAModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
        
    def save(self, path: str) -> None:
        """
        Save the ARIMA model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        model_state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'params': self.get_params(),
            'model': self.model
        }
        
        # Save model state to file
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
    def load(self, path: str) -> None:
        """
        Load the ARIMA model from disk.
        
        Args:
            path: Path to load the model from
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        # Load model state from file
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
            
        # Restore model state
        self.config = model_state['config']
        self.is_fitted = model_state['is_fitted']
        self.set_params(**model_state['params'])
        self.model = model_state['model'] 