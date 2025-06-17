"""
Exponential Smoothing model implementation
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
        # The model will be instantiated with data during the .train() call, as is standard for statsmodels
        self.model = None

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'ExponentialSmoothingModel':
        """
        Train the Exponential Smoothing model on given data.
        
        Args:
            X: Training features (ignored by this univariate model, but required for API consistency).
            y: Target time series values
        
        Returns:
            self: The fitted model instance.
        """
        # statsmodels works best if y is a Pandas Series
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        # Get model parameters from config
        model_params = self.config.get('model_params', {})
        
        print("Initializing and fitting ExponentialSmoothing model...")
        
        # Instantiate the model with data and parameters
        ets_model = ExponentialSmoothing(
            endog=y,
            trend=model_params.get('trend'),
            damped_trend=model_params.get('damped_trend', False),
            seasonal=model_params.get('seasonal'),
            seasonal_periods=model_params.get('seasonal_periods')
        )
        
        # Fit the model and store the results object
        self.model = ets_model.fit()
        self.is_fitted = True
        
        print("Training complete.")
        return self
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained Exponential Smoothing model.
        
        Args:
            X: Input data for prediction. The number of rows in X determines the
               number of steps to forecast. The content of X is ignored.
            
        Returns:
            np.ndarray: Model predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # The number of steps to forecast is determined by the length of the input X
        n_steps = len(X)
        
        # The .forecast() method of a fitted statsmodels object takes the number of steps
        predictions = self.model.forecast(steps=n_steps)
        
        return predictions.values

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
        self.model = None
        
        return self

    def save(self, path: str) -> None:
        """
        Save the trained statsmodels ETS model to disk using pickle.
        
        Args:
            path: Path to save the model.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # The fitted results object from statsmodels can be pickled
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, path: str) -> 'ExponentialSmoothingModel':
        """
        Load a trained statsmodels ETS model from disk.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self