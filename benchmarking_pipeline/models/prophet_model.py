"""
Prophet model implementation.

TO BE CHANGED: This model needs to be updated to match the new interface with y_context, x_context, y_target, x_target parameters.
"""

import os
import json
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from benchmarking_pipeline.models.base_model import BaseModel

class ProphetModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Prophet model with a given configuration.
        
        Args:
            config: Configuration dictionary for Prophet parameters.
                    e.g., {'model_params': {'seasonality_mode': 'multiplicative'}}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        self._build_model()
        
    def _build_model(self):
        """
        Build the Prophet model instance from the configuration.
        """
        model_params = self.config.get('model_params', {})
        self.model = Prophet(**model_params)
        self.is_fitted = False

    def train(self, y_context: Union[pd.Series, np.ndarray] = None, y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.DataFrame, np.ndarray] = None, x_target: Union[pd.DataFrame, np.ndarray] = None) -> 'ProphetModel':
        """
        Train the Prophet model on given data.
        
        Args:
            y_context: Past target values (time series) as a Pandas Series with a DatetimeIndex.
            y_target: Future target values (optional, for validation)
            x_context: Past exogenous variables (optional, DataFrame with same index as y_context)
            x_target: Future exogenous variables (optional, DataFrame with same index as y_target)
        
        Returns:
            self: The fitted model instance.
        """
        if self.model is None:
            self._build_model()

        if not isinstance(y_context, pd.Series) or not isinstance(y_context.index, pd.DatetimeIndex):
            raise TypeError("For Prophet, y_context must be a Pandas Series with a DatetimeIndex.")
        
        train_df = pd.DataFrame({'ds': y_context.index, 'y': y_context.values})
        
        # Handle exogenous regressors
        if x_context is not None:
            if not isinstance(x_context, pd.DataFrame):
                x_context = pd.DataFrame(x_context, index=y_context.index)
                x_context.columns = [f"exog_{i}" for i in range(x_context.shape[1])]
            for col in x_context.columns:
                self.model.add_regressor(col)
            train_df = train_df.join(x_context)
        
        print(f"Fitting Prophet model...")
        self.model.fit(train_df)
        self.is_fitted = True
        print("Training complete.")
        return self

    def predict(self, y_context: Union[pd.Series, np.ndarray] = None, y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.DataFrame, np.ndarray] = None, x_target: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Make predictions using the trained Prophet model.
        
        Args:
            y_context: Past target values (time series) as a Pandas Series with a DatetimeIndex.
            y_target: Future target values (used to determine forecast length and future dates)
            x_context: Past exogenous variables (optional, ignored for prediction)
            x_target: Future exogenous variables (optional, DataFrame with same index as y_target)
        
        Returns:
            np.ndarray: Model's point forecast ('yhat') with shape (1, forecast_steps)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length and future dates.")
        if not isinstance(y_target, pd.Series) or not isinstance(y_target.index, pd.DatetimeIndex):
            raise TypeError("For Prophet, y_target must be a Pandas Series with a DatetimeIndex.")
        
        # Build the future dataframe
        future_df = pd.DataFrame({'ds': y_target.index})
        
        # Add exogenous regressors if available
        if x_target is not None:
            if not isinstance(x_target, pd.DataFrame):
                x_target = pd.DataFrame(x_target, index=y_target.index)
                x_target.columns = [f"exog_{i}" for i in range(x_target.shape[1])]
            future_df = future_df.join(x_target)
        
        forecast = self.model.predict(future_df)
        yhat = forecast['yhat'].values.reshape(1, -1)
        return yhat

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the configuration.
        """
        # Prophet model attributes are set at initialization, so we return those.
        return self.config.get('model_params', {})

    def set_params(self, **params: Dict[str, Any]) -> 'ProphetModel':
        """
        Set model parameters. This will rebuild the Prophet model instance with the new parameters.
        """
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        self.config['model_params'].update(params)
        
        # Re-build the model with the new parameters
        self._build_model()
        return self

    def save(self, path: str) -> None:
        """
        Save the trained Prophet model to disk using its native JSON serialization.
        
        Args:
            path: Path to save the model file. The '.json' extension is recommended.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(path, 'w') as f:
            json.dump(model_to_json(self.model), f)
            
    def load(self, path: str) -> 'ProphetModel':
        """
        Load a trained Prophet model from a JSON file.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'r') as f:
            self.model = model_from_json(json.load(f))
        self.is_fitted = True
        return self