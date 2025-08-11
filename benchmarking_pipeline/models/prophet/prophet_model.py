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
        self.model = Prophet(**self.config.get('model_params', {}))
        self.is_fitted = False

    @staticmethod
    def ensure_series_with_datetimeindex(y, fallback_start='2000-01-01', freq='D'):
        if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
            return y
        return pd.Series(
            y.values if hasattr(y, 'values') else y,
            index=pd.date_range(fallback_start, periods=len(y), freq=freq)
        )

    def train(self, y_context, x_context=None, y_context_timestamps=None, **kwargs):
        if y_context_timestamps is None:
            raise ValueError("y_context_timestamps must be provided for Prophet training.")
        # Create a new Prophet model instance for each training
        self.model = Prophet(**self.config.get('model_params', {}))
        # Use the provided timestamps to create a DatetimeIndex for y_context
        y_context = pd.Series(
            y_context.values if hasattr(y_context, 'values') else y_context,
            index=y_context_timestamps
        )
        train_df = pd.DataFrame({'ds': y_context.index, 'y': y_context.values})
        if x_context is not None:
            if not isinstance(x_context, pd.DataFrame):
                x_context = pd.DataFrame(x_context, index=y_context.index)
            for col in x_context.columns:
                self.model.add_regressor(col)
            train_df = train_df.join(x_context)
        self.model.fit(train_df)
        self.is_fitted = True
        return self

    def predict(self, y_context, y_target=None, y_context_timestamps=None, y_target_timestamps=None, **kwargs):
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        if y_context is None or y_target is None:
            raise ValueError("y_context and y_target must both be provided.")
        
        # Use provided y_target_timestamps if available
        if y_target_timestamps is not None:
            future_index = pd.to_datetime(y_target_timestamps)
        else:
            if kwargs.get('start_date') is None:
                raise ValueError("Either y_target_timestamps or start_date must be provided.")
            # The first prediction should start after the last training data point
            # Assuming daily frequency for now
            future_index = pd.date_range(start=kwargs['start_date'], periods=len(y_target), freq='D')
        
        # Create future dataframe with the correct timestamps
        future_df = pd.DataFrame({'ds': future_index})
        
        # Make predictions
        forecast = self.model.predict(future_df)
        
        # Return the predicted values
        return forecast['yhat'].values

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
        self.model = Prophet(**self.config['model_params'])
        self.is_fitted = False
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