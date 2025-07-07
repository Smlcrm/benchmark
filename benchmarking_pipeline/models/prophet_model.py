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

    def train(self, y_context, x_context=None, y_start_date=None):
        if y_start_date is None:
            raise ValueError("y_start_date must be provided for Prophet training.")
        # Always create a DatetimeIndex for y_context
        y_context = pd.Series(
            y_context.values if hasattr(y_context, 'values') else y_context,
            index=pd.date_range(start=y_start_date, periods=len(y_context), freq='D')
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

    def predict(self, y_context, y_target, x_context=None, x_target=None, start_date=None):
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        if y_context is None or y_target is None or start_date is None:
            raise ValueError("y_context, y_target, and start_date must all be provided.")
        # The first prediction date is start_date + len(y_context) days
        first_pred_date = pd.to_datetime(start_date) + pd.Timedelta(days=len(y_context))
        future_index = pd.date_range(start=first_pred_date, periods=len(y_target), freq='D')
        future_df = pd.DataFrame({'ds': future_index})
        if x_target is not None:
            if not isinstance(x_target, pd.DataFrame):
                x_target = pd.DataFrame(x_target, index=future_index)
            future_df = future_df.join(x_target)
        forecast = self.model.predict(future_df)
        return forecast['yhat'].values.reshape(1, -1)

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