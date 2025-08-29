"""
Prophet model implementation.
TODO-COULD work multivariate
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
        # Ensure 1D series indexed by provided timestamps
        y_vals = y_context.values if hasattr(y_context, 'values') else y_context
        if isinstance(y_vals, np.ndarray) and y_vals.ndim > 1:
            y_vals = y_vals.flatten()
        
        # Convert timestamps to datetime if they're not already
        if not isinstance(y_context_timestamps, pd.DatetimeIndex):
            # Handle different timestamp formats
            if isinstance(y_context_timestamps[0], (int, np.integer)):
                # Convert from nanoseconds to datetime
                if y_context_timestamps[0] > 1e18:  # Likely nanoseconds
                    timestamps = pd.to_datetime(y_context_timestamps, unit='ns')
                elif y_context_timestamps[0] > 1e15:  # Likely microseconds
                    timestamps = pd.to_datetime(y_context_timestamps, unit='us')
                elif y_context_timestamps[0] > 1e12:  # Likely milliseconds
                    timestamps = pd.to_datetime(y_context_timestamps, unit='ms')
                else:  # Likely seconds
                    timestamps = pd.to_datetime(y_context_timestamps, unit='s')
            else:
                timestamps = pd.to_datetime(y_context_timestamps)
        else:
            timestamps = y_context_timestamps
            
        y_context = pd.Series(y_vals, index=timestamps)
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
        
        # Prophet doesn't need y_target for prediction, it predicts based on the trained model
        # We only need to know how many steps to predict
        
        # Use provided y_target_timestamps if available
        if y_target_timestamps is not None:
            future_index = pd.to_datetime(y_target_timestamps)
        elif y_target is not None:
            # If y_target is provided, use its length to determine forecast horizon
            # If no timestamps provided, we need to infer them from the context
            if y_context_timestamps is not None and len(y_context_timestamps) > 0:
                last_timestamp = pd.to_datetime(y_context_timestamps[-1])
                # Infer frequency from the training data
                if len(y_context_timestamps) > 1:
                    freq = pd.infer_freq(y_context_timestamps)
                    if freq is None:
                        # Fallback to 30-minute frequency for this dataset
                        freq = '30min'
                else:
                    freq = '30min'
                
                # Create future timestamps starting from the next point after the last training data
                future_index = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=30), 
                                          periods=len(y_target), freq=freq)
            else:
                # This should never happen in normal operation - raise error to catch the issue
                raise ValueError("Prophet predict called without y_context_timestamps. This indicates a bug in the calling code.")
        else:
            # Default forecast horizon if neither y_target nor y_target_timestamps provided
            forecast_horizon = 300  # Default to 300 steps
            if y_context_timestamps is not None and len(y_context_timestamps) > 0:
                last_timestamp = pd.to_datetime(y_context_timestamps[-1])
                freq = '30min'
                future_index = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=30), 
                                          periods=forecast_horizon, freq=freq)
            else:
                # This should never happen in normal operation - raise error to catch the issue
                raise ValueError("Prophet predict called without y_context_timestamps. This indicates a bug in the calling code.")
        
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
        # Return Prophet model parameters
        return self.config.get('model_params', {})

    def set_params(self, **params: Dict[str, Any]) -> 'ProphetModel':
        """
        Set model parameters. This will rebuild the Prophet model instance with the new parameters.
        """
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        
        # Filter out non-Prophet parameters
        valid_prophet_params = {}
        
        for key, value in params.items():
            if key in ['forecast_horizon', 'dataset']:
                # Skip these parameters - they're not for Prophet constructor
                continue
            elif key in ['seasonality_mode', 'changepoint_prior_scale', 'seasonality_prior_scale', 
                        'yearly_seasonality', 'weekly_seasonality', 'daily_seasonality',
                        'holidays', 'holidays_prior_scale', 'changepoint_range']:
                # These are valid Prophet parameters
                valid_prophet_params[key] = value
            else:
                # Skip unknown parameters instead of raising error to allow for future Prophet versions
                print(f"[WARNING] Skipping unknown Prophet parameter: {key}")
                continue
        
        self.config['model_params'].update(valid_prophet_params)
        
        # Re-build the model with the valid parameters only
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