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

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'ProphetModel':
        """
        Train the Prophet model on given data.
        
        Args:
            X: Training features (exogenous variables). Must have the same index as y.
            y: Target time series values. Must be a pd.Series with a DatetimeIndex.
        
        Returns:
            self: The fitted model instance.
        """
        if self.model is None:
            self._build_model()

        if not isinstance(y, pd.Series) or not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("For Prophet, `y` must be a Pandas Series with a DatetimeIndex.")
            
        # Prophet requires a DataFrame with 'ds' and 'y' columns
        train_df = pd.DataFrame({'ds': y.index, 'y': y.values})
        
        # Handle exogenous regressors
        exog_cols = []
        if X is not None:
            if not isinstance(X, pd.DataFrame):
                # Assume columns are named exog_0, exog_1, etc.
                X = pd.DataFrame(X, index=y.index)
                X.columns = [f"exog_{i}" for i in range(X.shape[1])]

            # Add regressors to the Prophet model instance
            for col in X.columns:
                self.model.add_regressor(col)
            
            # Join regressors with the main DataFrame
            train_df = train_df.join(X)

        print(f"Fitting Prophet model...")
        self.model.fit(train_df)
        self.is_fitted = True
        print("Training complete.")
        return self
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained Prophet model.
        
        Args:
            X: Input data for prediction. Must be a DataFrame with a DatetimeIndex
               and columns for any exogenous regressors used during training.
            
        Returns:
            np.ndarray: Model's point forecast ('yhat').
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        if not isinstance(X, pd.DataFrame) or not isinstance(X.index, pd.DatetimeIndex):
            raise TypeError("For Prophet prediction, `X` must be a Pandas DataFrame with a DatetimeIndex.")

        # Prophet's predict method needs a "future" dataframe with a 'ds' column and columns for any regressors.
        future_df = X.copy()
        future_df['ds'] = future_df.index
        
        forecast = self.model.predict(future_df)
        
        # Return the point forecast 'yhat' as a numpy array
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