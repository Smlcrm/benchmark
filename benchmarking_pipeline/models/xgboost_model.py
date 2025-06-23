"""
XGBoost model implementation.

TO BE CHANGED: This model needs to be updated to match the new interface with y_context, x_context, y_target, x_target parameters.
"""
import os
import pickle
from typing import Dict, Any, Union, List
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from benchmarking_pipeline.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize XGBoost model with a given configuration.
        
        Args:
            config: Configuration dictionary for XGBoostRegressor parameters.
                    e.g., {'model_params': {'n_estimators': 100, 'learning_rate': 0.1}}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        self._build_model()
        
    def _build_model(self):
        """
        Build the XGBRegressor model instance from the configuration.
        """
        # Get hyperparameters from config.
        model_params = self.config.get('model_params', {})
        
        # Ensure random_state for reproducibility if not provided
        if 'random_state' not in model_params:
            model_params['random_state'] = 42
            
        self.model = XGBRegressor(**model_params)
        self.is_fitted = False

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'XGBoostModel':
        """
        Train the XGBoost model on given data.
        
        Args:
            X: Training features of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
        
        Returns:
            self: The fitted model instance.
        """
        if self.model is None:
            self._build_model()
            
        print(f"Training XGBoost model with {X.shape[0]} samples...")
        self.model.fit(X, y)
        self.is_fitted = True
        print("Training complete.")
        return self
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained XGBoost model.
        
        Args:
            X: Input data for prediction, shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Model predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the underlying xgboost model.
        """
        if self.model:
            return self.model.get_params()
        return self.config.get('model_params', {})

    def set_params(self, **params: Dict[str, Any]) -> 'XGBoostModel':
        """
        Set model parameters. This will rebuild the model instance.
        """
        if self.model:
            self.model.set_params(**params)
        
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        self.config['model_params'].update(params)
        return self

    def save(self, path: str) -> None:
        """
        Save the trained xgboost model to disk using pickle.
        
        Args:
            path: Path to save the model.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, path: str) -> 'XGBoostModel':
        """
        Load a trained xgboost model from disk.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self