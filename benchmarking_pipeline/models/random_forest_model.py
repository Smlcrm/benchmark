"""
Random Forest model implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, List
import json
import os
import pickle
from benchmarking_pipeline.models.base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Random Forest model with given configuration.
        
        Args:
            config: Configuration dictionary for RandomForestRegressor parameters.
                    e.g., {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        self._build_model()
        
    def _build_model(self):
        """
        Build the RandomForestRegressor model instance from the configuration.
        """
        # Get hyperparameters from config, with sensible defaults from sklearn
        model_params = self.config.get('model_params', {})
        
        if 'random_state' not in model_params:
            model_params['random_state'] = 42

        if 'n_jobs' not in model_params:
            model_params['n_jobs'] = -1
            
        self.model = RandomForestRegressor(**model_params)
        self.is_fitted = False

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'RandomForestModel':
        """
        Train the Random Forest model on given data.
        
        Args:
            X: Training features of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
        
        Returns:
            self: The fitted model instance.
        """
        if self.model is None:
            self._build_model()
            
        self.model.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained Random Forest model.
        
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
        Get the current model parameters.
        """
        if self.model:
            return self.model.get_params()
        return self.config.get('model_params', {})

    def set_params(self, **params: Dict[str, Any]) -> 'RandomForestModel':
        """
        Set model parameters. This will rebuild the model.
        """
        if self.model:
            self.model.set_params(**params)
        # Update internal config as well
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        self.config['model_params'].update(params)
        return self

    def save(self, path: str) -> None:
        """
        Save the trained scikit-learn model to disk using pickle.
        
        Args:
            path: Path to save the model.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save an unfitted model")
        
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, path: str) -> 'RandomForestModel':
        """
        Load a trained scikit-learn model from disk.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self