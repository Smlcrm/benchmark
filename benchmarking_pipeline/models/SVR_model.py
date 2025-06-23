"""
SVR model implementation.

TO BE CHANGED: This model needs to be updated to match the new interface with y_context, x_context, y_target, x_target parameters.
"""

import os
import pickle
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from benchmarking_pipeline.models.base_model import BaseModel


class SVRModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Support Vector Regression (SVR) model with a given configuration.
        
        Args:
            config: Configuration dictionary for SVR parameters.
                    e.g., {'model_params': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}}
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        self.scaler = StandardScaler() # SVR is sensitive to feature scaling
        self._build_model()
        
    def _build_model(self):
        """
        Build the SVR model instance from the configuration.
        """
        model_params = self.config.get('model_params', {})
        self.model = SVR(**model_params)
        self.is_fitted = False

    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'SVRModel':
        """
        Train the SVR model on given data. This method will also fit a scaler 
        to the training data, which will be used for predictions.
        
        Args:
            X: Training features of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
        
        Returns:
            self: The fitted model instance.
        """
        if self.model is None:
            self._build_model()
        
        # Scale the data.
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
        
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained SVR model.
        This method will use the scaler fitted during training to transform the new data.
        
        Args:
            X: Input data for prediction, shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Model predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Scale the new data using the scaler that was fitted on the training data
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the underlying scikit-learn model.
        """
        if self.model:
            return self.model.get_params()
        # Return config params if model is not yet instantiated
        return self.config.get('model_params', {})

    def set_params(self, **params: Dict[str, Any]) -> 'SVRModel':
        """
        Set model parameters. This will rebuild the model instance with the new parameters.
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
        Save the trained SVR model and its scaler to disk using pickle.
        
        Args:
            path: Path to save the model objects.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        # We need to save both the model and the scaler for consistent predictions
        model_and_scaler = {'model': self.model, 'scaler': self.scaler}
        
        with open(path, 'wb') as f:
            pickle.dump(model_and_scaler, f)
            
    def load(self, path: str) -> 'SVRModel':
        """
        Load a trained SVR model and its scaler from disk.
        
        Args:
            path: Path to load the model objects from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            model_and_scaler = pickle.load(f)
            self.model = model_and_scaler['model']
            self.scaler = model_and_scaler['scaler']
            
        self.is_fitted = True
        return self