"""
Base model class that defines the interface for all models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union, Tuple, List
import pandas as pd
import json
import os
import pickle
from ..pipeline.evaluator import Evaluator


class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the base model.
        
        Args:
            config: Configuration dictionary containing model parameters
                - loss_functions: List[str], list of loss function names to use
                - primary_loss: str, primary loss function for training (defaults to first in loss_functions)
                - forecast_horizon: int, number of steps to forecast ahead
            config_file: Path to a JSON configuration file
        """
        if config_file is not None:
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        self.config = config or {}
        self.loss_functions = self.config.get('loss_functions', ['mse'])
        self.primary_loss = self.config.get('primary_loss', self.loss_functions[0])
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        self.is_fitted = False
        self.evaluator = Evaluator(config=self.config)
        
    @abstractmethod
    def train(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'BaseModel':
        """
        Train the model on given data.
        
        Args:
            X: Training features
            y: Target values
            
        Returns:
            self: The fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data for prediction
            
        Returns:
            np.ndarray: Model predictions with shape (n_samples, forecast_horizon)
        """
        pass
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: str = None) -> Dict[str, float]:
        """
        Compute all loss metrics between true and predicted values using the Evaluator class.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            loss_function: Name of the loss function to use (defaults to primary_loss)
            
        Returns:
            Dict[str, float]: Dictionary of computed loss metrics
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
            
        # Convert inputs to DataFrame format required by Evaluator
        eval_data = pd.DataFrame({
            self.evaluator.target_col_name: y_true,
            self.evaluator.pred_col_name: y_pred
        })
        
        # Use evaluator to compute all metrics
        return self.evaluator.evaluate(self, eval_data)
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Train and evaluate model on given data.
        
        Args:
            X: Evaluation features
            y: True target values
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        # Get predictions
        predictions = self.predict(X)
        loss = self.compute_loss(y, predictions)
        return loss
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        pass
    
    @abstractmethod
    def set_params(self, **params: Dict[str, Any]) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        pass
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
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
            'params': self.get_params()
        }
        
        # Save model state to file
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
    def load(self, path: str) -> None:
        """
        Load the model from disk.
        
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
        
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model's properties and performance.
        
        Returns:
            Dict[str, Any]: Dictionary containing model summary information
        """
        return {
            'model_type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'parameters': self.get_params()
        } 