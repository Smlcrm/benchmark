"""
Base model class that defines the interface for all models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Union, Tuple, List, Optional
import pandas as pd
import json
import os
import pickle
from benchmarking_pipeline.pipeline.evaluator import Evaluator


class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the base model.
        
        Args:
            config: Configuration dictionary containing model parameters
                - loss_functions: List[str], list of loss function names to use
                - primary_loss: str, primary loss function for training (defaults to first in loss_functions)
                - forecast_horizon: int, number of steps to forecast ahead
                - dataset: dict containing dataset configuration including target_cols
            config_file: Path to a JSON configuration file
        """
        if config_file is not None:
            if not os.path.exists(config_file):
                raise FileNotFoundError(f"Config file not found: {config_file}")
            with open(config_file, 'r') as f:
                config = json.load(f)
        
        self.config = config or {}
        self.loss_functions = self.config.get('loss_functions', ['mae'])
        self.primary_loss = self.config.get('primary_loss', self.loss_functions[0])
        
        # Extract target_cols and forecast_horizon from dataset configuration
        dataset_cfg = self.config.get('dataset', {})
        self.target_cols = dataset_cfg.get('target_cols')
        if not self.target_cols:
            raise ValueError("target_cols must be defined in dataset configuration")
        
        # Get forecast_horizon from model config (for hyperparameter tuning) or fall back to dataset config
        self.forecast_horizon = self.config.get('forecast_horizon', dataset_cfg.get('forecast_horizon', 1))
        if not self.forecast_horizon:
            raise ValueError("forecast_horizon must be defined in model configuration or dataset configuration")
        
        # If forecast_horizon is a list (from hyperparameter grid), take the first value for initialization
        if isinstance(self.forecast_horizon, list):
            self.forecast_horizon = self.forecast_horizon[0]
            print(f"[INFO] Using forecast_horizon: {self.forecast_horizon} from hyperparameter grid")
        
        self.is_fitted = False
        self.evaluator = Evaluator(config=self.config)
        # For logging last eval
        self._last_y_true = None
        self._last_y_pred = None
        
    @abstractmethod
    def train(self, 
              y_context: Optional[Union[pd.Series, np.ndarray]], 
              x_context: Optional[Union[pd.Series, np.ndarray]] = None, 
              y_target: Optional[Union[pd.Series, np.ndarray]] = None, 
              x_target: Optional[Union[pd.Series, np.ndarray]] = None,
              y_start_date: Optional[str] = None,
              x_start_date: Optional[str] = None
    ) -> 'BaseModel':
        """
        Train the model on given data.
        
        Args:
            y_context: Past target values - training data during tuning time, training + validation data during testing time
            x_context: Past exogenous variables - used during tuning and testing time
            y_target: Future target values - validation data during tuning time, None during testing time (avoid data leakage)
            x_target: Future exogenous variables - if provided, can be used with x_context for training (e.g., in ARIMA models)
                     or with y_target for validation during tuning time
            y_start_date: The start date timestamp for y_context and y_target in string form
            x_start_date: The start date timestamp for x_context and x_target in string form
            
        Returns:
            self: The fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        x_context: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        x_target: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        forecast_horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            y_context: Recent/past target values (for sequence models, optional for ARIMA)
            x_context: Recent/past exogenous variables (for sequence models, optional for ARIMA)
            x_target: Future exogenous variables for the forecast horizon (required if model uses exogenous variables)
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            
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
            
        # Handle shape mismatches
        if y_pred.ndim == 2 and y_true.ndim == 1:
            # If predictions are 2D and true values are 1D, flatten predictions
            if y_pred.shape[0] == 1:
                # Single prediction row, flatten it
                y_pred = y_pred.flatten()
            elif y_pred.shape[1] == 1:
                # Single prediction column, flatten it
                y_pred = y_pred.flatten()
            else:
                # Multiple predictions, take the first row
                y_pred = y_pred[0]
        
        # Ensure both arrays have the same length
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]
        # print(f"y_pred: {y_pred}")
        # print(f"y_true: {y_true}")
        # Store for TensorBoard logging
        self._last_y_true = y_true
        self._last_y_pred = y_pred
        
        # Convert inputs to DataFrame format required by Evaluator
        
        
        # Use evaluator to compute all metrics
        return self.evaluator.evaluate(y_pred, y_true)
    
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

    def get_last_eval_true_pred(self):
        """
        Return the last y_true and y_pred used in compute_loss for TensorBoard logging.
        """
        return self._last_y_true, self._last_y_pred 