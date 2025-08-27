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


class FoundationModel(ABC):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the base model for foundation models.
        
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
        self.loss_functions = self.config.get('loss_functions', ['mae'])
        self.primary_loss = self.config.get('primary_loss', self.loss_functions[0])
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        self.target_cols = self.config.get('target_cols')
        if not self.target_cols:
            raise ValueError("target_cols must be defined in config")
        self.is_fitted = False
        self.evaluator = Evaluator(config=self.config)
        # For logging last eval
        self._last_y_true = None
        self._last_y_pred = None
        
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