"""
Trainer class for model training and evaluation.
"""
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from .logger import Logger
from ..models.base_model import BaseModel


class Trainer: #update the trainer class to a folder for each model type
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Configuration dictionary
            config_file: Path to configuration file
        """
        self.config = config or {}
        if config_file:
            with open(config_file, 'r') as f:
                self.config.update(json.load(f))
        self.logger = Logger()
        
    def train(self, model: BaseModel, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> BaseModel:
        """
        Train the model on the provided data.
        
        Args:
            model: Model instance to train
            X: Training features
            y: Target values
            
        Returns:
            Trained model instance
        """
        try:
            # Train the model
            self.logger.info(f"Training {model.__class__.__name__}...")
            model.train(X, y)
            self.logger.info("Training completed successfully")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
            
    def evaluate(self, model: BaseModel, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate the model on the provided data.
        
        Args:
            model: Trained model instance
            X: Evaluation features
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            self.logger.info(f"Evaluating {model.__class__.__name__}...")
            metrics = model.evaluate(X, y)
            self.logger.info(f"Evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise
            
    def train_and_evaluate(self, model: BaseModel, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple[BaseModel, Dict[str, float]]:
        """
        Train and evaluate the model on the provided data.
        
        Args:
            model: Model instance to train
            X: Training features
            y: Target values (used for both training and evaluation)
            
        Returns:
            Tuple of (trained model, evaluation metrics)
        """
        try:
            # Train the model
            trained_model = self.train(model, X, y)
            
            # Evaluate the model
            metrics = self.evaluate(trained_model, X, y)
            
            return trained_model, metrics
            
        except Exception as e:
            self.logger.error(f"Error during train and evaluate: {str(e)}")
            raise