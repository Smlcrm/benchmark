"""
Multivariate model wrapper that handles multiple targets.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from .base_model import BaseModel
from .evaluator import Evaluator


class MultivariateWrapper(BaseModel):
    """
    Wrapper that creates separate models for each target variable.
    This allows any univariate model to work with multivariate datasets.
    """
    
    def __init__(self, model_class, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the multivariate wrapper.
        
        Args:
            model_class: The class of the univariate model to use for each target
            config: Configuration dictionary
            config_file: Path to config file
        """
        super().__init__(config, config_file)
        self.model_class = model_class
        self.models = {}  # Dictionary to store models for each target
        self.target_columns = []  # List of target column names
        self.is_fitted = False
        
    def train(self, 
              y_context: Optional[Union[pd.Series, np.ndarray, pd.DataFrame]] = None, 
              x_context: Optional[Union[pd.Series, np.ndarray]] = None, 
              y_target: Optional[Union[pd.Series, np.ndarray, pd.DataFrame]] = None, 
              x_target: Optional[Union[pd.Series, np.ndarray]] = None,
              y_start_date: Optional[str] = None,
              x_start_date: Optional[str] = None
    ) -> 'MultivariateWrapper':
        """
        Train separate models for each target variable.
        
        Args:
            y_context: Training data - can be DataFrame (multivariate) or Series (univariate)
            y_target: Target data for validation - can be DataFrame (multivariate) or Series (univariate)
            x_context: Exogenous variables for training
            x_target: Exogenous variables for validation
            y_start_date: Start date for y_context and y_target
            x_start_date: Start date for x_context and x_target
            
        Returns:
            self: The fitted wrapper instance
        """
        # Determine if we have multivariate data
        if isinstance(y_context, pd.DataFrame):
            # Multivariate case
            self.target_columns = list(y_context.columns)
            print(f"[DEBUG] Training multivariate models for targets: {self.target_columns}")
            
            # Create and train a model for each target
            for target_col in self.target_columns:
                print(f"[DEBUG] Training model for target: {target_col}")
                
                # Create model instance for this target
                model_config = self.config.copy()
                model_config['target_col'] = target_col
                model = self.model_class(model_config)
                
                # Get data for this specific target
                y_context_single = y_context[target_col]
                y_target_single = y_target[target_col] if y_target is not None else None
                
                # Train the model
                model.train(
                    y_context=y_context_single,
                    x_context=x_context,
                    y_target=y_target_single,
                    x_target=x_target,
                    y_start_date=y_start_date,
                    x_start_date=x_start_date
                )
                
                # Store the trained model
                self.models[target_col] = model
                
        else:
            # Univariate case - just create one model
            self.target_columns = ['y']
            model = self.model_class(self.config)
            model.train(
                y_context=y_context,
                x_context=x_context,
                y_target=y_target,
                x_target=x_target,
                y_start_date=y_start_date,
                x_start_date=x_start_date
            )
            self.models['y'] = model
            
        self.is_fitted = True
        return self
    
    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray, pd.DataFrame]] = None,
        x_context: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        x_target: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        forecast_horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions for all target variables.
        
        Args:
            y_context: Recent/past target values
            x_context: Recent/past exogenous variables
            x_target: Future exogenous variables
            forecast_horizon: Number of steps to forecast
            
        Returns:
            np.ndarray: Predictions with shape (n_samples, n_targets)
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
            
        predictions = []
        
        for target_col in self.target_columns:
            model = self.models[target_col]
            
            # Get context data for this target
            if isinstance(y_context, pd.DataFrame):
                y_context_single = y_context[target_col]
            else:
                y_context_single = y_context
                
            # Make prediction for this target
            pred = model.predict(
                y_context=y_context_single,
                x_context=x_context,
                x_target=x_target,
                forecast_horizon=forecast_horizon
            )
            
            predictions.append(pred)
            
        # Stack predictions into a 2D array (n_samples, n_targets)
        return np.column_stack(predictions)
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: str = None) -> Dict[str, float]:
        """
        Compute loss metrics averaged across all targets.
        
        Args:
            y_true: True target values with shape (n_samples, n_targets)
            y_pred: Predicted values with shape (n_samples, n_targets)
            loss_function: Name of the loss function to use
            
        Returns:
            Dict[str, float]: Dictionary of averaged loss metrics
        """
        # Store for TensorBoard logging
        self._last_y_true = y_true
        self._last_y_pred = y_pred
        
        # Convert to numpy arrays if needed
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values
            
        # Handle univariate case
        if y_true.ndim == 1:
            return super().compute_loss(y_true, y_pred, loss_function)
            
        # Multivariate case - compute metrics for each target and average
        all_metrics = []
        
        for i in range(y_true.shape[1]):
            target_true = y_true[:, i]
            target_pred = y_pred[:, i]
            
            # Compute metrics for this target
            target_metrics = self.evaluator.evaluate(target_pred, target_true)
            all_metrics.append(target_metrics)
            
        # Average metrics across all targets
        averaged_metrics = {}
        for metric_name in all_metrics[0].keys():
            values = [metrics[metric_name] for metrics in all_metrics]
            averaged_metrics[metric_name] = np.mean(values)
            
        return averaged_metrics
    
    def get_params(self) -> Dict[str, Any]:
        """Get parameters from the first model."""
        if self.models:
            first_model = next(iter(self.models.values()))
            return first_model.get_params()
        return {}
    
    def set_params(self, **params: Dict[str, Any]) -> 'MultivariateWrapper':
        """Set parameters for all models."""
        for model in self.models.values():
            model.set_params(**params)
        return self
    
    def get_last_eval_true_pred(self):
        """Return the last evaluation data for plotting."""
        return self._last_y_true, self._last_y_pred 