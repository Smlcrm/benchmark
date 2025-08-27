"""
Multivariate ARIMA model implementation.

This model extends the univariate ARIMA to handle multiple target variables simultaneously.
Uses Vector Autoregression (VAR) which is the multivariate extension of ARIMA models.
Design choices (similar to multivariate LSTM):
- Uses VAR from statsmodels for multivariate time series modeling
- Handles multiple targets simultaneously in one model
- Predicts all targets in a single forward pass
- Supports differencing for non-stationary series
"""
import os
import numpy as np
import math
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any, Union, Tuple, Optional
import pickle
from benchmarking_pipeline.models.base_model import BaseModel
import warnings
warnings.filterwarnings('ignore')


class MultivariateARIMAModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Multivariate ARIMA model with given configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
                - p: int, VAR order (autoregressive order)
                - d: int, differencing order for stationarity
                - maxlags: int, maximum number of lags to consider
                - target_cols: list of str, names of target columns (for multivariate)
                - loss_functions: List[str], list of loss function names to use
                - primary_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, config_file)
        self.p = int(self.config.get('p', 1))
        self.d = int(self.config.get('d', 1))
        self.maxlags = int(self.config.get('maxlags', 10))
        # target_cols is inherited from parent class (BaseModel/FoundationModel)
        self.forecast_horizon = int(self.config.get('forecast_horizon', 1))
        self.model = None
        self.fitted_model = None
        # n_targets will be calculated when needed: len(self.target_cols)
        self.differenced_data = None
        self.original_data = None
        self.is_fitted = False
        
    def _check_stationarity(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Check stationarity for each time series using Augmented Dickey-Fuller test.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            Dict with stationarity status for each column
        """
        stationarity = {}
        for col in data.columns:
            result = adfuller(data[col].dropna())
            # p-value < 0.05 indicates stationarity
            stationarity[col] = result[1] < 0.05
        return stationarity
        
    def _difference_data(self, data: pd.DataFrame, d: int = None) -> pd.DataFrame:
        """
        Apply differencing to make the series stationary.
        
        Args:
            data: Input DataFrame
            d: Order of differencing (if None, uses self.d)
            
        Returns:
            Differenced DataFrame
        """
        if d is None:
            d = self.d
            
        differenced = data.copy()
        for _ in range(d):
            differenced = differenced.diff().dropna()
        return differenced
        
    def _integrate_data(self, differenced_data: pd.DataFrame, original_data: pd.DataFrame, d: int = None) -> pd.DataFrame:
        """
        Reverse differencing to get back to original scale.
        
        Args:
            differenced_data: Differenced predictions
            original_data: Original data for integration
            d: Order of differencing (if None, uses self.d)
            
        Returns:
            Integrated DataFrame
        """
        if d is None:
            d = self.d
            
        integrated = differenced_data.copy()
        
        # Get the last d values from original data for each target
        for _ in range(d):
            # Add back the differences
            for col in integrated.columns:
                if col in original_data.columns:
                    last_val = original_data[col].iloc[-1]
                    integrated[col] = integrated[col].cumsum() + last_val
        
        return integrated
        
    def train(self, 
              y_context: Union[pd.Series, np.ndarray, pd.DataFrame], 
              y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None, 
              x_context: Union[pd.Series, np.ndarray] = None, 
              x_target: Union[pd.Series, np.ndarray] = None, 
              y_start_date: pd.Timestamp = None, 
              x_start_date: pd.Timestamp = None, 
              **kwargs
    ) -> 'MultivariateARIMAModel':
        """
        Train the Multivariate ARIMA model on given data.
        
        TECHNIQUE: Vector Autoregression (VAR) for Multiple Time Series
        - Extends ARIMA to handle multiple interdependent time series
        - Each variable depends on its own lags and lags of other variables
        - Captures cross-dependencies between multiple targets
        - Applies differencing if needed to achieve stationarity
        - Uses Maximum Likelihood Estimation for parameter fitting
        
        Args:
            y_context: Past target values (DataFrame for multivariate)
            y_target: Future target values (optional, for validation)
            x_context: Past exogenous variables (optional, ignored for now)
            x_target: Future exogenous variables (optional, ignored for now)
            y_start_date: The start date timestamp for y_context and y_target
            x_start_date: The start date timestamp for x_context and x_target
            **kwargs: Additional keyword arguments

        Returns:
            self: The fitted model instance
        """
        # Convert input to DataFrame
        if isinstance(y_context, pd.DataFrame):
            # Multivariate case - use all columns
            data = y_context.copy()
            # Update target_cols if not already set
            # target_cols should already be set from config, validate consistency
            if self.target_cols and len(y_context.columns) > 0:
                expected_cols = set(self.target_cols)
                actual_cols = set(y_context.columns)
                if expected_cols != actual_cols:
                    raise ValueError(f"Data columns {actual_cols} don't match configured target_cols {expected_cols}")
        elif isinstance(y_context, pd.Series):
            # Convert Series to DataFrame
            data = pd.DataFrame({y_context.name if y_context.name else 'target_0': y_context.values})
        else:
            # Numpy array case
            if y_context.ndim == 1:
                data = pd.DataFrame({'target_0': y_context})
            else:
                # Multivariate numpy array
                columns = [f'target_{i}' for i in range(y_context.shape[1])]
                data = pd.DataFrame(y_context, columns=columns)
        
        # Calculate n_targets from inherited target_cols
        self.n_targets = len(self.target_cols)
        
        # Store original data for integration
        self.original_data = data.copy()
        
        # Check stationarity and apply differencing if needed
        stationarity = self._check_stationarity(data)
        print(f"[DEBUG] Stationarity check: {stationarity}")
        
        if not all(stationarity.values()) and self.d > 0:
            print(f"[DEBUG] Applying differencing of order {self.d}")
            data = self._difference_data(data, self.d)
            self.differenced_data = data.copy()
        else:
            print(f"[DEBUG] Data is stationary, no differencing needed")
            self.differenced_data = data.copy()
            self.d = 0  # No differencing needed
        
        # Fit VAR model
        self.model = VAR(data)
        
        # Determine optimal lag order if not specified
        if self.p == 'auto':
            lag_results = self.model.select_order(maxlags=self.maxlags)
            optimal_p = lag_results.aic  # Use AIC criterion
            print(f"[DEBUG] Optimal lag order (AIC): {optimal_p}")
            self.p = optimal_p
        
        # Fit the model
        self.fitted_model = self.model.fit(self.p)
        self.is_fitted = True
        
        print(f"[DEBUG] VAR model fitted with order p={self.p}, targets={self.n_targets}")
        return self
        
    def predict(self, 
                y_context: Union[pd.Series, np.ndarray, pd.DataFrame] = None, 
                y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None, 
                x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, 
                x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None, 
                **kwargs
    ) -> np.ndarray:
        """
        Make predictions using the trained Multivariate ARIMA model.
        
        TECHNIQUE: VAR Forecasting for Multiple Time Series
        - Uses fitted VAR model to predict multiple steps ahead
        - Predicts all targets simultaneously using their interdependencies
        - Handles both in-sample and out-of-sample forecasting
        - Reverses differencing to get predictions in original scale
        
        Args:
            y_context: Past target values for prediction context
            y_target: Future target values (used to determine prediction length)
            x_context: Past exogenous variables (optional, ignored for now)
            x_target: Future exogenous variables (optional, ignored for now)
            **kwargs: Additional keyword arguments
            
        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, n_targets)
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted. Call train first.")
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")

        # Convert input to DataFrame if needed
        if isinstance(y_context, pd.DataFrame):
            context_data = y_context.copy()
        elif isinstance(y_context, pd.Series):
            context_data = pd.DataFrame({y_context.name if y_context.name else 'target_0': y_context.values})
        else:
            # Numpy array case
            if y_context.ndim == 1:
                context_data = pd.DataFrame({'target_0': y_context})
            else:
                columns = [f'target_{i}' for i in range(y_context.shape[1])]
                context_data = pd.DataFrame(y_context, columns=columns)
        
        # Apply same differencing as during training
        if self.d > 0:
            differenced_context = self._difference_data(context_data, self.d)
        else:
            differenced_context = context_data.copy()
        
        forecast_steps = len(y_target) if hasattr(y_target, '__len__') else self.forecast_horizon
        
        # Calculate how many prediction windows we need
        num_windows = math.ceil(forecast_steps / self.forecast_horizon)
        
        all_predictions = []
        current_data = differenced_context.copy()
        
        for window in range(num_windows):
            # Predict using VAR model
            steps_to_predict = min(self.forecast_horizon, forecast_steps - len(all_predictions))
            
            # VAR forecast
            forecast_result = self.fitted_model.forecast(
                current_data.values[-self.p:], 
                steps=steps_to_predict
            )
            
            # Add predictions to list
            all_predictions.extend(forecast_result)
            
            # Update data for next window if needed
            if window < num_windows - 1:
                # Add predictions to current_data for next iteration
                pred_df = pd.DataFrame(forecast_result, columns=current_data.columns)
                current_data = pd.concat([current_data, pred_df], ignore_index=True)
        
        # Convert to numpy array
        predictions = np.array(all_predictions[:forecast_steps])
        
        # Reverse differencing if it was applied
        if self.d > 0:
            # Create DataFrame for integration
            pred_df = pd.DataFrame(predictions, columns=self.target_cols)
            integrated_predictions = self._integrate_data(pred_df, self.original_data, self.d)
            predictions = integrated_predictions.values
        
        return predictions
        
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        return {
            'p': self.p,
            'd': self.d,
            'maxlags': self.maxlags,
            'target_cols': self.target_cols,
            'n_targets': self.n_targets,
            'loss_functions': self.loss_functions,
            'primary_loss': self.primary_loss,
            'forecast_horizon': self.forecast_horizon
        }
        
    def set_params(self, **params: Dict[str, Any]) -> 'MultivariateARIMAModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        for key, value in params.items():
            if hasattr(self, key):
                # Ensure proper type conversion for numeric parameters
                if key in ['p', 'd', 'maxlags', 'forecast_horizon']:
                    value = int(value)
                setattr(self, key, value)
        # Note: target_cols is inherited from parent class and shouldn't be modified
        return self
        
    def save(self, path: str) -> None:
        """
        Save the Multivariate ARIMA model to disk.
        
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
            'params': self.get_params(),
            'fitted_model': self.fitted_model,
            'original_data': self.original_data,
            'differenced_data': self.differenced_data
        }
        
        # Save model state to file
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
    def load(self, path: str) -> None:
        """
        Load the Multivariate ARIMA model from disk.
        
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
        self.fitted_model = model_state['fitted_model']
        self.original_data = model_state['original_data']
        self.differenced_data = model_state['differenced_data']
        self.set_params(**model_state['params'])
        
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
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        elif isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
            
        # Store for logging
        self._last_y_true = y_true
        self._last_y_pred = y_pred
        
        # For multivariate data, compute loss for each target and average
        if y_true.ndim == 2 and y_pred.ndim == 2:
            # Multivariate case - compute loss for each target and average
            all_metrics = {}
            for i in range(y_true.shape[1]):
                target_true = y_true[:, i]
                target_pred = y_pred[:, i]
                
                # Ensure both arrays have the same length
                min_length = min(len(target_true), len(target_pred))
                target_true = target_true[:min_length]
                target_pred = target_pred[:min_length]
                
                # Compute metrics for this target
                target_metrics = self.evaluator.evaluate(target_pred, target_true)
                
                # Store metrics for this target
                for metric_name, metric_value in target_metrics.items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)
            
            # Average metrics across all targets
            averaged_metrics = {}
            for metric_name, metric_values in all_metrics.items():
                averaged_metrics[metric_name] = np.mean(metric_values)
            
            return averaged_metrics
        else:
            # Univariate case - use original logic
            # Handle shape mismatches
            if y_pred.ndim == 2 and y_true.ndim == 1:
                # If predictions are 2D and true values are 1D, flatten predictions
                if y_pred.shape[0] == 1:
                    y_pred = y_pred.flatten()
                elif y_pred.shape[1] == 1:
                    y_pred = y_pred.flatten()
                else:
                    y_pred = y_pred[0]
            
            # Ensure both arrays have the same length
            min_length = min(len(y_true), len(y_pred))
            y_true = y_true[:min_length]
            y_pred = y_pred[:min_length]
            
            # Use evaluator to compute all metrics
            return self.evaluator.evaluate(y_pred, y_true)
