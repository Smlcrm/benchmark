"""
Multivariate XGBoost model implementation for time series forecasting.

This model extends the univariate XGBoost to handle multiple target variables simultaneously.
Uses sklearn's MultiOutputRegressor to handle multiple targets with advanced feature engineering.
"""

import os
import pickle
from typing import Dict, Any, Union, List, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from benchmarking_pipeline.models.base_model import BaseModel


class MultivariateXGBoostModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Multivariate XGBoost model with given configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
                - lookback_window: int, number of past timesteps to use as features
                - forecast_horizon: int, number of future timesteps to predict
                - target_cols: list of str, names of target columns (for multivariate)
                - model_params: dict, parameters for the underlying XGBRegressor model
                - loss_functions: List[str], list of loss function names to use
                - primary_loss: str, primary loss function for training
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, config_file)
        
        # Extract model parameters from config
        self.lookback_window = self.config.get('lookback_window', 10)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        self.target_cols = self.config.get('target_cols')
        self.n_targets = len(self.target_cols)  # Number of target variables
        
        self._build_model()
        
    def _build_model(self):
        """
        Build the Multivariate XGBoost model instance using MultiOutputRegressor 
        for direct multi-output forecasting across multiple target variables.
        """
        # Get hyperparameters from config
        model_params = self.config.get('model_params', {})
        
        # Ensure random_state for reproducibility if not provided
        if 'random_state' not in model_params:
            model_params['random_state'] = 42
            
        base_xgb = XGBRegressor(**model_params)
        self.model = MultiOutputRegressor(base_xgb)
        self.is_fitted = False

    def _create_multivariate_features(self, y_series: np.ndarray, x_series: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create advanced multivariate time series features for XGBoost.
        
        Args:
            y_series: Target time series with shape (timesteps, n_targets)
            x_series: Exogenous variables (optional)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets arrays
        """
        if y_series.ndim == 1:
            y_series = y_series.reshape(-1, 1)
            
        n_samples = len(y_series) - self.lookback_window - self.forecast_horizon + 1
        
        if n_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {self.lookback_window + self.forecast_horizon} samples.")
        
        features = []
        targets = []
        
        for i in range(n_samples):
            sample_features = []
            
            # Get lookback window for all targets
            lookback_data = y_series[i:i + self.lookback_window]  # Shape: (lookback_window, n_targets)
            
            # 1. Lag features for each target (flattened)
            lag_features = lookback_data.flatten()  # Shape: (lookback_window * n_targets,)
            sample_features.extend(lag_features)
            
            # 2. Rolling statistics for each target individually
            for target_idx in range(self.n_targets):
                target_data = lookback_data[:, target_idx]
                
                # Basic statistics
                rolling_mean = np.mean(target_data)
                rolling_std = np.std(target_data)
                rolling_min = np.min(target_data)
                rolling_max = np.max(target_data)
                rolling_median = np.median(target_data)
                
                # Trend features
                trend = np.polyfit(range(self.lookback_window), target_data, 1)[0]
                
                # Volatility features
                rolling_range = rolling_max - rolling_min
                rolling_iqr = np.percentile(target_data, 75) - np.percentile(target_data, 25)
                
                sample_features.extend([
                    rolling_mean, rolling_std, rolling_min, rolling_max, rolling_median,
                    trend, rolling_range, rolling_iqr
                ])
            
            # 3. Cross-correlation features between targets (if multivariate)
            if self.n_targets > 1:
                for i_target in range(self.n_targets):
                    for j_target in range(i_target + 1, self.n_targets):
                        # Correlation between target pairs
                        corr = np.corrcoef(lookback_data[:, i_target], lookback_data[:, j_target])[0, 1]
                        if np.isnan(corr):
                            corr = 0.0  # Handle constant series
                        sample_features.append(corr)
                        
                        # Ratio features
                        mean_i = np.mean(lookback_data[:, i_target])
                        mean_j = np.mean(lookback_data[:, j_target])
                        if mean_j != 0:
                            ratio = mean_i / mean_j
                        else:
                            ratio = 0.0
                        sample_features.append(ratio)
            
            # 4. Temporal features (if window is large enough)
            if self.lookback_window >= 7:
                # Weekly patterns (last 7 values for each target)
                recent_data = lookback_data[-7:]
                for target_idx in range(self.n_targets):
                    recent_mean = np.mean(recent_data[:, target_idx])
                    sample_features.append(recent_mean)
            
            # 5. Add exogenous features if available
            if x_series is not None and len(x_series) > i + self.lookback_window:
                current_x = x_series[i + self.lookback_window]
                sample_features.extend(current_x.flatten())
            
            features.append(sample_features)
            
            # Create target: next forecast_horizon steps for all targets, flattened
            future_values = y_series[i + self.lookback_window:i + self.lookback_window + self.forecast_horizon]
            targets.append(future_values.flatten())  # Shape: (forecast_horizon * n_targets,)
        
        return np.array(features), np.array(targets)

    def train(self, y_context: Union[pd.Series, np.ndarray, pd.DataFrame], y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None, x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None, y_start_date: pd.Timestamp = None, x_start_date: pd.Timestamp = None, **kwargs) -> 'MultivariateXGBoostModel':
        """
        Train the Multivariate XGBoost model for direct multi-output forecasting.
        
        TECHNIQUE: Advanced Multivariate Feature Engineering with Gradient Boosting
        - Creates lag features from all target variables
        - Adds rolling statistics per target (mean, std, min, max, median, trend, range, IQR)
        - Includes cross-correlation and ratio features between target pairs
        - Incorporates temporal patterns (weekly means if window ≥ 7)
        - Uses XGBoost's gradient boosting with MultiOutputRegressor for non-linear pattern learning
        
        Args:
            y_context: Past target values (time series) - used for training (can be DataFrame for multivariate)
            y_target: Future target values (optional, for extended training)
            x_context: Past exogenous variables (optional)
            x_target: Future exogenous variables (optional)
            y_start_date: The start date timestamp for y_context and y_target
            x_start_date: The start date timestamp for x_context and x_target
            **kwargs: Additional keyword arguments
            
        Returns:
            self: The fitted model instance
        """
        if self.model is None:
            self._build_model()
        
        if y_context is None:
            raise ValueError("y_context (target series) must be provided for training.")

        # Convert input to numpy array and handle multivariate data
        if isinstance(y_context, pd.DataFrame):
            # Multivariate case - use all columns
            y_data = y_context.values
            # Update target_cols if not already set
            # target_cols should already be set from config, validate consistency
            if self.target_cols and len(y_context.columns) > 0:
                expected_cols = set(self.target_cols)
                actual_cols = set(y_context.columns)
                if expected_cols != actual_cols:
                    raise ValueError(f"Data columns {actual_cols} don't match configured target_cols {expected_cols}")
        elif isinstance(y_context, pd.Series):
            # Univariate case - reshape to 2D
            y_data = y_context.values.reshape(-1, 1)
                            # For univariate case, validate against config
                if len(self.target_cols) != 1:
                    raise ValueError(f"Expected 1 target column for univariate data, but config has {len(self.target_cols)}: {self.target_cols}")
            self.n_targets = 1
        else:
            # Numpy array case
            y_data = y_context
            if y_data.ndim == 1:
                y_data = y_data.reshape(-1, 1)
                self.n_targets = 1
            else:
                self.n_targets = y_data.shape[1]

        # Handle exogenous variables
        if isinstance(x_context, (pd.Series, pd.DataFrame)):
            x_data = x_context.values
        else:
            x_data = x_context

        # Combine context and target for full training series if y_target is provided
        if y_target is not None:
            if isinstance(y_target, pd.DataFrame):
                y_target_vals = y_target.values
            elif isinstance(y_target, pd.Series):
                y_target_vals = y_target.values.reshape(-1, 1)
            else:
                y_target_vals = y_target
                if y_target_vals.ndim == 1:
                    y_target_vals = y_target_vals.reshape(-1, 1)
            
            y_data = np.concatenate([y_data, y_target_vals], axis=0)

        # Create features and targets
        X, y = self._create_multivariate_features(y_data, x_data)
        
        print(f"[DEBUG][MultivariateXGBoost] Training X shape: {X.shape}, y shape: {y.shape}")
        print(f"[DEBUG][MultivariateXGBoost] n_targets: {self.n_targets}, target_cols: {self.target_cols}")
        print(f"Training Multivariate XGBoost model with {X.shape[0]} samples and {X.shape[1]} features...")
        
        # Handle NaNs in training data
        if np.isnan(X).any() or np.isnan(y).any():
            print("[WARNING][MultivariateXGBoost] NaNs found in training data! Attempting to fix...")
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)
        
        try:
            # Train model
            self.model.fit(X, y)
            self.is_fitted = True
            print("[INFO][MultivariateXGBoost] Training completed successfully")
        except Exception as e:
            print(f"[ERROR][MultivariateXGBoost] Training failed: {e}")
            # Try with more conservative parameters
            try:
                print("[INFO][MultivariateXGBoost] Trying with conservative parameters...")
                conservative_params = {
                    'n_estimators': 50,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'random_state': 42
                }
                base_xgb = XGBRegressor(**conservative_params)
                self.model = MultiOutputRegressor(base_xgb)
                self.model.fit(X, y)
                self.is_fitted = True
                print("[INFO][MultivariateXGBoost] Training succeeded with fallback parameters")
            except Exception as e2:
                print(f"[ERROR][MultivariateXGBoost] Fallback training also failed: {e2}")
                raise ValueError(f"Multivariate XGBoost training failed: {e}")
        
        return self

    def rolling_predict(self, y_context: np.ndarray, y_target: Union[pd.Series, np.ndarray, pd.DataFrame], x_context: np.ndarray = None, **kwargs) -> np.ndarray:
        """
        Autoregressive rolling prediction for Multivariate XGBoost.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.
        
        Args:
            y_context: Historical context data with shape (timesteps, n_targets)
            y_target: Target data to determine prediction length
            x_context: Exogenous context data (optional)
            
        Returns:
            np.ndarray: Predictions with shape (forecast_steps, n_targets)
        """
        # Convert y_context to proper shape
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_context = y_context.values
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)

        print(f"[DEBUG][MultivariateXGBoost] Rolling predict - y_context shape: {y_context.shape}")
        print(f"[DEBUG][MultivariateXGBoost] Rolling predict - lookback_window: {self.lookback_window}")
        
        if len(y_context) < self.lookback_window:
            raise ValueError(f"y_context too short: {len(y_context)} < lookback_window {self.lookback_window}")
        if np.isnan(y_context).any():
            raise ValueError("NaNs in initial y_context!")
        
        preds = []
        # Use the entire context, maintaining the multivariate structure
        context = y_context.copy()
        total_steps = len(y_target) if hasattr(y_target, '__len__') else self.forecast_horizon
        steps_done = 0
        
        while steps_done < total_steps:
            steps = min(self.forecast_horizon, total_steps - steps_done)
            
            # Take the last lookback_window timesteps for feature creation
            current_window = context[-self.lookback_window:]
            
            try:
                # Create features using the current window
                sample_features = []
                
                # 1. Lag features (flattened)
                lag_features = current_window.flatten()
                sample_features.extend(lag_features)
                
                # 2. Rolling statistics for each target
                for target_idx in range(self.n_targets):
                    target_data = current_window[:, target_idx]
                    
                    rolling_mean = np.mean(target_data)
                    rolling_std = np.std(target_data)
                    rolling_min = np.min(target_data)
                    rolling_max = np.max(target_data)
                    rolling_median = np.median(target_data)
                    trend = np.polyfit(range(self.lookback_window), target_data, 1)[0]
                    rolling_range = rolling_max - rolling_min
                    rolling_iqr = np.percentile(target_data, 75) - np.percentile(target_data, 25)
                    
                    sample_features.extend([
                        rolling_mean, rolling_std, rolling_min, rolling_max, rolling_median,
                        trend, rolling_range, rolling_iqr
                    ])
                
                # 3. Cross-correlation features (if multivariate)
                if self.n_targets > 1:
                    for i_target in range(self.n_targets):
                        for j_target in range(i_target + 1, self.n_targets):
                            corr = np.corrcoef(current_window[:, i_target], current_window[:, j_target])[0, 1]
                            if np.isnan(corr):
                                corr = 0.0
                            sample_features.append(corr)
                            
                            mean_i = np.mean(current_window[:, i_target])
                            mean_j = np.mean(current_window[:, j_target])
                            ratio = mean_i / mean_j if mean_j != 0 else 0.0
                            sample_features.append(ratio)
                
                # 4. Temporal features (if applicable)
                if self.lookback_window >= 7:
                    recent_data = current_window[-7:]
                    for target_idx in range(self.n_targets):
                        recent_mean = np.mean(recent_data[:, target_idx])
                        sample_features.append(recent_mean)
                
                # 5. Handle exogenous features (simplified - use last available)
                if x_context is not None:
                    # For simplicity, repeat the last exogenous values
                    if x_context.ndim == 1:
                        sample_features.extend(x_context[-1:])
                    else:
                        sample_features.extend(x_context[-1].flatten())
                
                # Convert to numpy array and predict
                X_pred = np.array(sample_features).reshape(1, -1)
                
                # Check for NaNs and handle them
                if np.isnan(X_pred).any():
                    print("[WARNING][MultivariateXGBoost] NaNs found in prediction features! Attempting to fix...")
                    X_pred = np.nan_to_num(X_pred, nan=0.0)
                
                pred_flat = self.model.predict(X_pred).flatten()
                
                # Check for NaNs in predictions and handle them
                if np.isnan(pred_flat).any():
                    print("[WARNING][MultivariateXGBoost] NaNs in predictions! Replacing with last valid value")
                    pred_flat = np.nan_to_num(pred_flat, nan=0.0)
                
                # Reshape predictions back to (forecast_horizon, n_targets)
                pred_reshaped = pred_flat.reshape(self.forecast_horizon, self.n_targets)
                
                print(f"[DEBUG][MultivariateXGBoost] Rolling predict X_pred shape: {X_pred.shape}, pred_reshaped shape: {pred_reshaped.shape}")
                
                # Only take as many steps as needed
                pred_steps = pred_reshaped[:steps]
                preds.append(pred_steps)
                
                # Update context with new predictions
                context = np.concatenate([context, pred_steps], axis=0)
                steps_done += steps
                
            except Exception as e:
                print(f"[ERROR][MultivariateXGBoost] Prediction failed: {e}")
                # Fallback: use last values repeated
                fallback_pred = np.tile(context[-1], (steps, 1))
                preds.append(fallback_pred)
                context = np.concatenate([context, fallback_pred], axis=0)
                steps_done += steps
        
        # Concatenate all predictions
        result = np.concatenate(preds, axis=0)
        return result

    def predict(self, y_context: Union[pd.Series, np.ndarray, pd.DataFrame] = None, y_target: Union[pd.Series, np.ndarray, pd.DataFrame] = None, x_context: Union[pd.Series, pd.DataFrame, np.ndarray] = None, x_target: Union[pd.Series, pd.DataFrame, np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Make predictions using the trained Multivariate XGBoost model.
        
        TECHNIQUE: Autoregressive Rolling Window Multi-step Forecasting with Advanced Features
        - Uses last lookback_window values to create comprehensive multivariate features
        - Predicts forecast_horizon steps ahead using trained MultiOutputRegressor
        - Uses its own predictions to update the window and predict further steps
        - Repeats until forecast_steps are reached
        
        Args:
            y_context: Historical context data
            y_target: Target data (used to determine prediction length)
            x_context: Exogenous context data (optional)
            x_target: Exogenous target data (optional)
            
        Returns:
            np.ndarray: Model predictions with shape (forecast_steps, n_targets)
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        if y_target is None:
            raise ValueError("y_target must be provided to determine prediction length.")

        # Convert y_context to proper format
        if isinstance(y_context, pd.DataFrame):
            y_context_vals = y_context.values
        elif isinstance(y_context, pd.Series):
            y_context_vals = y_context.values.reshape(-1, 1)
        else:
            y_context_vals = y_context
            if y_context_vals.ndim == 1:
                y_context_vals = y_context_vals.reshape(-1, 1)

        # Convert x_context if provided
        x_context_vals = None
        if x_context is not None:
            if isinstance(x_context, (pd.Series, pd.DataFrame)):
                x_context_vals = x_context.values
            else:
                x_context_vals = x_context

        if y_target is not None:
            # Use rolling prediction to cover the full test set
            preds = self.rolling_predict(y_context_vals, y_target, x_context_vals)
            print(f"[DEBUG][MultivariateXGBoost] Final rolling preds shape: {preds.shape}")
            return preds
        
        # Single prediction case (forecast_horizon steps)
        if len(y_context_vals) < self.lookback_window:
            raise ValueError(f"y_context too short for prediction: {len(y_context_vals)} < {self.lookback_window}")
        
        # Use last lookback_window values for single prediction
        current_window = y_context_vals[-self.lookback_window:]
        
        # Create features (same logic as in rolling_predict)
        sample_features = []
        
        # Lag features
        lag_features = current_window.flatten()
        sample_features.extend(lag_features)
        
        # Rolling statistics for each target
        for target_idx in range(self.n_targets):
            target_data = current_window[:, target_idx]
            
            rolling_mean = np.mean(target_data)
            rolling_std = np.std(target_data)
            rolling_min = np.min(target_data)
            rolling_max = np.max(target_data)
            rolling_median = np.median(target_data)
            trend = np.polyfit(range(self.lookback_window), target_data, 1)[0]
            rolling_range = rolling_max - rolling_min
            rolling_iqr = np.percentile(target_data, 75) - np.percentile(target_data, 25)
            
            sample_features.extend([
                rolling_mean, rolling_std, rolling_min, rolling_max, rolling_median,
                trend, rolling_range, rolling_iqr
            ])
        
        # Cross-correlation features
        if self.n_targets > 1:
            for i_target in range(self.n_targets):
                for j_target in range(i_target + 1, self.n_targets):
                    corr = np.corrcoef(current_window[:, i_target], current_window[:, j_target])[0, 1]
                    if np.isnan(corr):
                        corr = 0.0
                    sample_features.append(corr)
                    
                    mean_i = np.mean(current_window[:, i_target])
                    mean_j = np.mean(current_window[:, j_target])
                    ratio = mean_i / mean_j if mean_j != 0 else 0.0
                    sample_features.append(ratio)
        
        # Temporal features
        if self.lookback_window >= 7:
            recent_data = current_window[-7:]
            for target_idx in range(self.n_targets):
                recent_mean = np.mean(recent_data[:, target_idx])
                sample_features.append(recent_mean)
        
        # Exogenous features
        if x_context_vals is not None:
            if x_context_vals.ndim == 1:
                sample_features.extend(x_context_vals[-1:])
            else:
                sample_features.extend(x_context_vals[-1].flatten())
        
        X_pred = np.array(sample_features).reshape(1, -1)
        preds_flat = self.model.predict(X_pred).flatten()
        
        # Reshape to (forecast_horizon, n_targets)
        preds = preds_flat.reshape(self.forecast_horizon, self.n_targets)
        print(f"[DEBUG][MultivariateXGBoost] Single predict X_pred shape: {X_pred.shape}, preds shape: {preds.shape}")
        return preds

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        params = {
            'lookback_window': self.lookback_window,
            'forecast_horizon': self.forecast_horizon,
            'target_cols': self.target_cols,
            'n_targets': self.n_targets,
            'loss_functions': self.loss_functions,
            'primary_loss': self.primary_loss
        }
        
        if self.model:
            params.update(self.model.get_params())
            
        return params

    def set_params(self, **params: Dict[str, Any]) -> 'MultivariateXGBoostModel':
        """
        Set model parameters. This will update the model instance with new parameters.
        Handles MultiOutputRegressor by prefixing XGBoost params with 'estimator__'.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        # Handle model-level params
        model_level_keys = ['lookback_window', 'forecast_horizon', 'target_cols']
        for k in model_level_keys:
            if k in params:
                setattr(self, k, params[k])
        
        # Update n_targets if target_cols changed
        if 'target_cols' in params:
            self.n_targets = len(self.target_cols)
        
        # Prepare params for underlying XGBRegressor
        xgb_param_names = XGBRegressor().get_params().keys()
        mo_params = {}
        config_params = {}
        
        for k, v in params.items():
            if k in xgb_param_names:
                mo_params[f'estimator__{k}'] = v
                config_params[k] = v
            elif k == 'n_jobs':
                mo_params[k] = v
                config_params[k] = v
                
        if hasattr(self, 'model') and isinstance(self.model, MultiOutputRegressor):
            self.model.set_params(**mo_params)
        elif hasattr(self, 'model') and self.model is not None:
            # Fallback for single XGBRegressor
            single_params = {k: v for k, v in params.items() if k in xgb_param_names or k == 'n_jobs'}
            self.model.set_params(**single_params)
            
        # Update config as well
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        self.config['model_params'].update(config_params)
        
        for k in model_level_keys:
            if k in params:
                self.config[k] = params[k]
                
        return self

    def save(self, path: str) -> None:
        """
        Save the trained Multivariate XGBoost model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        
        model_state = {
            'config': self.config,
            'is_fitted': self.is_fitted,
            'lookback_window': self.lookback_window,
            'forecast_horizon': self.forecast_horizon,
            'target_cols': self.target_cols,
            'n_targets': self.n_targets,
            'model': self.model
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
            
    def load(self, path: str) -> 'MultivariateXGBoostModel':
        """
        Load a trained Multivariate XGBoost model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            self: The loaded model instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        self.config = model_state['config']
        self.is_fitted = model_state['is_fitted']
        self.lookback_window = model_state['lookback_window']
        self.forecast_horizon = model_state['forecast_horizon']
        self.target_cols = model_state['target_cols']
        self.n_targets = model_state['n_targets']
        self.model = model_state['model']
        
        return self

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_function: str = None) -> Dict[str, float]:
        """
        Compute all loss metrics between true and predicted values using the Evaluator class.
        
        Args:
            y_true: True target values with shape (timesteps, n_targets)
            y_pred: Predicted values with shape (timesteps, n_targets)
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
            
        # Store for potential logging
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
            
            # Use evaluator to compute all metrics
            return self.evaluator.evaluate(y_pred, y_true)