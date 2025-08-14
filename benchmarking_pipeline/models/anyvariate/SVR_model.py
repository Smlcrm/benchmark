"""
SVR model implementation.
"""

import os
import pickle
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from benchmarking_pipeline.models.base_model import BaseModel
from sklearn.multioutput import MultiOutputRegressor


class SVRModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Support Vector Regression (SVR) model with a given configuration.
        Uses direct multi-output strategy via sklearn's MultiOutputRegressor.
        """
        super().__init__(config, config_file)
        self.scaler = StandardScaler() # SVR is sensitive to feature scaling
        # Extract lookback_window and forecast_horizon from config
        self.lookback_window = self.config.get('lookback_window', 10)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        self._build_model()

    def _build_model(self):
        """
        Build separate SVR models for each target variable.
        This approach is more appropriate for multivariate time series forecasting.
        """
        model_params = self.config.get('model_params', {})
        
        # Create separate SVR models for each target
        self.models = {}
        self.scalers = {}
        
        # Handle target columns - support both univariate and multivariate
        if 'target_cols' in self.config:
            self.target_cols = self.config['target_cols']
        else:
            self.target_cols = ['y']  # Default for univariate
            
        # Ensure target_cols is always a list
        if isinstance(self.target_cols, str):
            self.target_cols = [self.target_cols]
            
        # Create a model for each target
        for target in self.target_cols:
            base_svr = SVR(**model_params)
            self.models[target] = base_svr
            self.scalers[target] = StandardScaler()
        
        self.is_fitted = False
        
        # Set target_col for compatibility with hyperparameter tuner
        self.target_col = self.target_cols[0] if len(self.target_cols) == 1 else 'y'

    def _create_features_targets(self, y_series, lookback_window, forecast_horizon, target_idx=None):
        """
        Create features and targets for one-step-ahead prediction.
        Each sample uses the previous lookback_window values from ALL targets as features 
        and the next single value of the specific target as the target.
        This captures cross-target relationships and dependencies.
        """
        # Handle multivariate data
        if y_series.ndim == 1:
            # Univariate case - reshape to 2D
            y_series = y_series.reshape(-1, 1)
            n_targets = 1
            target_idx = 0
        else:
            # Multivariate case
            n_targets = y_series.shape[1]
            if target_idx is None:
                target_idx = 0  # Default to first target
                
        X, y = [], []
        for i in range(len(y_series) - lookback_window):
            # Features: use the last lookback_window values from ALL targets
            # This captures cross-target relationships
            features = []
            for target_j in range(n_targets):
                features.extend(y_series[i:i+lookback_window, target_j])
            X.append(features)
            
            # Target: use the next single value for the specific target
            target = y_series[i+lookback_window, target_idx]
            y.append(target)
        return np.array(X), np.array(y)

    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, np.ndarray] = None, x_target: Union[pd.Series, np.ndarray] = None, **kwargs) -> 'SVRModel':
        """
        Train separate SVR models for each target variable.
        Supports both univariate and multivariate data.
        
        Args:
            y_context: Training data (should be used for training)
            y_target: Validation/test data (should NOT be used for training during hyperparameter tuning)
        """
        if self.models is None:
            self._build_model()
        if y_context is None:
            raise ValueError("y_context (target series) must be provided for training.")
        
        # Convert to numpy array if needed
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_context = y_context.values
        
        # For training, use ONLY y_context (training data)
        # y_target should only be used for validation/testing, not for training
        y_series = y_context
            
        # Handle multivariate data - ensure 2D
        if y_series.ndim == 1:
            y_series = y_series.reshape(-1, 1)
            
        # Train a separate model for each target
        for target_idx, target_name in enumerate(self.target_cols):
            print(f"[DEBUG][SVR] Training model for target: {target_name}")
            
            # Create features and targets for this specific target using ONLY training data
            X, y = self._create_features_targets(y_series, self.lookback_window, self.forecast_horizon, target_idx)
            print(f"[DEBUG][SVR] Training X shape: {X.shape}, y shape: {y.shape}")
            
            # Handle NaNs in training data
            if np.isnan(X).any() or np.isnan(y).any():
                print(f"[WARNING][SVR] NaNs found in training data for {target_name}! Attempting to fix...")
                X = np.nan_to_num(X, nan=0.0)
                y = np.nan_to_num(y, nan=0.0)
            
            try:
                # Scale features for this target
                self.scalers[target_name].fit(X)
                X_scaled = self.scalers[target_name].transform(X)
                self.models[target_name].fit(X_scaled, y)
                print(f"[DEBUG][SVR] Successfully trained model for {target_name}")
            except Exception as e:
                print(f"[ERROR][SVR] Training failed for {target_name}: {e}")
                # Try with more robust parameters
                try:
                    # Use more conservative SVR parameters
                    base_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
                    self.models[target_name] = base_svr
                    self.scalers[target_name].fit(X)
                    X_scaled = self.scalers[target_name].transform(X)
                    self.models[target_name].fit(X_scaled, y)
                    print(f"[INFO][SVR] Training succeeded for {target_name} with fallback parameters")
                except Exception as e2:
                    print(f"[ERROR][SVR] Fallback training also failed for {target_name}: {e2}")
                    raise ValueError(f"SVR training failed for {target_name}: {e}")
        
        self.is_fitted = True
        return self

    def rolling_predict(self, y_context, y_target, **kwargs):
        """
        Autoregressive rolling prediction using separate SVR models for each target.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.
        Uses one-step-ahead prediction for each iteration.
        Supports both univariate and multivariate data.
        """
        import numpy as np
        # Debug: Print initial context info
        print(f"[DEBUG][SVR] Initial y_context: {y_context}")
        print(f"[DEBUG][SVR] Initial y_context length: {len(y_context)}, lookback_window: {self.lookback_window}")
        
        # Convert to numpy array if needed
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_context = y_context.values
            
        # Handle multivariate data - ensure 2D
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)
            
        if len(y_context) < self.lookback_window:
            raise ValueError(f"y_context too short: {len(y_context)} < lookback_window {self.lookback_window}")
        if np.isnan(y_context).any():
            raise ValueError("NaNs in initial y_context!")
        
        # Determine number of targets
        n_targets = y_context.shape[1]
        
        preds = []
        context = y_context.copy()
        total_steps = len(y_target)
        
        for step in range(total_steps):
            # Make one-step-ahead prediction for this step using separate models
            step_predictions = []
            for target_idx, target_name in enumerate(self.target_cols):
                # Create features using lagged values from ALL targets
                features = []
                for target_j in range(context.shape[1]):
                    features.extend(context[-self.lookback_window:, target_j])
                features = np.array(features).reshape(1, -1)
                
                # Scale features for this target
                features_scaled = self.scalers[target_name].transform(features)
                
                # Predict for this target (one-step-ahead)
                target_pred = self.models[target_name].predict(features_scaled)
                step_predictions.append(target_pred)
            
            # Combine predictions from all targets
            combined_pred = np.column_stack(step_predictions)  # Shape: (1, n_targets)
            preds.append(combined_pred.flatten())  # Flatten to (n_targets,)
            
            # Update context with the prediction
            context = np.concatenate([context, combined_pred], axis=0)
        
        return np.array(preds)

    def predict(self, y_context=None, y_target=None, x_context=None, x_target=None, forecast_steps=None, **kwargs) -> np.ndarray:
        """
        Make predictions using the trained SVR models for each target variable.
        For single-step prediction, returns shape (1, n_targets).
        For multi-step prediction, uses rolling prediction and returns shape (forecast_steps, n_targets).
        Supports both univariate and multivariate data.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        if y_context is None:
            raise ValueError("y_context must be provided to determine prediction context.")
        if y_target is not None:
            # Use rolling prediction to cover the full test set
            preds = self.rolling_predict(y_context, y_target)
            print(f"[DEBUG][SVR] Final rolling preds shape: {preds.shape}")
            return preds
            
        # Use the last lookback_window values as features for a single prediction
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_context = y_context.values
            
        # Handle multivariate data - ensure 2D
        if y_context.ndim == 1:
            y_context = y_context.reshape(-1, 1)
            
        # Make predictions for each target
        all_predictions = []
        for target_idx, target_name in enumerate(self.target_cols):
            # Create features using lagged values from ALL targets
            features = []
            for target_j in range(y_context.shape[1]):
                features.extend(y_context[-self.lookback_window:, target_j])
            features = np.array(features).reshape(1, -1)
            
            # Scale features for this target
            features_scaled = self.scalers[target_name].transform(features)
            
            # Predict for this target (one-step-ahead)
            pred = self.models[target_name].predict(features_scaled)
            all_predictions.append(pred)
        
        # Combine predictions from all targets
        combined_predictions = np.column_stack(all_predictions)  # Shape: (1, n_targets)
        
        # If forecast_steps > 1, use rolling prediction
        if forecast_steps is not None and forecast_steps > 1:
            # Create dummy target for rolling prediction
            dummy_target = np.zeros((forecast_steps, y_context.shape[1]))
            return self.rolling_predict(y_context, dummy_target)
            
        print(f"[DEBUG][SVR] Single predict features shape: {features.shape}, preds shape: {combined_predictions.shape}")
        return combined_predictions

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
        Handles MultiOutputRegressor by prefixing SVR params with 'estimator__'.
        Model-level params (lookback_window, forecast_horizon) are set as attributes.
        """
        # Check if target_cols is being changed
        target_cols_changed = 'target_cols' in params and params['target_cols'] != getattr(self, 'target_cols', None)
        
        # Handle model-level params
        model_level_keys = ['lookback_window', 'forecast_horizon']
        for k in model_level_keys:
            if k in params:
                setattr(self, k, params[k])
        
        # Convert parameter types for scikit-learn compatibility
        converted_params = {}
        for k, v in params.items():
            if k == 'gamma':
                # Handle gamma parameter: convert string numbers to float
                if isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit():
                    converted_params[k] = float(v)
                else:
                    converted_params[k] = v
            elif k == 'C' and isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit():
                converted_params[k] = float(v)
            elif k == 'epsilon' and isinstance(v, str) and v.replace('.', '').replace('-', '').isdigit():
                converted_params[k] = float(v)
            else:
                converted_params[k] = v
        
        # Get SVR parameter names
        svr_param_names = SVR().get_params().keys()
        
        # If target_cols changed, we need to rebuild the model completely
        if target_cols_changed:
            self._build_model()
        else:
            # Prepare params for underlying SVR
            mo_params = {}
            for k, v in converted_params.items():
                if k in svr_param_names:
                    mo_params[f'estimator__{k}'] = v
                elif k == 'n_jobs':
                    mo_params[k] = v
            if hasattr(self, 'model') and isinstance(self.model, MultiOutputRegressor):
                self.model.set_params(**mo_params)
            elif hasattr(self, 'model') and self.model is not None:
                # fallback for non-multioutput
                self.model.set_params(**{k: v for k, v in converted_params.items() if k in svr_param_names or k == 'n_jobs'})
        
        # Update config as well
        if 'model_params' not in self.config:
            self.config['model_params'] = {}
        self.config['model_params'].update({k: v for k, v in converted_params.items() if k in svr_param_names or k == 'n_jobs'})
        for k in model_level_keys:
            if k in params:
                self.config[k] = params[k]
        if 'target_cols' in params:
            self.config['target_cols'] = params['target_cols']
                
        # Reset fitted state when parameters change
        self.is_fitted = False
        
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