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


class SvrModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Support Vector Regression (SVR) model with a given configuration.
        Uses direct multi-output strategy via sklearn's MultiOutputRegressor.
        """
        super().__init__(config, config_file)
        self.scaler = StandardScaler() # SVR is sensitive to feature scaling
        # Extract lookback_window from config
        self.lookback_window = self.config.get('lookback_window', 10)
        # forecast_horizon is inherited from parent class (BaseModel)
        self._build_model()

    def _build_model(self):
        """
        Build the SVR model instance from the configuration using MultiOutputRegressor for direct multi-output forecasting.
        """
        model_params = self.config.get('model_params', {})
        base_svr = SVR(**model_params)
        self.model = MultiOutputRegressor(base_svr)
        self.is_fitted = False

    def _create_features_targets(self, y_series, lookback_window, forecast_horizon):
        """
        Create features and multi-step targets for direct multi-output forecasting.
        Each sample uses the previous lookback_window values as features and the next forecast_horizon values as targets.
        """
        X, y = [], []
        for i in range(len(y_series) - lookback_window - forecast_horizon + 1):
            X.append(y_series[i:i+lookback_window])
            y.append(y_series[i+lookback_window:i+lookback_window+forecast_horizon])
        return np.array(X), np.array(y)

    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, **kwargs) -> 'SvrModel':
        """
        Train the SVR model for direct multi-output forecasting using MultiOutputRegressor.
        """
        if self.model is None:
            self._build_model()
        if y_context is None:
            raise ValueError("y_context (target series) must be provided for training.")
        # Combine context and target for full training series if y_target is provided
        # Ensure both arrays are 1D before concatenation
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_context = y_context.values.flatten()
        elif isinstance(y_context, np.ndarray) and y_context.ndim > 1:
            y_context = y_context.flatten()
            
        if y_target is not None:
            if isinstance(y_target, (pd.Series, pd.DataFrame)):
                y_target = y_target.values.flatten()
            elif isinstance(y_target, np.ndarray) and y_target.ndim > 1:
                y_target = y_target.flatten()
            y_series = np.concatenate([y_context, y_target])
        else:
            y_series = y_context
        X, y = self._create_features_targets(y_series, self.lookback_window, self.forecast_horizon)
        print(f"[DEBUG][SVR] Training X shape: {X.shape}, y shape: {y.shape}")
        
        # Scale features (time index is not used; features are lagged values)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self

    def rolling_predict(self, y_context, y_target, **kwargs):
        """
        Autoregressive rolling prediction for MultiOutputRegressor SVR.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.
        """
        # Debug: Print initial context info
        print(f"[DEBUG][SVR] Initial y_context: {y_context}")
        print(f"[DEBUG][SVR] Initial y_context length: {len(y_context)}, lookback_window: {self.lookback_window}")
        if len(y_context) < self.lookback_window:
            raise ValueError(f"y_context too short: {len(y_context)} < lookback_window {self.lookback_window}")

        preds = []
        context = np.array(y_context).flatten().tolist()
        total_steps = len(y_target)
        steps_done = 0
        
        while steps_done < total_steps:
            steps = min(self.forecast_horizon, total_steps - steps_done)
            X_pred = np.array(context[-self.lookback_window:]).reshape(1, -1)              
            X_pred_scaled = self.scaler.transform(X_pred)
            pred = self.model.predict(X_pred_scaled).flatten()
            
            print(f"[DEBUG][SVR] Rolling predict X_pred shape: {X_pred.shape}, pred shape: {pred.shape}")
            # Only take as many steps as needed
            pred = pred[:steps]
            preds.extend(pred)
            context.extend(pred)
            steps_done += steps
        
        return np.array(preds)

    def predict(self, y_context=None, y_target=None, **kwargs) -> np.ndarray:
        """
        Make direct multi-output predictions using the trained SVR model.
        If y_target is provided, use rolling_predict to predict the entire length autoregressively.
        Returns a vector of length forecast_horizon or full test length.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        if y_context is None:
            raise ValueError("y_context must be provided to determine prediction context.")
        if y_target is None:
            raise ValueError("y_target is required to determine prediction length. No forecast_horizon fallback allowed.")
        
        # Use rolling prediction to cover the full test set
        preds = self.rolling_predict(y_context, y_target)
        print(f"[DEBUG][SVR] Final rolling preds shape: {preds.shape}")
        return preds

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the underlying scikit-learn model.
        """
        if self.model:
            return self.model.get_params()
        # Return config params if model is not yet instantiated
        return self.config.get('model_params', {})

    def set_params(self, **params: Dict[str, Any]) -> 'SvrModel':
        """
        Set model parameters. This will rebuild the model instance with the new parameters.
        Handles MultiOutputRegressor by prefixing SVR params with 'estimator__'.
        Model-level params (lookback_window, forecast_horizon) are set as attributes.
        """
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
        
        # Prepare params for underlying SVR
        svr_param_names = SVR().get_params().keys()
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
            
    def load(self, path: str) -> 'SvrModel':
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