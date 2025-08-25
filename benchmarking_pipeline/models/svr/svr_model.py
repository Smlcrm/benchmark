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
        # Extract lookback_window and forecast_horizon from config
        self.lookback_window = self.config.get('lookback_window', 10)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
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

    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, np.ndarray] = None, x_target: Union[pd.Series, np.ndarray] = None, **kwargs) -> 'SvrModel':
        """
        Train the SVR model for direct multi-output forecasting using MultiOutputRegressor.
        """
        if self.model is None:
            self._build_model()
        if y_context is None:
            raise ValueError("y_context (target series) must be provided for training.")
        # Combine context and target for full training series if y_target is provided
        if y_target is not None:
            if isinstance(y_target, (pd.Series, pd.DataFrame)):
                y_target = y_target.values.flatten()
            y_series = np.concatenate([y_context, y_target])
        else:
            y_series = y_context if not isinstance(y_context, (pd.Series, pd.DataFrame)) else y_context.values.flatten()
        X, y = self._create_features_targets(y_series, self.lookback_window, self.forecast_horizon)
        print(f"[DEBUG][SVR] Training X shape: {X.shape}, y shape: {y.shape}")
        
        # Handle NaNs in training data
        if np.isnan(X).any() or np.isnan(y).any():
            print("[WARNING][SVR] NaNs found in training data! Attempting to fix...")
            X = np.nan_to_num(X, nan=0.0)
            y = np.nan_to_num(y, nan=0.0)
        
        try:
            # Scale features (time index is not used; features are lagged values)
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.fit(X_scaled, y)
            self.is_fitted = True
        except Exception as e:
            print(f"[ERROR][SVR] Training failed: {e}")
            # Try with more robust parameters
            try:
                # Use more conservative SVR parameters
                base_svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
                self.model = MultiOutputRegressor(base_svr)
                self.scaler.fit(X)
                X_scaled = self.scaler.transform(X)
                self.model.fit(X_scaled, y)
                self.is_fitted = True
                print("[INFO][SVR] Training succeeded with fallback parameters")
            except Exception as e2:
                print(f"[ERROR][SVR] Fallback training also failed: {e2}")
                raise ValueError(f"SVR training failed: {e}")
        
        return self

    def rolling_predict(self, y_context, y_target, **kwargs):
        """
        Autoregressive rolling prediction for MultiOutputRegressor SVR.
        Predicts the entire length of y_target by repeatedly using its own predictions as context.
        """
        import numpy as np
        # Debug: Print initial context info
        print(f"[DEBUG][SVR] Initial y_context: {y_context}")
        print(f"[DEBUG][SVR] Initial y_context length: {len(y_context)}, lookback_window: {self.lookback_window}")
        if len(y_context) < self.lookback_window:
            raise ValueError(f"y_context too short: {len(y_context)} < lookback_window {self.lookback_window}")
        if np.isnan(y_context).any():
            raise ValueError("NaNs in initial y_context!")
        
        preds = []
        context = np.array(y_context).flatten().tolist()
        total_steps = len(y_target)
        steps_done = 0
        
        while steps_done < total_steps:
            steps = min(self.forecast_horizon, total_steps - steps_done)
            X_pred = np.array(context[-self.lookback_window:]).reshape(1, -1)
            
            # Check for NaNs and handle them
            if np.isnan(X_pred).any():
                print("[WARNING][SVR] NaNs found in prediction input! Attempting to fix...")
                # Replace NaNs with the last valid value or 0
                X_pred = np.nan_to_num(X_pred, nan=0.0)
            
            try:
                X_pred_scaled = self.scaler.transform(X_pred)
                pred = self.model.predict(X_pred_scaled).flatten()
                
                # Check for NaNs in predictions and handle them
                if np.isnan(pred).any():
                    print("[WARNING][SVR] NaNs in predictions! Replacing with last valid value or 0")
                    pred = np.nan_to_num(pred, nan=0.0)
                
                print(f"[DEBUG][SVR] Rolling predict X_pred shape: {X_pred.shape}, pred shape: {pred.shape}")
                # Only take as many steps as needed
                pred = pred[:steps]
                preds.extend(pred)
                context.extend(pred)
                steps_done += steps
                
            except Exception as e:
                print(f"[ERROR][SVR] Prediction failed: {e}")
                # Fallback: use simple prediction (last value repeated)
                fallback_pred = [context[-1]] * steps
                preds.extend(fallback_pred)
                context.extend(fallback_pred)
                steps_done += steps
        
        return np.array(preds)

    def predict(self, y_context=None, y_target=None, x_context=None, x_target=None, **kwargs) -> np.ndarray:
        """
        Make direct multi-output predictions using the trained SVR model.
        If y_target is provided, use rolling_predict to predict the entire length autoregressively.
        Returns a vector of length forecast_horizon or full test length.
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
        # Use the last lookback_window values as features for a single multi-output prediction
        if isinstance(y_context, (pd.Series, pd.DataFrame)):
            y_context = y_context.values.flatten()
        X_pred = np.array(y_context[-self.lookback_window:]).reshape(1, -1)
        X_pred_scaled = self.scaler.transform(X_pred)
        preds = self.model.predict(X_pred_scaled)
        print(f"[DEBUG][SVR] Single predict X_pred shape: {X_pred.shape}, preds shape: {preds.shape}")
        return preds  # shape: (1, forecast_horizon)

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