"""
Theta model implementation.

#HANDLING: This model needs to be updated to match the new interface with y_context, x_context, y_target, x_target parameters.

Updated : Updated to match the new interface with y_context, x_context, y_target, x_target parameters.
"""

import os
import pickle
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from sktime.forecasting.theta import ThetaForecaster
from benchmarking_pipeline.models.base_model import BaseModel

class ThetaModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize the Theta model with a given configuration.
        
        Args:
            config: Configuration dictionary for model parameters.
                    Example Format:
                    {'sp': 12} - for monthly data with yearly seasonality.   
            config_file: Path to a JSON configuration file.
        """
        super().__init__(config, config_file)
        self.sp = self.config.get('sp', 1)
        self.model = None
        
    def _build_model(self):
        """
        Build the ThetaForecaster model instance from the configuration.
        """
        # Only pass supported arguments to ThetaForecaster
        model_params = {"sp": self.sp}
        self.model = ThetaForecaster(**model_params)
        self.is_fitted = False

    def train(self, 
              y_context: Union[pd.Series, np.ndarray], 
              x_context: Union[pd.Series, np.ndarray] = None, 
              y_target: Union[pd.Series, np.ndarray] = None, 
              x_target: Union[pd.Series, np.ndarray] = None, 
              y_start_date: Optional[str] = None,
              x_start_date: Optional[str] = None,
              **kwargs
    ) -> 'ThetaModel':
        """
        Train the Theta model on given data. For this model, "training" involves
        decomposing the series and fitting exponential smoothing.
        
        Args:
            y_context: Historical target time series values (pd.Series or np.ndarray).
            x_context: Historical exogenous features (ignored by this univariate model).
            y_target: Target values for validation (ignored during training).
            x_target: Future exogenous features (ignored during training).
            y_start_date: Start date for y_context (optional).
            x_start_date: Start date for x_context (optional).
        
        Returns:
            self: The fitted model instance.
        """
        if self.model is None:
            self._build_model()
            
        if not isinstance(y_context, pd.Series):
            # sktime works best with Pandas Series with a proper index
            y_context = pd.Series(y_context)
            
        print(f"Fitting ThetaForecaster with parameters: {self.model.get_params()}...")
        # Theta is a univariate method, so we only use y_context and ignore x_context
        self.model.fit(y=y_context, X=x_context)  # X is ignored by ThetaForecaster
        self.is_fitted = True
        print("Training complete.")
        return self
        
    def predict(self, 
                y_context: Optional[Union[pd.Series, np.ndarray]] = None,
                x_context: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
                x_target: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
                forecast_horizon: Optional[int] = None) -> np.ndarray:
        """
        Make predictions using the trained Theta model.
        
        Args:
            y_context: Recent/past target values (ignored by Theta - uses training data).
            x_context: Recent/past exogenous variables (ignored by univariate model).
            x_target: Future exogenous variables for forecast horizon (ignored by univariate model,
                     but used to determine forecast length if forecast_horizon not provided).
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided).
            
        Returns:
            np.ndarray: Model predictions with shape (n_samples, forecast_horizon).
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Determine forecast horizon
        if forecast_horizon is not None:
            fh = np.arange(1, forecast_horizon + 1)
        elif x_target is not None:
            fh = np.arange(1, len(x_target) + 1)
        else:
            fh = np.arange(1, self.forecast_horizon + 1)
        
        # The sktime predict method uses the forecasting horizon (fh)
        predictions = self.model.predict(fh=fh)
        
        return predictions.values

    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters from the underlying sktime model.
        """
        if self.model:
            return self.model.get_params()
        return {"sp", self.sp}

    def set_params(self, **params: Dict[str, Any]) -> 'ThetaModel':
        """
        Set model parameters. This will rebuild the sktime model instance.
        """
        model_params_changed = False
        for key, value in params.items():
            if hasattr(self, key):
                # Check if this is a model parameter that requires refitting
                if key in ['sp'] and getattr(self, key) != value:
                    model_params_changed = True
                setattr(self, key, value)
            else:
                # Update config if parameter not found in instance attributes
                self.config[key] = value
        
        # If model parameters changed, reset the fitted model
        if model_params_changed and self.is_fitted:
            self.model = None
            self.is_fitted = False
            
        return self

    def save(self, path: str) -> None:
        """
        Save the trained sktime model to disk using pickle.
        
        Args:
            path: Path to save the model.
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Cannot save an unfitted model")
            
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load(self, path: str) -> 'ThetaModel':
        """
        Load a trained sktime model from disk.
        
        Args:
            path: Path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
            
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True
        return self