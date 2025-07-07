"""
Theta model implementation.

HANDLING: This model needs to be updated to match the new interface with y_context, x_context, y_target, x_target parameters.
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
        # Get hyperparameters from config
        model_params = {"sp": self.sp}
        
        self.model = ThetaForecaster(**model_params)
        self.is_fitted = False

    def train(self, 
              y_context: Union[pd.Series, np.ndarray], 
              y_target: Union[pd.Series, np.ndarray] = None, 
              x_context: Union[pd.Series, np.ndarray] = None, 
              x_target: Union[pd.Series, np.ndarray] = None, 
              y_start_date: Optional[str] = None,
              x_start_date: Optional[str] = None,
              **kwargs
    ) -> 'ThetaModel':
        """
        Train the Theta model on given data. For this model, "training" involves
        decomposing the series and fitting exponential smoothing.
        
        Args:
            x_context: Training features (ignored by this univariate model, but required for API consistency).
            y_context: Target time series values (pd.Series or np.ndarray).
        
        Returns:
            self: The fitted model instance.
        """
        if self.model is None:
            self._build_model()
            
        if not isinstance(y_context, pd.Series):
            # sktime works best with Pandas Series with a proper index
            y_context = pd.Series(y_context)
            
        print(f"Fitting ThetaForecaster with parameters: {self.model.get_params()}...")
        self.model.fit(y=y_context, X=x_context) # X is ignored
        self.is_fitted = True
        print("Training complete.")
        return self
        
    def predict(self, y_context, y_target=None, y_context_timestamps=None, y_target_timestamps=None, **kwargs):
        """
        Make predictions using the trained Theta model.
        
        Args:
            x_target: Input data for prediction. The number of rows in X determines the
               number of steps to forecast. The content of X is ignored.
            
        Returns:
            np.ndarray: Model predictions.
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call train() first.")
        
        # Create a forecasting horizon based on the number of samples in the input X.
        if y_target is not None:
            fh = np.arange(1, len(y_target) + 1)
        else:
            fh = self.forecast_horizon
        
        # The sktime predict method uses the forecasting horizon (fh).
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