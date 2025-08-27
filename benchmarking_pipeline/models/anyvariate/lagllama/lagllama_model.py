import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import warnings
import os
import subprocess
import sys
from gluonts.dataset.pandas import PandasDataset
from gluonts.evaluation import make_evaluation_predictions
from benchmarking_pipeline.models.foundation_model import FoundationModel

from .lag_llama.gluon.estimator import LagLlamaEstimator
from .lag_llama.gluon.predictor import LagLlamaPredictor

# Try to import lag_llama, install if not available


class LagllamaModel(FoundationModel):
    """
    Lag-Llama model implementation that inherits from BaseModel.
    Works seamlessly like TimesFM with automatic setup.
    """
    
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize Lag-Llama model with BaseModel interface.
        
        Args:
            config: Configuration dictionary containing:
                - checkpoint_path: str, path to checkpoint (default: "lag-llama.ckpt")
                - context_length: int, context window size (default: 128)
                - prediction_length: int, number of time series elements to predict (30)
                - num_samples: int, number of probabilistic samples (default: 5)
                - device: str, device to use (default: "auto")
            config_file: Path to JSON config file
        """
        
        # Initialize base model
        super().__init__(config, config_file)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model-specific attributes
        self.context_length = self.config.get('context_length', 4)
        self.num_samples = self.config.get('num_samples', 5)
        self.batch_size = self.config.get('batch_size', 4)
        # forecast_horizon is inherited from parent class (FoundationModel)
        self.model = None
        self.predictor = None
        
        print(f"🦙 Lag-Llama initialized - Device: {self.device}, Context: {self.context_length}")
    
    def _create_predictor_for_horizon(self, forecast_horizon: int):
        """Create a predictor for a specific forecast horizon."""
        # Calculate scaling factor based on context and forecast horizon
        scaling_factor = max(1.0, (self.context_length + forecast_horizon) /
                           self.context_length)
        
        # Create the predictor with the specified horizon
        predictor = LagLlamaPredictor.from_pretrained(
            "time-series-foundation-models/Lag-Llama",
            prediction_length=forecast_horizon,
            context_length=self.context_length,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            scaling_factor=scaling_factor,
        )
        return predictor

    def fit(self, df: pd.DataFrame, forecast_horizon: int = None, verbose: bool = True):
        """
        Fit the Lag-Llama model to the data.
        
        Args:
            df: DataFrame with time series data
            forecast_horizon: Forecast horizon (uses model config if not specified)
            verbose: Whether to print progress information
        """
        if forecast_horizon is None:
            forecast_horizon = self.forecast_horizon
        
        if verbose:
            print(f"🦙 Fitting Lag-Llama with forecast_horizon={forecast_horizon}")
        
        # Create predictor for this horizon
        self.predictor = self._create_predictor_for_horizon(forecast_horizon)
        
        # Fit the model
        self.predictor.train(df)
        self.is_fitted = True
        
        if verbose:
            print("✅ Lag-Llama training completed!")
        
        return self
    
    def predict(self,
            y_context: Optional[Union[pd.Series, np.ndarray]] = None,
            y_target: Union[pd.Series, np.ndarray] = None,
            y_context_timestamps = None,
            y_target_timestamps = None,
            **kwargs) -> Union[np.ndarray, Dict[str, List[float]]]:
        """
        Generate predictions using the fitted Lag-Llama model.
        
        Args:
            y_context: Historical time series data
            y_target: Future target values (not used for prediction)
            y_context_timestamps: Timestamps for context data
            y_target_timestamps: Timestamps for target data (not used)
            **kwargs: Additional arguments
            
        Returns:
            Predicted values as numpy array or dictionary
        """
        if y_context is None:
            raise ValueError("y_context is required for prediction")
        
        horizon = self.forecast_horizon
        
        # Convert input to DataFrame format
        if isinstance(y_context, np.ndarray):
            if y_context.ndim == 1:
                df = pd.DataFrame({'series': y_context})
            else:
                df = pd.DataFrame(y_context.T)
        else:
            df = y_context
        
        # Ensure we have a fitted predictor
        if not self.is_fitted:
            self.fit(df, verbose=False)
        
        # Generate predictions
        try:
            predictions = self.predictor.predict(df)
            return predictions
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            # Return zero predictions as fallback
            series_names = df.columns if hasattr(df, 'columns') else [f'series_{i}' for i in range(df.shape[1])]
            return {name: [0.0] * horizon for name in series_names}

    def _predict_internal(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        freq: str = "D",
        return_samples: bool = False
    ) -> Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        """Internal prediction method - similar to standalone forecaster"""
        
        # Create predictor for this prediction length
        predictor = self._create_predictor_for_horizon(prediction_length)
        
        # Prepare data for each series
        all_series_data = []
        series_names = []
        
        for series_name in df.columns:
            series_data = df[series_name].dropna()
            
            if len(series_data) == 0:
                warnings.warn(f"Series '{series_name}' has no valid data. Skipping.")
                continue
            
            # Create timestamps
            end_date = datetime.now()
            start_date = end_date - timedelta(days=len(series_data)-1)
            timestamps = pd.date_range(start=start_date, periods=len(series_data), freq=freq)
            
            # Create series DataFrame
            series_df = pd.DataFrame({
                'ds': timestamps,
                'target': series_data.values,
                'unique_id': series_name
            })
            
            all_series_data.append(series_df)
            series_names.append(series_name)
        
        if not all_series_data:
            return {}
        
        # Combine all series
        combined_df = pd.concat(all_series_data, ignore_index=True)
        
        # Create GluonTS dataset
        try:
            dataset = PandasDataset.from_long_dataframe(
                combined_df, 
                target='target', 
                item_id='unique_id', 
                timestamp='ds', 
                freq=freq
            )
        except Exception:
            # Fallback
            dataset_dict = {}
            for series_name in series_names:
                series_df = combined_df[combined_df['unique_id'] == series_name]
                dataset_dict[series_name] = series_df.set_index('ds')['target']
            dataset = PandasDataset(dataset_dict, target='target')
        
        # Generate forecasts
        try:
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=dataset,
                predictor=predictor,
                num_samples=self.num_samples
            )
            
            forecasts = list(forecast_it)
            
        except Exception as e:
            print(f"Error during forecasting: {e}")
            return {name: [0.0] * prediction_length for name in series_names}
        
        # Process results
        results = {}
        for forecast in forecasts:
            series_name = getattr(forecast, 'item_id', 'unknown')
            
            if return_samples:
                try:
                    results[series_name] = {
                        'mean': forecast.mean.tolist(),
                        'median': forecast.quantile(0.5).tolist(),
                        'q10': forecast.quantile(0.1).tolist(),
                        'q90': forecast.quantile(0.9).tolist(),
                        'samples': forecast.samples.tolist()
                    }
                except Exception:
                    results[series_name] = {
                        'mean': [0.0] * prediction_length,
                        'median': [0.0] * prediction_length,
                        'q10': [0.0] * prediction_length,
                        'q90': [0.0] * prediction_length,
                        'samples': [[0.0] * prediction_length]
                    }
            else:
                try:
                    results[series_name] = forecast.mean.tolist()
                except Exception:
                    results[series_name] = [0.0] * prediction_length
        
        return results
    
    def get_params(self) -> Dict[str, Any]:
        return {
            'context_length': self.context_length,
            'num_samples': self.num_samples,
            'batch_size': self.batch_size,
            'forecast_horizon': self.forecast_horizon,
        }
    
    def set_params(self, **params: Dict[str, Any]) -> 'LagllamaModel':
        """Set model parameters"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            if key in self.config:
                self.config[key] = value
        return self
    
    # TimesFM-style convenience methods
    def predict_df(
        self,
        df: pd.DataFrame,
        forecast_horizon: int,
        return_samples: bool = False
    ) -> Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        """
        TimesFM-style prediction on DataFrame.
        
        Args:
            df: DataFrame with time series columns
            forecast_horizon: Number of steps to forecast
            return_samples: Whether to return probabilistic samples
            
        Returns:
            Dictionary with forecasts for each series
        """
        return self._predict_internal(df, forecast_horizon, return_samples=return_samples)
    
    def predict_quantiles(
        self,
        df: pd.DataFrame,
        forecast_horizon: int,
        quantile_levels: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Generate quantile forecasts.
        
        Args:
            df: Historical time series data
            forecast_horizon: Number of future steps to predict
            quantile_levels: List of quantile levels to compute
            
        Returns:
            Nested dict with series names and quantile forecasts
        """
        sample_results = self._predict_internal(df, forecast_horizon, return_samples=True)
        
        quantile_results = {}
        for series_name, forecasts in sample_results.items():
            if 'samples' in forecasts:
                samples = np.array(forecasts['samples'])
                quantiles = {}
                
                for q in quantile_levels:
                    quantiles[f'q{int(q*100)}'] = np.percentile(samples, q*100, axis=0).tolist()
                
                quantile_results[series_name] = quantiles
            else:
                quantile_results[series_name] = {
                    f'q{int(q*100)}': forecasts.get('mean', [0.0] * forecast_horizon) 
                    for q in quantile_levels
                }
        
        return quantile_results


# Convenience wrapper for standalone usage (like TimesFM)
class LagLlamaForecaster:
    """
    Standalone forecaster wrapper for easy usage (mirrors TimesFM interface)
    """
    
    def __init__(self, checkpoint_path: str = "lag-llama.ckpt", **kwargs):
        """Initialize with TimesFM-like interface"""
        config = {'checkpoint_path': checkpoint_path}
        config.update(kwargs)
        self.model = LagllamaModel(config)
    
    def predict(self, df: pd.DataFrame, forecast_horizon: int, **kwargs):
        """TimesFM-style predict method"""
        return self.model.predict_df(df, forecast_horizon, **kwargs)
    
    def predict_quantiles(self, df: pd.DataFrame, forecast_horizon: int, **kwargs):
        """TimesFM-style quantile prediction"""
        return self.model.predict_quantiles(df, forecast_horizon, **kwargs)