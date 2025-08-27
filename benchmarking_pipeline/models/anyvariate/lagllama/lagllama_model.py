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
        self.prediction_length = self.config.get('prediction_length', 30)
        self.context_length = self.config.get('context_length', 4)
        self.num_samples = self.config.get('num_samples', 5)
        self.batch_size = self.config.get('batch_size', 4)
        self.target_col = self.config.get('target_col', 'y')
        
        print(f"🦙 Lag-Llama initialized - Device: {self.device}, Context: {self.context_length}")
    
    def _create_predictor_for_horizon(self, prediction_length: int):
        """Create a predictor with specific prediction length"""

        """
        try:
            # Load checkpoint
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            estimator_args = ckpt.get("hyper_parameters", {}).get("model_kwargs", {})
            
            # RoPE scaling
            rope_scaling_arguments = {
                "type": "linear",
                "factor": max(1.0, (self.context_length + prediction_length) / 
                             estimator_args.get("context_length", 32)),
            }
            
            # Create estimator
            estimator = LagLlamaEstimator(
                ckpt_path=None,
                prediction_length=prediction_length,
                context_length=self.context_length,
                input_size=estimator_args.get("input_size", 1),
                n_layer=estimator_args.get("n_layer", 32),
                n_embd_per_head=estimator_args.get("n_embd_per_head", 32),
                n_head=estimator_args.get("n_head", 32),
                scaling=estimator_args.get("scaling", "mean"),
                time_feat=estimator_args.get("time_feat", True),
                rope_scaling=rope_scaling_arguments,
                batch_size=1,
                num_parallel_samples=self.num_samples,
                device=self.device,
            )
            
            # Load weights
            lightning_module = estimator.create_lightning_module()
            lightning_module.load_state_dict(ckpt["state_dict"], strict=False)
            
            # Create predictor
            transformation = estimator.create_transformation()
            return estimator.create_predictor(transformation, lightning_module)
        
        except Exception as e:
            print(f"⚠️  Error creating predictor: {e}")"""
        
        # Fallback
        estimator = LagLlamaEstimator(
            ckpt_path=None,
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            rope_scaling={"type": "linear", "factor": 2.0},
            batch_size=self.batch_size,
            num_parallel_samples=self.num_samples,
            device=self.device,
        )
        
        lightning_module = estimator.create_lightning_module()
        transformation = estimator.create_transformation()
        return estimator.create_predictor(transformation, lightning_module)
    
    def predict(self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Union[pd.Series, np.ndarray] = None,
        y_context_timestamps = None,
        y_target_timestamps = None,
        **kwargs) -> np.ndarray:
        """
        Make predictions using Lag-Llama.
        
        Args:
            y_context: Historical target values
            x_context: Historical exogenous variables (ignored)
            x_target: Future exogenous variables (ignored)
            forecast_horizon: Number of steps to forecast
            
        Returns:
            np.ndarray: Predictions with shape (forecast_horizon,)
        """
        
        if y_context is None:
            raise ValueError("y_context is required for prediction")
        
        horizon = self.prediction_length
        
        # Convert input to DataFrame format
        if isinstance(y_context, pd.Series):
            df = pd.DataFrame({'series_0': y_context.values})
        elif isinstance(y_context, np.ndarray):
            df = pd.DataFrame({'series_0': y_context})
        else:
            df = pd.DataFrame({'series_0': y_context})
        
        # Use the forecaster logic
        results = self._predict_internal(df, horizon)
        
        # Return as numpy array for single series
        if 'series_0' in results:
            return np.array(results['series_0'])
        else:
            # Fallback
            print("[DEBUG] We return an array of all zeros here for lag llama")
            return np.zeros(horizon)
    
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
        """Get model parameters"""
        return {
            'checkpoint_path': self.checkpoint_path,
            'context_length': self.context_length,
            'num_samples': self.num_samples,
            'device': str(self.device),
            'forecast_horizon': self.forecast_horizon
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
        prediction_length: int,
        return_samples: bool = False
    ) -> Union[Dict[str, List[float]], Dict[str, Dict[str, List[float]]]]:
        """
        TimesFM-style prediction on DataFrame.
        
        Args:
            df: DataFrame with time series columns
            prediction_length: Number of steps to forecast
            return_samples: Whether to return probabilistic samples
            
        Returns:
            Dictionary with forecasts for each series
        """
        return self._predict_internal(df, prediction_length, return_samples=return_samples)
    
    def predict_quantiles(
        self,
        df: pd.DataFrame,
        prediction_length: int,
        quantile_levels: List[float] = [0.1, 0.5, 0.9]
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Generate quantile forecasts.
        
        Args:
            df: Historical time series data
            prediction_length: Number of future steps to predict
            quantile_levels: List of quantile levels to compute
            
        Returns:
            Nested dict with series names and quantile forecasts
        """
        sample_results = self._predict_internal(df, prediction_length, return_samples=True)
        
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
                    f'q{int(q*100)}': forecasts.get('mean', [0.0] * prediction_length) 
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
    
    def predict(self, df: pd.DataFrame, prediction_length: int, **kwargs):
        """TimesFM-style predict method"""
        return self.model.predict_df(df, prediction_length, **kwargs)
    
    def predict_quantiles(self, df: pd.DataFrame, prediction_length: int, **kwargs):
        """TimesFM-style quantile prediction"""
        return self.model.predict_quantiles(df, prediction_length, **kwargs)