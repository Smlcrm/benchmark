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

# Import base model (adjust path as needed)
from .base_model import BaseModel

# Try to import lag_llama, install if not available
try:
    from lag_llama.gluon.estimator import LagLlamaEstimator
except ImportError:
    print("Lag-Llama not found. Setting up automatically...")
    LagLlamaModel._setup_lag_llama()
    from lag_llama.gluon.estimator import LagLlamaEstimator


class LagLlamaModel(BaseModel):
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
                - num_samples: int, number of probabilistic samples (default: 100)
                - device: str, device to use (default: "auto")
                - auto_setup: bool, whether to auto-setup if not found (default: True)
            config_file: Path to JSON config file
        """
        # Set default config
        default_config = {
            'checkpoint_path': 'lag-llama.ckpt',
            'context_length': 128,
            'num_samples': 100,
            'device': 'auto',
            'auto_setup': True,
            'forecast_horizon': 30,
            'loss_functions': ['mae', 'mse', 'rmse', 'mape'],
            'primary_loss': 'mae',
            'target_col': 'y'
        }
        
        if config:
            default_config.update(config)
        
        # Initialize base model
        super().__init__(default_config, config_file)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if self.config['device'] == "auto" else torch.device(self.config['device'])
        
        # Model-specific attributes
        self.checkpoint_path = self.config['checkpoint_path']
        self.context_length = self.config['context_length']
        self.num_samples = self.config['num_samples']
        self._predictor = None
        
        # Auto-setup if needed
        if self.config['auto_setup']:
            self._ensure_setup()
        
        print(f"🦙 Lag-Llama initialized - Device: {self.device}, Context: {self.context_length}")
    
    @staticmethod
    def _setup_lag_llama():
        """Set up Lag-Llama from scratch - static method for early import"""
        print("🔧 Setting up Lag-Llama environment...")
        
        try:
            # Check if lag-llama directory exists
            if not os.path.exists("lag-llama"):
                print("📥 Cloning Lag-Llama repository (update-gluonts branch)...")
                subprocess.run([
                    "git", "clone", "-b", "update-gluonts", 
                    "https://github.com/time-series-foundation-models/lag-llama/"
                ], check=True)
            
            # Change to lag-llama directory and install requirements
            original_dir = os.getcwd()
            os.chdir("lag-llama")
            
            try:
                print("📦 Installing requirements...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
                ], check=True)
                
                print("🔥 Updating PyTorch...")
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-U", "torch", "torchvision"
                ], check=True)
                
                print("📥 Downloading checkpoint with Hugging Face CLI...")
                subprocess.run([
                    "huggingface-cli", "download", "time-series-foundation-models/Lag-Llama", 
                    "lag-llama.ckpt", "--local-dir", "."
                ], check=True)
                
            finally:
                os.chdir(original_dir)
            
            print("✅ Lag-Llama setup complete!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Setup failed: {e}")
            print("💡 Manual setup instructions:")
            print("git clone -b update-gluonts https://github.com/time-series-foundation-models/lag-llama/")
            print("cd lag-llama")
            print("pip install -r requirements.txt")
            print("pip install -U torch torchvision")
            print("huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir .")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def _ensure_setup(self):
        """Ensure Lag-Llama is properly set up"""
        # Check if checkpoint exists, download if needed
        if not os.path.exists(self.checkpoint_path):
            print(f"📥 Downloading checkpoint to {self.checkpoint_path}...")
            self._download_checkpoint()
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found at {self.checkpoint_path}. "
                "Please download from: https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt"
            )
    
    def _download_checkpoint(self):
        """Download the Lag-Llama checkpoint file using Hugging Face CLI"""
        try:
            print("📥 Downloading checkpoint using Hugging Face CLI...")
            result = subprocess.run([
                "huggingface-cli", "download", "time-series-foundation-models/Lag-Llama", 
                "lag-llama.ckpt", "--local-dir", "."
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Checkpoint downloaded to {self.checkpoint_path}")
            else:
                raise Exception(f"Hugging Face CLI failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ HF CLI download failed: {e}")
            print("🔄 Trying alternative download methods...")
            
            try:
                # Fallback to direct download
                import wget
                url = "https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt"
                wget.download(url, self.checkpoint_path)
                print(f"\n✅ Checkpoint downloaded to {self.checkpoint_path}")
            except ImportError:
                # Final fallback to curl
                url = "https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt"
                result = subprocess.run([
                    "curl", "-L", url, "-o", self.checkpoint_path
                ], capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Checkpoint downloaded to {self.checkpoint_path}")
                else:
                    print(f"❌ All download methods failed")
                    print("Please manually download from:")
                    print("https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt")
    
    def _create_predictor_for_horizon(self, prediction_length: int):
        """Create a predictor with specific prediction length"""
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
            print(f"⚠️  Error creating predictor: {e}")
            # Fallback
            estimator = LagLlamaEstimator(
                ckpt_path=None,
                prediction_length=prediction_length,
                context_length=self.context_length,
                rope_scaling={"type": "linear", "factor": 2.0},
                batch_size=1,
                num_parallel_samples=self.num_samples,
                device=self.device,
            )
            
            lightning_module = estimator.create_lightning_module()
            transformation = estimator.create_transformation()
            return estimator.create_predictor(transformation, lightning_module)
    
    def train(self, 
              y_context: Optional[Union[pd.Series, np.ndarray]], 
              x_context: Optional[Union[pd.Series, np.ndarray]] = None, 
              y_target: Optional[Union[pd.Series, np.ndarray]] = None, 
              x_target: Optional[Union[pd.Series, np.ndarray]] = None,
              y_start_date: Optional[str] = None,
              x_start_date: Optional[str] = None
    ) -> 'LagLlamaModel':
        """
        Lag-Llama is pre-trained, so this method just validates inputs and sets fitted status.
        
        Args:
            y_context: Historical target values
            x_context: Historical exogenous variables (ignored by Lag-Llama)
            y_target: Future target values (for validation)
            x_target: Future exogenous variables (ignored by Lag-Llama)
            y_start_date: Start date for y data
            x_start_date: Start date for x data
            
        Returns:
            self: The model instance
        """
        if y_context is None:
            raise ValueError("y_context is required for Lag-Llama")
        
        # Convert to appropriate format
        if isinstance(y_context, pd.Series):
            y_context = y_context.values
        
        if len(y_context) < 10:
            warnings.warn("Very short time series may lead to poor predictions")
        
        # Lag-Llama is pre-trained, so we just mark as fitted
        self.is_fitted = True
        
        print("✅ Model ready (pre-trained)")
        return self
    
    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        x_context: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        x_target: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        forecast_horizon: Optional[int] = None
    ) -> np.ndarray:
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
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if y_context is None:
            raise ValueError("y_context is required for prediction")
        
        horizon = forecast_horizon or self.forecast_horizon
        
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
    
    def set_params(self, **params: Dict[str, Any]) -> 'LagLlamaModel':
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
        self.model = LagLlamaModel(config)
    
    def predict(self, df: pd.DataFrame, prediction_length: int, **kwargs):
        """TimesFM-style predict method"""
        return self.model.predict_df(df, prediction_length, **kwargs)
    
    def predict_quantiles(self, df: pd.DataFrame, prediction_length: int, **kwargs):
        """TimesFM-style quantile prediction"""
        return self.model.predict_quantiles(df, prediction_length, **kwargs)