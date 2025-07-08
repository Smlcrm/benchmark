"""
DeepAR model implementation.

TO BE CHANGED: This model needs to be updated to match the new interface with y_context, x_context, y_target, x_target parameters.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import lightning.pytorch as pl
from pytorch_forecasting import DeepAR, TimeSeriesDataSet
from typing import Dict, Any, Union, Tuple, Optional
import pickle
import os
from benchmarking_pipeline.models.base_model import BaseModel
from pytorch_lightning.loggers import TensorBoardLogger
import time

# Using this link to assist in building this model file implementation:
# https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/deepar.html

# Useful for understanding how to work with TimeSeriesDataset
# https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/building.html#passing-data
# time_idx is the index of the particular element in the time series
# group_ids or group is what time series the element belongs to. 
# If you only have one time series, you can set group to be 0.

class DeepARModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Initialize LSTM model with given configuration.
        
        Args:
            config: Configuration dictionary containing model parameters
                - hidden_size: int, number of DeepAR units per recurrent layer
                - rnn_layers: int, number of DeepAR RNN layers
                - dropout: float, dropout rate in DeepAR RNN layers
                - learning_rate: float, learning rate for the DeepAR model
                - batch_size: int, batch size for training
                - target_col: str, name of target column
                - feature_cols: list of str, names of feature columns
                - loss_functions: List[str], list of loss function names to use
                - primary_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
                - max_encoder_length: int, number of steps to use as input data during autoregressive training
                - max_prediction_length: int, number of steps to use as output data during autoregressive training
                - epochs: int, number of training epochs
                - gradient_clip_val: float, threshold to clip gradient to during training
                - num_workers: int, number of workers used for the dataloaders
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, config_file)
        self.hidden_size = self.config.get('hidden_size', 10)
        self.rnn_layers = self.config.get('rnn_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.target_col = self.config.get('target_col', 'y')
        self.feature_cols = self.config.get('feature_cols', None)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        self.max_encoder_length = self.config.get('max_encoder_length', 2)
        self.max_prediction_length = self.config.get('max_prediction_length', 2)
        self.epochs = self.config.get('epochs', 100)
        self.gradient_clip_val = self.config.get('gradient_clip_val', 0.1)
        self.num_workers = self.config.get('num_workers', 7)
        self.model = None
    
    def _series_to_TimeSeriesDataset(self, series):
        
        values = None
        if isinstance(series, pd.Series):
            values = series.values
        else:
            values = series

        dataset_altered_form = pd.DataFrame({
            "value": values,
            "time_idx": list(range(len(series))),
            "group_id": ["0"] * len(series)
        })

        dataset = TimeSeriesDataSet(
            dataset_altered_form,
            time_idx="time_idx",
            target="value",
            group_ids=["group_id"],
            time_varying_unknown_reals=["value"],
            max_encoder_length = self.max_encoder_length,
            max_prediction_length = self.max_prediction_length,
            static_categoricals=["group_id"]
        )

        return dataset
    
    def _build_model(self, training_dataset):
        self.model = DeepAR.from_dataset(training_dataset,
                                           learning_rate=self.learning_rate,
                                           hidden_size=self.hidden_size,
                                           rnn_layers=self.rnn_layers,
                                           dropout=self.dropout)
    
    def train(self, 
              y_context: Union[pd.Series, np.ndarray], 
              y_target: Union[pd.Series, np.ndarray] = None, 
              x_context: Union[pd.Series, np.ndarray] = None, 
              x_target: Union[pd.Series, np.ndarray] = None, 
              y_start_date: Optional[str] = None,
              x_start_date: Optional[str] = None,
              **kwargs
    ) -> 'DeepARModel':
        training_dataset = self._series_to_TimeSeriesDataset(y_context)
        validation_dataset = self._series_to_TimeSeriesDataset(y_target)

        if self.model is None:
            self._build_model(training_dataset)

        train_dataloader = training_dataset.to_dataloader(
            train=True, batch_size=self.batch_size, batch_sampler="synchronized",
            num_workers=self.num_workers, persistent_workers=True
        )

        validation_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, batch_sampler="synchronized",
            num_workers=self.num_workers, persistent_workers=True
        )
        # Set up TensorBoard logger
        log_dir = os.path.join("runs", "DeepAR")
        tb_logger = TensorBoardLogger(save_dir=log_dir, name=time.strftime("%Y%m%d-%H%M%S"))
        # Create the PyTorch Lightning trainer with logger
        trainer = pl.Trainer(logger=tb_logger, accelerator="auto", gradient_clip_val=self.gradient_clip_val, max_epochs=self.epochs)
        trainer.fit(self.model,train_dataloader,validation_dataloader)
        return self
        
        
    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Union[pd.Series, np.ndarray] = None,
        x_context: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        x_target: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        forecast_horizon: Optional[int] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            y_context: Recent/past target values 
            y_target: Future target values 
            x_context: Recent/past exogenous variables 
            x_target: Future exogenous variables for the forecast horizon 
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            
        Returns:
            np.ndarray: Model predictions with shape (n_samples, forecast_horizon)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train first.")
        validation_dataset = self._series_to_TimeSeriesDataset(y_target)
        validation_dataloader = validation_dataset.to_dataloader(
            train=False, batch_size=self.batch_size, batch_sampler="synchronized",
            num_workers=self.num_workers, persistent_workers=True
        )

        predictions = self.model.predict(validation_dataloader)
        predictions = predictions.cpu().numpy() # shape is (length of val set, self.max_prediction_length)
        predictions = predictions[:,0]
        return predictions
        
        

      
    def set_params(self, **params: Dict[str, Any]) -> 'BaseModel':
        """
        Set model parameters. This will rebuild the sktime model instance.
        """
        model_params_changed = False
        for key, value in params.items():
            if hasattr(self, key):
                # Check if this is a model parameter that requires refitting
                if key in ['learning_rate', 'hidden_size', 'rnn_layers', 'dropout'] and getattr(self, key) != value:
                    model_params_changed = True
                setattr(self, key, value)
            else:
                # Update config if parameter not found in instance attributes
                self.config[key] = value
        
        # If model parameters changed, reset the fitted model
        if model_params_changed:
            self.model = None
            
        return self
      
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call train first.")
        return({
        "hidden_size": self.hidden_size, 
        "rnn_layers" : self.rnn_layers,
        "dropout" : self.dropout,
        "learning_rate" : self.learning_rate,
        "batch_size" : self.batch_size,
        "target_col" : self.target_col,
        "feature_cols" : self.feature_cols,
        "forecast_horizon" : self.forecast_horizon,
        "max_encoder_length" : self.max_encoder_length,
        "max_prediction_length" : self.max_prediction_length,
        "epochs" : self.epochs,
        "gradient_clip_val" : self.gradient_clip_val,
        "num_workers" : self.num_workers
        })
    