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
                - target_col: str, name of target column
                - feature_cols: list of str, names of feature columns
                - loss_functions: List[str], list of loss function names to use
                - primary_loss: str, primary loss function for training
                - forecast_horizon: int, number of steps to forecast ahead
            config_file: Path to a JSON configuration file
        """
        super().__init__(config, config_file)
        self.hidden_size = self.config.get('hidden_size', 10)
        self.rnn_layers = self.config.get('rnn_layers', 2)
        self.dropout = self.config.get('dropout', 0.1)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.target_col = self.config.get('target_col', 'y')
        self.feature_cols = self.config.get('feature_cols', None)
        self.forecast_horizon = self.config.get('forecast_horizon', 1)
        self.model = None
    
    def _dataframe_to_TimeSeriesDataset(self, dataframe):
          """
          Takes a Pandas dataframe and converts it into a TimeSeriesDataset from Pytorch Forecasting
          """
          pass
    
    def train(self, y_context: Union[pd.Series, np.ndarray], y_target: Union[pd.Series, np.ndarray] = None, x_context: Union[pd.Series, np.ndarray] = None, x_target: Union[pd.Series, np.ndarray] = None) -> 'DeepARModel':
        print(f"y_context: {y_context}")
        print(f"y_target: {y_target}")
        print(f"x_context: {x_context}")
        print(f"x_target: {x_target}")
      
    def set_params(self, **params: Dict[str, Any]) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Model parameters to set
            
        Returns:
            self: The model instance with updated parameters
        """
        pass
      
    def get_params(self) -> Dict[str, Any]:
        """
        Get the current model parameters.
        
        Returns:
            Dict[str, Any]: Dictionary of model parameters
        """
        pass

    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        x_context: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        x_target: Optional[Union[pd.Series, pd.DataFrame, np.ndarray]] = None,
        forecast_horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            y_context: Recent/past target values (for sequence models, optional for ARIMA)
            x_context: Recent/past exogenous variables (for sequence models, optional for ARIMA)
            x_target: Future exogenous variables for the forecast horizon (required if model uses exogenous variables)
            forecast_horizon: Number of steps to forecast (defaults to model config if not provided)
            
        Returns:
            np.ndarray: Model predictions with shape (n_samples, forecast_horizon)
        """
        pass