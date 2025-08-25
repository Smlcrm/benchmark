import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Tuple, Optional
import pickle
import os
from benchmarking_pipeline.models.foundation_model import FoundationModel
from sktime.forecasting.ttm import TinyTimeMixerForecaster


class TinyTimeMixerModel(FoundationModel):

  def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
    """
    Args:
      prediction length: any positive integer that shows many steps to forecast
    """
    
    super().__init__(config, config_file)
    self.prediction_length = self.config.get('prediction_length', '96')
  
  def set_params(self, **params: Dict[str, Any]) -> 'TinyTimeMixerModel':
    for key, value in params.items():
      if hasattr(self, key):
        setattr(self, key, value)
    return self
         
  def predict(self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Union[pd.Series, np.ndarray] = None,
        y_context_timestamps = None,
        y_target_timestamps = None,
        **kwargs):
    #print("HUH")
    #print(y_target)
    #print("YUHUHU?")
    #print(y_target_timestamps[0].strftime('%Y-%m-%d %X'))
    #raise Exception("UNgas")
    #timestamp_strings = [ts.strftime('%Y-%m-%d %X') for ts in y_target_timestamps]
    
    # Construct DataFrame
    if len(y_context.shape) == 1:
      columns = ['1']
    else:
      columns = list(range(y_context.shape[0])) 
    df = pd.DataFrame(y_context, index=y_context_timestamps, columns=columns)
    self.ctx = len(df)
    results = self._sub_predict(df)
    if len(list(results.keys())) == 1:
      return np.array(results["1"])
    else:
      raise Exception("Multivariate for tiny time mixer has not been done yet.")
  

      
  
  def _sub_predict(self, dataframe: pd.DataFrame):
    """
    We assume dataframe is in the following format:
    Its index column is a bunch of date timestamps.
    We assume each of the rest of its columns are a different time series.
    The header of these columns should be some identifying mark distinguishing 
    the time series from each other. The actual name chosen does not matter.
    """

    forecaster = TinyTimeMixerForecaster() 

    # performs zero-shot forecasting, as default config (unchanged) is used
    forecaster.fit(dataframe, fh=list(range(1,self.prediction_length+1))) 
    y_pred = forecaster.predict() 

    all_time_series_names = dataframe.columns.values

    results_dict = dict()
    for time_series_name in all_time_series_names:
        results_dict[time_series_name] = []

    forecaster.fit(dataframe, fh=list(range(1,901))) 

    forecast = forecaster.predict() 
    
    forecast = forecast[all_time_series_names[0]].values
        
    if len(forecast.shape) == 1:
      results_dict[all_time_series_names[0]].extend(forecast)

    else:
      raise Exception("Multivariate for tiny time mixer has not been done yet.")
    
    return results_dict

    