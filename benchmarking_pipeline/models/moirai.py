import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from typing import Dict, Any
from benchmarking_pipeline.models.foundation_model import FoundationModel
from typing import Optional, List, Union
from einops import rearrange

class MoiraiModel(FoundationModel):

  def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
    """
    Args:
      model_name: the type of moirai model you want to use - choose from {'moirai', 'moirai_moe'}
      size: the model size - choose from {'small', 'base', 'large'}
      pdt: prediction length - any positive integer
      ctz: context length - any positive integer
      psz: patch size - choose from {"auto", 8, 16, 32, 64, 128}
      bsz: batch size - any positive integer
      test: test set length - any positive integer
      num_samples: number of samples to generate during prediction time - any positive integer
    """
    
    super().__init__(config, config_file)
    self.model_name = self.config.get('model_name', 'moirai')
    self.size = self.config.get('size', 'small')
    self.pdt = self.config.get('pdt', '4')
    self.psz = self.config.get('psz', '8')
    self.bsz = self.config.get('bsz', '8')
    self.test = self.config.get('test', '8')
    self.num_samples = self.config.get('num_samples', '5')
    self.target_col = self.config.get('target_col', 'y')
    self.is_fitted = False
  
  def set_params(self, **params: Dict[str, Any]) -> 'MoiraiModel':
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
      multivariate_values = []
      for key in results.keys():
        multivariate_values.append(results[key])
      return np.array(multivariate_values)
  

      
  
  def _sub_predict(self, dataframe: pd.DataFrame):
    """
    We assume dataframe is in the following format:
    Its index column is a bunch of date timestamps.
    We assume each of the rest of its columns are a different time series.
    The header of these columns should be some identifying mark distinguishing 
    the time series from each other. The actual name chosen does not matter.
    """

    all_time_series_names = dataframe.columns.values

    past_target_data = None
    if dataframe.shape[1] == 1:
      # Univariate
      past_target_data = dataframe["1"].to_numpy()
    else:
      past_target_data = dataframe[all_time_series_names].to_numpy().T

    # Create either a Moirai or Moirai_MoE model.
    if self.model_name == "moirai":
      model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{self.size}"),
            prediction_length=self.pdt,
            context_length=self.ctx,
            patch_size=self.psz,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    elif self.model_name == "moirai_moe":
      model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{self.size}"),
            prediction_length=self.pdt,
            context_length=self.ctx,
            patch_size=self.psz,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
    else:
      raise ValueError("self.model_name must have the value 'moirai' or 'moirai_moe'.")

    past_target = rearrange(
    torch.as_tensor(past_target_data, dtype=torch.float32), "t -> 1 t 1"
    )
    # 1s if the value is observed, 0s otherwise. Shape: (batch, time, variate)
    past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
    # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
    past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

    #print("past_target.shape:", past_target.shape)
    #print("past_observed_target.shape:", past_observed_target.shape)
    #print("past_is_pad.shape:", past_is_pad.shape)
    forecast = model(
        past_target=past_target,
        past_observed_target=past_observed_target,
        past_is_pad=past_is_pad,
    )
    
    forecasted_values = np.round(np.median(forecast[0], axis=0), decimals=4)

    results_dict = dict()
    for time_series_name in all_time_series_names:
        results_dict[time_series_name] = []
        
    if len(forecasted_values.shape) == 1:
      results_dict[all_time_series_names[0]].extend(forecasted_values)
    else:

      # This is just an index that tracks which forecast we want to add to which mapping in our results dict.
      current_forecasted_timeseries_idx = 0
      while current_forecasted_timeseries_idx < len(all_time_series_names):
        current_time_series_name = all_time_series_names[current_forecasted_timeseries_idx]
        current_forecast = forecasted_values[current_forecasted_timeseries_idx]

        results_dict[current_time_series_name].extend(current_forecast)
        current_forecasted_timeseries_idx += 1
    
    return results_dict

    