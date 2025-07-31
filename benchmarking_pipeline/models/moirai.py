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
    self.pdt = self.config.get('pdt', 4)
    self.ctx = self.config.get('ctx', 10)
    self.psz = self.config.get('psz', 8)
    self.bsz = self.config.get('bsz', 8)
    self.test = self.config.get('test', 8)
    self.num_samples = self.config.get('num_samples', 5)
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
        y_target_timestamps = None,
        **kwargs):
    #print("HUH")
    #print(y_target)
    #print("YUHUHU?")
    #print(y_target_timestamps[0].strftime('%Y-%m-%d %X'))
    #raise Exception("UNgas")
    #timestamp_strings = [ts.strftime('%Y-%m-%d %X') for ts in y_target_timestamps]
    
    # Construct DataFrame
    if len(y_target.shape) == 1:
      columns = ['1']
    else:
      columns = list(range(y_target.shape[0])) 
    df = pd.DataFrame(y_target, index=y_target_timestamps, columns=columns)
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

    # Convert it into another dataset format for dataset splitting and prediction.
    gluon_pandas_dataset = PandasDataset(dict(dataframe))

    # Last self.test elements will be the test set.
    train, test_template = split(gluon_pandas_dataset, offset=-self.test)
    test_data = test_template.generate_instances(

      # The following three comments are straight from the MoirAI example 
      # notebook comments.

      # number of time steps for each prediction
      prediction_length=self.pdt, 
      # number of windows in rolling window evaluation
      windows=self.test//self.pdt, 
      # number of time steps between each window - distance=self.pdt for non-overlapping windows
      distance=self.pdt 
    )

    # Create either a Moirai or Moirai_MoE model.
    if self.model_name == "moirai":
      model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{self.size}"),
            prediction_length=self.pdt,
            context_length=self.ctx,
            patch_size=self.psz,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=gluon_pandas_dataset.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=gluon_pandas_dataset.num_past_feat_dynamic_real,
        )
    elif self.model_name == "moirai_moe":
      model = MoiraiMoEForecast(
            module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{self.size}"),
            prediction_length=self.pdt,
            context_length=self.ctx,
            patch_size=self.psz,
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=gluon_pandas_dataset.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=gluon_pandas_dataset.num_past_feat_dynamic_real,
        )
    else:
      raise ValueError("self.model_name must have the value 'moirai' or 'moirai_moe'.")

    predictor = model.create_predictor(batch_size=self.bsz)
    forecasts = predictor.predict(test_data.input)

    forecast_it = iter(forecasts)

    results_dict = dict()
    for time_series_name in all_time_series_names:
        results_dict[time_series_name] = []
        
    for forecast in forecast_it:
        #print(fore)
        for time_series_name in all_time_series_names:
            if forecast.item_id == time_series_name:
                #print(fore.samples)
                results_dict[time_series_name].extend(np.median(forecast.samples,axis=0))
    
    return results_dict

    