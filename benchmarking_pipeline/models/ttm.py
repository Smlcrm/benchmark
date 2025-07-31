import math
import os
import tempfile

import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.integrations import INTEGRATION_TO_CALLBACK

from tsfm_public import TimeSeriesPreprocessor, TrackingCallback, count_parameters, get_datasets
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions
from benchmarking_pipeline.models.foundation_model import FoundationModel

from typing import Dict, Any, Optional, Union, List


class TinyTimeMixer(FoundationModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Args:
            model_name (str): Name of the model. Choice of three: 
            {"ibm-granite/granite-timeseries-ttm-r2", "ibm-granite/granite-timeseries-ttm-r1", 
            or "ibm-research/ttm-research-r2"}
            context_length (int): Length of the context. Minimum context length is 52 (this is from testing the model).
        """

        super().__init__(config, config_file)
        self.model_name = self.config.get('model_name', 'ibm-granite/granite-timeseries-ttm-r2')
        

        # Context length, Or Length of the history.
        # Currently supported values are: 512/1024/1536 for Granite-TTM-R2 and Research-Use-TTM-R2, and 512/1024 for Granite-TTM-R1

        # Shortest context length is 52.
        self.context_length = self.config.get('context_length', 52)

        # Granite-TTM-R2 supports forecast length upto 720 and Granite-TTM-R1 supports forecast length up to 96
        self.prediction_length = self.config.get('prediction_length', 7)

        self.timestamp_column_name = self.config.get('timestamp_column_name', 'date')

        self.batch_size = self.config.get('batch_size', 8)
        self.split_config = self.config.get('split_config', {})
    
    def set_params(self, **params: Dict[str, Any]) -> 'TinyTimeMixer':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

        
    def predict(self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Union[pd.Series, np.ndarray] = None,
        y_target_timestamps: List[Any] = None,
        **kwargs):
        #print("HUH")
        #print(y_target)
        #print("YUHUHU?")
        #print(y_target_timestamps[0].strftime('%Y-%m-%d %X'))
        #raise Exception("UNgas")
        #timestamp_strings = [ts.strftime('%Y-%m-%d %X') for ts in y_target_timestamps]
        
        
        # We add two temporary values to y_target, and two temporary timestamps 
        # to y_target_timestamps. We do this step because get_datasets demands 
        # we specify a train, val,and test split. Thus, we add two fake elements
        #  - the first for train, and the second for val. Since we don't use 
        # these elements anyways, it doesn't matter what goes in these first 
        # two slots.

        """
        y_target = np.concatenate( (np.array([0,0]), y_target) )

        delta = y_target_timestamps[1] - y_target_timestamps[0]
        y_target_timestamps = [
            y_target_timestamps[0] - 2 * delta,
            y_target_timestamps[0] - delta,
        ] + y_target_timestamps"""
        
        if len(y_target.shape) == 1:
            columns = ['1']
        else:
            columns = list(range(y_target.shape[0])) 
        df = pd.DataFrame(y_target, index=y_target_timestamps, columns=columns)
        df.index.rename("date", inplace=True)
        df.reset_index(inplace=True)
        return self._sub_predict(df)
        
    
    def _sub_predict(self, dataframe : pd.DataFrame):
        """
        We assume the dataframe looks as follows:
        It has one column called 'date', which contains all the 
        date timestamps for each row. Every other row contains a time series. The
        names of these rows are the names of the time series. The actual time series
        names don't matter too much. The index itself does not matter too much.
        """

        column_specifiers = {
          "timestamp_column": self.timestamp_column_name,
          "id_columns": [],
          "target_columns": dataframe.drop(columns=[self.timestamp_column_name]).columns.tolist(),
          "control_columns": [],
        }

        inferred_freq = pd.infer_freq(dataframe[self.timestamp_column_name])

        tsp = TimeSeriesPreprocessor(
              **column_specifiers,
              context_length=self.context_length,
              prediction_length=self.prediction_length,
              scaling=True,
              encode_categorical=False,
              scaler_type="standard",
              freq=inferred_freq
        )

        # Obtain model
        zeroshot_model = get_model(
              self.model_name,
              context_length=self.context_length,
              prediction_length=self.prediction_length,
              freq_prefix_tuning=False,
              freq=None,
              prefer_l1_loss=False,
              prefer_longer_context=True,
        )
        
        dset_train, dset_valid, dset_test = get_datasets(
              tsp, dataframe, self.split_config, use_frequency_token=zeroshot_model.config.resolution_prefix_tuning
        )

        # Get model trainer
        zeroshot_trainer = Trainer(
              model=zeroshot_model,
              args=TrainingArguments(
                  per_device_eval_batch_size=self.batch_size,
                  seed=42,
                  report_to="none",
              ),
        )
        # evaluate = zero-shot performance
        #print("+" * 20, "Test MSE zero-shot", "+" * 20)

        # Evaluation
        zeroshot_output = zeroshot_trainer.evaluate(dset_test)
        #print(zeroshot_output)

        # Get predictions
        predictions_dict = zeroshot_trainer.predict(dset_test)

        predictions_np = predictions_dict.predictions[0]

        return(np.transpose(predictions_np[0]))
        


        