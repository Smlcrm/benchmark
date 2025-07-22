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


class TinyTimeMixer:
    def __init__(self, model_name, context_length=52, prediction_length=7, timestamp_column_name="date", split_config={}, batch_size=8):
        """
        Args:
            model_name (str): Name of the model. Choice of three: 
            {"ibm-granite/granite-timeseries-ttm-r2", "ibm-granite/granite-timeseries-ttm-r1", 
            or "ibm-research/ttm-research-r2"}
            context_length (int): Length of the context. Minimum context length is 52 (this is from testing the model).
        """
        self.model_name = model_name
        

        # Context length, Or Length of the history.
        # Currently supported values are: 512/1024/1536 for Granite-TTM-R2 and Research-Use-TTM-R2, and 512/1024 for Granite-TTM-R1

        # Shortest context length is 52.
        self.context_length = context_length

        # Granite-TTM-R2 supports forecast length upto 720 and Granite-TTM-R1 supports forecast length upto 96
        self.prediction_length = prediction_length

        self.timestamp_column_name = timestamp_column_name

        self.batch_size = batch_size
        self.split_config = split_config

        
    
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
          "target_columns": dataframe.drop(columns=[self]).columns.tolist(),
          "control_columns": [],
        }

        tsp = TimeSeriesPreprocessor(
              **column_specifiers,
              context_length=self.context_length,
              prediction_length=self.prediction_length,
              scaling=True,
              encode_categorical=False,
              scaler_type="standard",
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
        


        