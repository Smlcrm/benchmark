import os
import torch
import numpy as np
import pandas as pd

from benchmarking_pipeline.models.toto.toto.model.toto import Toto
from benchmarking_pipeline.models.toto.toto.data.util.dataset import MaskedTimeseries
from benchmarking_pipeline.models.toto.toto.inference.forecaster import TotoForecaster
from benchmarking_pipeline.models.foundation_model import FoundationModel
from typing import Optional, Union, Dict, Any


class TotoModel(FoundationModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None, prediction_length=4, num_samples=4, samples_per_batch=4):
        """
        Args:
            prediction_length (int): Number of timesteps to forecast.
            num_samples (int): Number of samples for probabilistic forecasting.
            samples_per_batch (int): Controls memory usage during inference.
        """
        super().__init__(config, config_file)
        self.prediction_length = self.config.get('prediction_length', 4)
        self.num_samples = self.config.get('num_samples', 4)
        self.samples_per_batch = self.config.get('samples_per_batch', 4)

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(self.device)

        # JIT compilation for faster inference
        self.model.compile()  
        self.forecaster = TotoForecaster(self.model.model)
    
    def set_params(self, **params: Dict[str, Any]) -> 'TotoModel':
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
        if len(y_context.shape) == 1:
            y_context = y_context.reshape(1, -1)
        input_tensor = torch.from_numpy(y_context).float()
        time_series_delta = int((y_context_timestamps[1] - y_context_timestamps[0]).total_seconds())
        return self._sub_predict(input_tensor, time_series_delta)

    def _sub_predict(self, input_series: torch.Tensor, time_interval_sec: int = 900) -> dict:
        """
        Args:
            input_series (torch.Tensor): Shape (num_series, time_steps)
            time_interval_sec (int): Interval between timesteps, default is 900s (15min)

        Returns:
            dict with keys: 'median', 'samples', 'quantile_0.1', 'quantile_0.9'
        """
        num_series, time_steps = input_series.shape
        input_series = input_series.to(self.device)

        # Dummy timestamp-related info for compatibility
        timestamp_seconds = torch.zeros_like(input_series).to(self.device)
        time_interval_seconds = torch.full((num_series,), time_interval_sec).to(self.device)

        # Construct MaskedTimeseries
        inputs = MaskedTimeseries(
            series=input_series,
            padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
            id_mask=torch.zeros_like(input_series),
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
        )

        # Forecast
        forecast = self.forecaster.forecast(
            inputs,
            prediction_length=self.prediction_length,
            num_samples=self.num_samples,
            samples_per_batch=self.samples_per_batch,
        )

        return np.array(forecast.median)