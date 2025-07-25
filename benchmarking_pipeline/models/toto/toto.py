import os
import torch
import numpy as np
import pandas as pd

from toto.model.toto import Toto
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster


class TotoModel:
    def __init__(self, prediction_length=4, num_samples=4, samples_per_batch=4):
        """
        Args:
            prediction_length (int): Number of timesteps to forecast.
            num_samples (int): Number of samples for probabilistic forecasting.
            samples_per_batch (int): Controls memory usage during inference.
        """
        self.prediction_length = prediction_length
        self.num_samples = num_samples
        self.samples_per_batch = samples_per_batch

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0').to(self.device)

        # JIT compilation for faster inference
        self.model.compile()  
        self.forecaster = TotoForecaster(self.model.model)

    def predict(self, input_series: torch.Tensor, time_interval_sec: int = 900) -> dict:
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
        timestamp_seconds = torch.zeros(input_series, time_steps).to(self.device)
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

        return {
            "median": forecast.median,
            "samples": forecast.samples,
            "quantile_0.1": forecast.quantile(0.1),
            "quantile_0.9": forecast.quantile(0.9),
        }