import pandas as pd
import numpy as np
import torch
import timesfm
from typing import Dict, List
import warnings
from benchmarking_pipeline.models.foundation_model import FoundationModel

class TimesFmForecaster:
    def __init__(
        self,
        config: Dict = None,
        config_file: str = None,
    ):
        """
        Initializes the TimesFM model

        Args:
            config (dict): Configuration dictionary for the model.
            config_file (str): Optional path to a config file.
        """
        self.config = config or {}
        self.config_file = config_file

        self.model_path = self.config.get("model_path", "google/timesfm-1.0-200m")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading TimesFM model: {self.model_path} to device '{self.device}'")

        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu" if self.device == "cuda" else "cpu",
                per_core_batch_size=self.config.get("per_core_batch_size", 32),
                horizon_len=self.config.get("horizon_len", 128),
                num_layers=self.config.get("num_layers", 50),
                use_positional_embedding=self.config.get("use_positional_embedding", False),
                context_len=self.config.get("context_len", 2048),
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=self.model_path
            ),
        )
        self.context_length = self.model.context_len
        print("Model loaded successfully :D")

    def predict(
        self,
        df: pd.DataFrame,
        prediction_length: int,
    ) -> Dict[str, List[float]]:
        """
        Generates a point forecast for future time steps.

        Args:
            df (pd.DataFrame): DataFrame with historical time series data.
            prediction_length (int): The number of future time steps to predict.

        Returns:
            Dict[str, List[float]]:
                A dictionary where keys are series names and values are the
                corresponding list of point forecasted values.
        """
        # Prepare context windows
        contexts = []
        for series_name in df.columns:
            series_data = df[series_name].values
            context = series_data[-self.context_length:]

            if len(context) < self.context_length:
                padding = np.zeros(self.context_length - len(context))
                context = np.concatenate([padding, context])
                warnings.warn(
                    f"Series '{series_name}' is shorter than the context length of {self.context_length}. "
                    "It has been padded with zeros.",
                    UserWarning
                )
            contexts.append(context)

        # Convert data to a tensor for the model
        input_data = torch.tensor(np.stack(contexts), dtype=torch.float32).to(self.device)

        # Generate forecasts using the timesfm API
        point_forecasts = self.model.forecast(
            input_data,
        )  # Returns a list of numpy arrays

        results = {}
        for i, series_name in enumerate(df.columns):
            results[series_name] = point_forecasts[i].tolist()
            
        return results