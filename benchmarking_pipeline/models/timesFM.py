import pandas as pd
import numpy as np
import torch
from transformers import TimesFmConfig, TimesFmForPrediction
from typing import Dict, List
import warnings

class TimesFmForecaster:
    def __init__(
        self,
        model_path: str = "google/timesfm-1.0-200m",
    ):
        """
        Initializes the TimesFM model wrapper.

        Args:
            model_path (str): The Hugging Face path to the TimesFM model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading TimesFM model: {model_path} to device '{self.device}'")

        self.config = TimesFmConfig.from_pretrained(model_path)
        self.model = TimesFmForPrediction.from_pretrained(model_path).to(self.device)
        self.context_length = self.config.context_length
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

        # Generate forecasts
        point_forecasts = self.model.forecast(
            input_data,
            prediction_length=prediction_length
        ) # Returns a list of numpy arrays

        results = {}
        for i, series_name in enumerate(df.columns):
            results[series_name] = point_forecasts[i].tolist()
            
        return results