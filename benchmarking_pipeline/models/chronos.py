import pandas as pd
import numpy as np
import torch
from chronos import ChronosPipeline
from typing import Dict, List, Optional

class ChronosForecaster:

    def __init__(
        self,
        model_size: str = "small",
        context_length: int = 64,
        num_samples: int = 10,
    ):
        """
        Initializes the Chronos model wrapper.

        Args:
            model_size (str): The size of the Chronos model to use.
                              Options: 'tiny', 'mini', 'small', 'base', 'large'.
            context_length (int): The number of past time steps the model uses as context.
            num_samples (int): The number of predictive samples to generate for each forecast.
        """
        valid_sizes = {'tiny', 'mini', 'small', 'base', 'large'}
        if model_size not in valid_sizes:
            raise ValueError(f"model_size must be one of {valid_sizes}")

        self.context_length = context_length
        self.num_samples = num_samples
        
        hf_model_name = f"amazon/chronos-t5-{model_size}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Chronos model '{hf_model_name}' to device '{device}'...")
        self.pipeline = ChronosPipeline.from_pretrained(
            hf_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("Model loaded successfully :D")

    def predict(
        self, 
        df: pd.DataFrame, 
        prediction_length: int
    ) -> Dict[str, List[float]]:
        """
        Generates forecasts for future time steps based on the most recent data.

        This method uses the last `context_length` data points from each time series
        in the DataFrame to predict the next `prediction_length` steps.

        Args:
            df (pd.DataFrame): DataFrame with a datetime index and one column per time series.
                               The model will use the end of this data as context.
            prediction_length (int): The number of future time steps to predict.

        Returns:
            Dict[str, List[float]]: A dictionary where keys are time series names (column headers)
                                    and values are the list of forecasted points.
        """
        # Create one context window for each time series
        all_contexts = []
        for series_name in df.columns:
            series_data = df[series_name].values
            # Ensure we don't take more data than available
            context_data = series_data[-self.context_length:]
            all_contexts.append(torch.tensor(context_data))

        # Generate forecasts
        all_forecasts = self.pipeline.predict(
            context=all_contexts,
            prediction_length=prediction_length,
            num_samples=self.num_samples,
        )

        # Prepare results
        results = {}
        for i, series_name in enumerate(df.columns):
            # For each series, take the median of the prediction samples
            median_forecast = np.median(all_forecasts[i], axis=0)
            results[series_name] = median_forecast.tolist()
            
        return results