import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline
from benchmarking_pipeline.models.foundation_model import FoundationModel
from typing import Dict, List, Optional, Any, Union

class ChronosModel(FoundationModel):

    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        """
        Args:
        model_size: the model size - choose from {'tiny', 'mini', 'small', 'base', 'large'}
        context_length: context length - any positive integer
        num_samples: number of samples to generate during prediction time - any positive integer
        """
        
        super().__init__(config, config_file)
        self.model_size = self.config.get('model_size', 'small')
        self.context_length = self.config.get('context_length', 8)
        self.num_samples = self.config.get('num_samples', 5)
        self.target_col = self.config.get('target_col', 'y')
        self.prediction_length = self.config.get('prediction_length', 40)
        self.is_fitted = False

    
    """
        Initializes the Chronos model wrapper.

        Args:
            model_size (str): The size of the Chronos model to use.
                              Options: 'tiny', 'mini', 'small', 'base', 'large'.
            context_length (int): The number of past time steps the model uses as context.
            num_samples (int): The number of predictive samples to generate for each forecast.
        """
    """
    def __init__(
        self,
        model_size: str = "small",
        context_length: int = 8,
        num_samples: int = 10,
    ):
        valid_sizes = {'tiny', 'mini', 'small', 'base', 'large'}
        if model_size not in valid_sizes:
            raise ValueError(f"model_size must be one of {valid_sizes}")

        self.context_length = context_length
        self.num_samples = num_samples
        
        hf_model_name = f"amazon/chronos-t5-{model_size}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Chronos model '{hf_model_name}' to device '{device}'...")
        self.pipeline = BaseChronosPipeline.from_pretrained(
            hf_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("Model loaded successfully :D")"""
    
    def set_params(self, **params: Dict[str, Any]) -> 'ChronosModel':
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

    def _sub_predict(
        self, 
        df: pd.DataFrame
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

        hf_model_name = f"amazon/chronos-t5-{self.model_size}"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Chronos model '{hf_model_name}' to device '{device}'...")
        pipeline = BaseChronosPipeline.from_pretrained(
            hf_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print("Model loaded successfully :D")

        # Create one context window for each time series
        all_contexts = []
        for series_name in df.columns:
            series_data = df[series_name].values
            # Ensure we don't take more data than available
            context_data = series_data[-self.context_length:]
            all_contexts.append(torch.tensor(context_data))

        # Generate forecasts
        all_forecasts = pipeline.predict(
            context=all_contexts,
            prediction_length=self.prediction_length,
            num_samples=self.num_samples,
        )

        # Prepare results
        results = {}
        for i, series_name in enumerate(df.columns):
            # For each series, take the median of the prediction samples
            median_forecast = np.median(all_forecasts[i], axis=0)
            results[series_name] = median_forecast.tolist()
            
        return results