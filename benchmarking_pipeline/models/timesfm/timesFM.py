import pandas as pd
import numpy as np
import torch
import timesfm
from typing import Dict, Any, Optional, List, Union
from benchmarking_pipeline.models.foundation_model import FoundationModel

class TimesFMModel(FoundationModel):
    def __init__(self, config: Dict[str, Any] = None, config_file: str = None):
        super().__init__(config, config_file)
        self.model_path = self.config.get("model_path", "google/timesfm-1.0-200m")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.per_core_batch_size = self.config.get("per_core_batch_size", 32)
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
        self.is_fitted = True

    def set_params(self, **params: Dict[str, Any]) -> 'TimesFMModel':
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            self.config[key] = value
        return self

    def predict(
        self,
        y_context: Optional[Union[pd.Series, np.ndarray]] = None,
        y_target: Union[pd.Series, np.ndarray] = None,
        y_context_timestamps=None,
        y_target_timestamps=None,
        **kwargs
    ):
        # Convert y_context to DataFrame
        if y_context is None:
            raise ValueError("y_context must be provided")
        if isinstance(y_context, pd.Series):
            df = y_context.to_frame()
        elif isinstance(y_context, np.ndarray):
            if y_context.ndim == 1:
                df = pd.DataFrame(y_context, columns=["series"])
            else:
                df = pd.DataFrame(y_context.T)
        elif isinstance(y_context, pd.DataFrame):
            df = y_context.copy()
        else:
            raise ValueError("Unsupported y_context type")

        # Prepare context windows
        contexts = []
        for series_name in df.columns:
            series_data = df[series_name].values
            context = series_data[-self.context_length:]
            if len(context) < self.context_length:
                padding = np.zeros(self.context_length - len(context))
                context = np.concatenate([padding, context])
            contexts.append(context)

        input_data = torch.tensor(np.stack(contexts), dtype=torch.float32).to(self.device)

        # Generate forecasts
        point_forecasts = self.model.forecast(input_data)
        results = {}
        for i, series_name in enumerate(df.columns):
            results[series_name] = point_forecasts[i].tolist()
        if len(results) == 1:
            return np.array(list(results.values())[0])
        else:
            return np.array(list(results.values()))