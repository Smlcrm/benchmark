import pandas as pd
import numpy as np
from typing import List
import warnings
import os

from tabpfn import TabPFNRegressor 
from benchmarking_pipeline.models.foundation_model import FoundationModel


def make_time_features(n: int) -> pd.DataFrame:
    """
    Produce basic cyclic time features for positions 0..n-1.
    Mirrors TabPFN-TS style feature engineering for univariate forecasting.
    """
    t = np.arange(n)
    features = {
        "t": t,
        "sin_1": np.sin(2 * np.pi * t / max(1, n)),
        "cos_1": np.cos(2 * np.pi * t / max(1, n)),
        "sin_2": np.sin(4 * np.pi * t / max(1, n)),
        "cos_2": np.cos(4 * np.pi * t / max(1, n)),
    }
    return pd.DataFrame(features)


class TabPFNForecaster:

    def __init__(
        self,
        n_ensemble_configs: int = 32,  # retained for compatibility, not used directly here
        device: str = 'cpu',
        allow_large_cpu_dataset: bool = False,
    ):
        """
        Initializes a TabPFN-TS forecaster

        Args:
            n_ensemble_configs (int): Number of ensemble configurations (kept in signature).
            device (str): 'cpu' or 'cuda' for the underlying TabPFN model.
            allow_large_cpu_dataset (bool): If True, bypasses the default CPU sample limit by
                setting ignore_pretraining_limits=True. Otherwise will error if >1000 samples.
        """
        if device.startswith("cuda"):
            try:
                import torch
                if not torch.cuda.is_available():
                    warnings.warn("CUDA requested but not available; falling back to CPU.", UserWarning)
                    device = "cpu"
            except ImportError:
                warnings.warn("PyTorch not installed; assuming CPU.", UserWarning)
                device = "cpu"

        self.device = device
        self.allow_large_cpu_dataset = True
        self.max_sequence_length = 1024

        if self.device == "cpu" and self.allow_large_cpu_dataset:
            # optional convenience: set env var to mirror behavior
            os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

        print(f"Initialized TabPFN-TS-style forecaster on device '{self.device}' "
              f"(allow_large_cpu_dataset={self.allow_large_cpu_dataset})...")

    def _build_tabular(self, y_history: np.ndarray, X_history: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Given history of target and exogenous features, build tabular training matrix and target vector.
        """
        n = len(y_history)
        time_feats = make_time_features(n).reset_index(drop=True)
        if X_history is None or X_history.size == 0:
            df = time_feats
        else:
            exog_df = pd.DataFrame(X_history, index=range(n)).reset_index(drop=True)
            df = pd.concat([time_feats, exog_df], axis=1)
        X = df.to_numpy()
        y = y_history
        return X, y

    def predict(
        self,
        df: pd.DataFrame,
        target_col: str,
        prediction_length: int
    ) -> List[float]:
        """
        Multi-step forecast for the target column.

        Args:
            df (pd.DataFrame): Historical time series with target and optional features.
            target_col (str): Column to forecast.
            prediction_length (int): Horizon length.

        Returns:
            List[float]: Forecasted values.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        y_history = df[target_col].astype(float).values
        X_full = df.drop(columns=[target_col]).values  # exogenous inputs

        if len(y_history) > self.max_sequence_length:
            warnings.warn(
                f"Data has more than {self.max_sequence_length} points. "
                f"Using the last {self.max_sequence_length} points as context.",
                UserWarning
            )
            y_history = y_history[-self.max_sequence_length:]
            X_full = X_full[-self.max_sequence_length:, :]

        forecasts = []

        for step in range(prediction_length):
            # Build training data from current history
            X_train, y_train = self._build_tabular(y_history, X_full)

            if self.device == "cpu" and not self.allow_large_cpu_dataset and X_train.shape[0] > 1000:
                raise RuntimeError(
                    "Running on CPU with more than 1000 samples is disallowed by default. "
                    "Either set allow_large_cpu_dataset=True when constructing the forecaster to "
                    "bypass this (which will set ignore_pretraining_limits), or use a GPU."
                )

            # Prepare next time-step features
            next_pos = len(y_history)
            next_time_feats = make_time_features(next_pos + 1).iloc[[-1]].reset_index(drop=True)
            if X_full is None or X_full.size == 0:
                query_df = next_time_feats
            else:
                last_exog = pd.DataFrame(X_full, columns=[f"feat_{i}" for i in range(X_full.shape[1])])
                last_exog_row = last_exog.iloc[[-1]].reset_index(drop=True)
                query_df = pd.concat([next_time_feats, last_exog_row], axis=1)

            X_query = query_df.to_numpy()

            # Instantiate and use a fresh TabPFNRegressor (zero-shot / in-context)
            # Pass ignore_pretraining_limits if allowed and on CPU
            if self.device == "cpu" and self.allow_large_cpu_dataset:
                model = TabPFNRegressor(device=self.device, ignore_pretraining_limits=True)
            else:
                model = TabPFNRegressor(device=self.device)

            model.fit(X_train, y_train)  # provide context
            pred = model.predict(X_query)  # only test features passed here

            next_val = float(np.asarray(pred).ravel()[0])
            forecasts.append(next_val)

            # Append the predicted value to history (autoregressive)
            y_history = np.concatenate([y_history, [next_val]])
            if X_full is not None and X_full.size != 0:
                X_full = np.vstack([X_full, X_full[-1]])

            # Truncate history if exceeding max length
            if len(y_history) > self.max_sequence_length:
                y_history = y_history[-self.max_sequence_length:]
                if X_full is not None and X_full.size != 0:
                    X_full = X_full[-self.max_sequence_length:, :]

        return forecasts