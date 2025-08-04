import pandas as pd
import numpy as np
from tabpfn_ts.forecaster import TimeSeriesForecaster
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from typing import List
import warnings

class TabPFNForecaster:

    def __init__(
        self,
        n_ensemble_configs: int = 32,
        device: str = 'cpu',
    ):
        """
        Initializes the TabPFN-ts model wrapper.

        Args:
            n_ensemble_configs (int): The number of ensemble configurations to use.
            device (str): The device to run the model on ('cpu' or 'cuda').
        """
        print(f"Loading TabPFN-ts model to device '{device}'...")
        classifier = TabPFNClassifier(device=device)

        self.forecaster = TimeSeriesForecaster(
            classifier,
            n_ensemble_configurations=n_ensemble_configs,
        )
        self.max_sequence_length = 1024
        print("Model loaded successfully :D")

    def predict(
        self,
        df: pd.DataFrame,
        target_col: str,
        prediction_length: int
    ) -> List[float]:
        """
        Generates a mean forecast for a single target series using other series as features.

        Note: This method uses the historical values of feature columns to predict the
        future of the target column. It does not require future values for the features.

        Args:
            df (pd.DataFrame): DataFrame with historical time series data.
            target_col (str): The name of the column you want to predict.
            prediction_length (int): The number of future time steps to predict.

        Returns:
            List[float]: A list containing the mean forecasted points for the target series.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        # Separate target from predictive features
        y_context = df[target_col].values
        X_context = df.drop(columns=[target_col]).values

        if len(y_context) > self.max_sequence_length:
            warnings.warn(
                f"Data has more than {self.max_sequence_length} points. "
                f"Using the last {self.max_sequence_length} points as context.",
                UserWarning
            )
            y_context = y_context[-self.max_sequence_length:]
            X_context = X_context[-self.max_sequence_length:, :]

        # Generate the mean forecast
        mean_forecast = self.forecaster.predict(
            X_context,
            y_context,
            n_predictions=prediction_length
        )

        return mean_forecast.tolist()