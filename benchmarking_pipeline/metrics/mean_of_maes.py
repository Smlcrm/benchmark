import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates the mean of MAEs for multivariate forecasts.
"""
class MeanOfMAEs:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Computes the Mean of MAEs."""
        if y_true.ndim != 2:
            return np.nan
        maes_per_series = np.mean(np.abs(y_true - y_pred), axis=0)
        return np.mean(maes_per_series)