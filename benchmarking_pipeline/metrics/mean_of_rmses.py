import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates the mean of RMSEs for multivariate forecasts.
"""
class MeanOfRMSEs:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """Computes the Mean of RMSEs."""
        if y_true.ndim != 2:
            return np.nan
        rmses_per_series = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
        return np.mean(rmses_per_series)
