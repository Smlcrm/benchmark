import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Mean Absolute Error.
"""
class MAE:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
        """
        Computes the MAE.
        """
        if y_true.ndim == 1:
            return np.mean(np.abs(y_true - y_pred))
        elif y_true.ndim == 2:
            return np.mean(np.abs(y_true - y_pred), axis=0)
        return np.nan
