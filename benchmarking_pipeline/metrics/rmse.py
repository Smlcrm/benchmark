import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Root Mean Squared Error.
"""
class RMSE:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
        """
        Computes the RMSE.
        
        Args:
            y_true: Actual observed values.
            y_pred: Predicted values.
            **kwargs: Ignored, for API consistency.

        Returns:
            The calculated RMSE score.
        """
        if y_true.ndim == 1:
            return np.sqrt(np.mean((y_true - y_pred)**2))
        elif y_true.ndim == 2:
            return np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
        return np.nan