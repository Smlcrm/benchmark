import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates the Interval Score (Winkler Score).
"""
class IntervalScore:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
        """
        Computes Interval Score.
        Requires 'y_pred_lower_bound', 'y_pred_upper_bound', and 'interval_alpha' in kwargs. y_pred is ignored.
        """
        lower = kwargs.get('y_pred_lower_bound')
        upper = kwargs.get('y_pred_upper_bound')
        alpha = kwargs.get('interval_alpha', 0.1)
        if lower is None or upper is None:
            raise ValueError("Lower and upper bounds must be in kwargs for IntervalScore.")
        
        width = upper - lower
        penalty_lower = (2 / alpha) * np.maximum(0, lower - y_true)
        penalty_upper = (2 / alpha) * np.maximum(0, y_true - upper)
        score = width + penalty_lower + penalty_upper
        return np.mean(score, axis=0)