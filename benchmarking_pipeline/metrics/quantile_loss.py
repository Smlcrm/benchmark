import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Quantile Loss (Pinball Loss).
"""
class QuantileLoss:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Computes Quantile Loss.
        Requires 'y_pred_quantiles' and 'quantiles_q_values' in kwargs. y_pred is ignored.
        """
        y_pred_quantiles = kwargs.get('y_pred_quantiles')
        q_values = kwargs.get('quantiles_q_values')
        if y_pred_quantiles is None or q_values is None:
            raise ValueError("y_pred_quantiles and quantiles_q_values must be in kwargs for QuantileLoss.")

        losses = {}
        for i, q in enumerate(q_values):
            errors = y_true - y_pred_quantiles[..., i]
            loss = np.mean(np.maximum(q * errors, (q - 1) * errors))
            losses[f"q_{q:.2f}"] = loss
        return losses