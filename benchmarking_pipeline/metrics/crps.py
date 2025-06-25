import pandas as pd
import numpy as np
from typing import Dict, Any, Union

"""
Calculates Continuous Ranked Probability Score.
"""
class CRPS:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Union[float, np.ndarray]:
        """
        Computes the CRPS.
        Requires 'y_pred_dist_samples' in kwargs. y_pred is ignored.
        """
        y_pred_dist_samples = kwargs.get('y_pred_dist_samples')
        if y_pred_dist_samples is None:
            raise ValueError("y_pred_dist_samples must be provided in kwargs for CRPS.")
        
        y_s = np.asarray(y_pred_dist_samples)
        
        if y_true.ndim == 1: # Univariate
            term1 = np.mean(np.abs(y_s - y_true[:, np.newaxis]), axis=1)
            term2 = np.mean(np.abs(y_s[:, :, np.newaxis] - y_s[:, np.newaxis, :]), axis=(1, 2)) / 2
            return np.mean(term1 - term2)
            
        elif y_true.ndim == 2: # Multivariate
            crps_per_series = []
            for s_idx in range(y_true.shape[1]):
                y_t_series = y_true[:, s_idx]
                y_s_series = y_s[:, s_idx, :]
                t1 = np.mean(np.abs(y_s_series - y_t_series[:, np.newaxis]), axis=1)
                t2 = np.mean(np.abs(y_s_series[:, :, np.newaxis] - y_s_series[:, np.newaxis, :]), axis=(1,2)) / 2
                crps_per_series.append(np.mean(t1 - t2))
            return np.array(crps_per_series)
            
        return np.nan
