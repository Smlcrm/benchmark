import pandas as pd
import numpy as np
from ..metrics.rmse import RMSE
from ..metrics.mae import MAE
from ..metrics.mase import MASE
from ..metrics.crps import CRPS
from ..metrics.quantile_loss import QuantileLoss
from ..metrics.interval_score import IntervalScore

"""
Model evaluation.
"""

class Evaluator:
    def __init__(self, config=None):
        """
        Initialize evaluator with configuration.
        """
        self.config = config if config is not None else {}
        self.metrics_to_calculate = self.config.get('metrics_to_calculate', ['mae', 'rmse'])
        # target_cols must be explicitly specified in dataset config - no defaults allowed
        dataset_cfg = self.config.get('dataset', {})
        self.target_cols = dataset_cfg.get('target_cols')
        if not self.target_cols:
            raise ValueError("target_cols must be defined in dataset configuration")
        
        # maps string names to metric class instances
        self.metric_registry = {
            'rmse': RMSE(),
            'mae': MAE(),
            'mase': MASE(),
            'crps': CRPS(),
            'quantile_loss': QuantileLoss(),
            'interval_score': IntervalScore()
        }

    def evaluate(self, y_predictions, y_true, y_train=None, **metric_kwargs):
        """
        Evaluate model performance on given data.
        
        Args:
            y_predictions (pd.Series or np.array): Model predictions.
            y_true (pd.Series or np.array): True target values.
            y_train (pd.Series or np.array, optional): Training target values for metrics like MASE.
            **metric_kwargs: Additional keyword arguments for metrics (e.g., y_pred_dist_samples for CRPS).

        Returns:
            Dictionary of evaluation metrics.
        """
        if isinstance(y_predictions, pd.Series):
            y_predictions = y_predictions.values
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if y_train is not None and isinstance(y_train, pd.Series):
            y_train = y_train.values
        if len(y_predictions) != len(y_true):
            raise ValueError("Length of predictions and true values must match.")

        results = {}
        for metric_name in self.metrics_to_calculate:
            if metric_name not in self.metric_registry:
                raise ValueError(f"Metric '{metric_name}' is not recognized. Available metrics: {list(self.metric_registry.keys())}")

            metric = self.metric_registry[metric_name]
            if metric_name == 'mase':
                if y_train is None:
                    raise ValueError("y_train must be provided for MASE calculation.")
                metric_value = metric(y_true, y_predictions, y_train=y_train)
            elif metric_name == 'crps':
                if 'y_pred_dist_samples' not in metric_kwargs:
                    raise ValueError("y_pred_dist_samples must be provided for CRPS calculation.")
                metric_value = metric(y_true, y_predictions, **metric_kwargs)
            elif metric_name == 'quantile_loss':
                if 'y_pred_quantiles' not in metric_kwargs or 'quantiles_q_values' not in metric_kwargs:
                    raise ValueError("y_pred_quantiles and quantiles_q_values must be provided for QuantileLoss calculation.")
                metric_value = metric(y_true, y_predictions, **metric_kwargs)
            elif metric_name == 'interval_score':
                if 'y_pred_lower_bound' not in metric_kwargs or 'y_pred_upper_bound' not in metric_kwargs:
                    raise ValueError("y_pred_lower_bound and y_pred_upper_bound must be provided for IntervalScore calculation.")
                metric_value = metric(y_true, y_predictions, **metric_kwargs)
            else:
                metric_value = metric(y_true, y_predictions)

            # If the metric returns a dict, merge it into results
            if isinstance(metric_value, dict):
                results.update(metric_value)
            else:
                results[metric_name] = metric_value

        return results
