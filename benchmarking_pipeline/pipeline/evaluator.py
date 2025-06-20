import pandas as pd
import numpy as np
from ..metrics.rmse import RMSE
from ..metrics.mae import MAE
from ..metrics.mean_of_maes import MeanOfMAEs
from ..metrics.mean_of_rmses import MeanOfRMSEs
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
        self.target_col_name = self.config.get('target_col', 'y_true')
        
        # Metric registry maps string names to metric class instances
        self.metric_registry = {
            'rmse': RMSE(),
            'mae': MAE(),
            'mean_of_maes': MeanOfMAEs(),
            'mean_of_rmses': MeanOfRMSEs(),
            'mase': MASE(),
            'crps': CRPS(),
            'quantile_loss': QuantileLoss(),
            'interval_score': IntervalScore()
        }

    def evaluate(self, model, data, y_train_series=None):
        """
        Evaluate model performance on given data.
        
        Args:
            model: Trained model instance with a .predict() method.
            data: Evaluation data (pd.DataFrame) containing features and the target column.
            y_train_series (pd.Series or np.array, optional): Training data for MASE.
                                                            
        Returns:
            Dictionary of evaluation metrics.
        """
        y_true_eval = data[self.target_col_name].values
        X_eval = data.drop(columns=[self.target_col_name], errors='ignore')
        
        predictions_output = model.predict(X_eval)

        # Unpack predictions
        if isinstance(predictions_output, dict):
            y_pred_eval = predictions_output.get('point')
            probabilistic_kwargs = predictions_output
        else:
            y_pred_eval = predictions_output
            probabilistic_kwargs = {}
        
        # Add y_train to kwargs for MASE
        probabilistic_kwargs['y_train'] = y_train_series

        # Add other config items to kwargs for metrics that need them
        probabilistic_kwargs['quantiles_q_values'] = self.config.get('quantiles_q_values')
        probabilistic_kwargs['interval_alpha'] = self.config.get('interval_alpha')

        results = {}
        for metric_name in self.metrics_to_calculate:
            if metric_name in self.metric_registry:
                metric_func = self.metric_registry[metric_name]
                try:
                    # For probabilistic metrics, y_pred is a placeholder and not used
                    score = metric_func(y_true_eval, y_pred_eval, **probabilistic_kwargs)
                    results[metric_name] = score
                except Exception as e:
                    print(f"Could not compute metric '{metric_name}': {e}")
                    results[metric_name] = "Error"
            else:
                print(f"Warning: Metric '{metric_name}' not recognized.")
                
        return results
