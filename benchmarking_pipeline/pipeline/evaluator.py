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
        self.target_col_name = self.config.get('target_col', 'y_true')
        
        # Metric registry maps string names to metric class instances
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
    

# Temporarily for testing purposes
class EvaluatorTest:
    def __init__(self, config=None):
        """
        Initialize evaluator test with configuration.
        """
        self.evaluator = Evaluator(config)

    def run_tests(self):
        """
        Run basic tests to validate evaluator functionality.
        """
        # Test data
        y_true = pd.Series([3, -0.5, 2, 7])
        y_predictions = pd.Series([2.5, 0.0, 2, 8])
        y_train = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])  # Example training data for MASE

        # For CRPS: shape (n_samples, n_draws)
        y_pred_dist_samples = np.array([
            [2.7, 3.1, 2.9],
            [0.1, -0.2, 0.0],
            [2.1, 1.9, 2.0],
            [7.2, 6.8, 7.0]
        ])

        # For QuantileLoss: shape (n_samples, n_quantiles)
        y_pred_quantiles = np.array([
            [2.0, 3.0],
            [0.0, 5.0],
            [2.0, 2.5],
            [7.0, 8.0]
        ])
        quantiles_q_values = [0.1, 0.9]

        # For IntervalScore: lower and upper bounds
        y_pred_lower_bound = np.array([2.0, -1.0, 1.5, 6.5])
        y_pred_upper_bound = np.array([3.5, 0.5, 1.5, 8.5])
        interval_alpha = 0.1

        self.evaluator.metrics_to_calculate = [
            'rmse', 'mae', 'mase', 'crps', 'quantile_loss', 'interval_score'
        ]

        # Evaluate
        results = self.evaluator.evaluate(
            y_predictions,
            y_true,
            y_train=y_train,
            y_pred_dist_samples=y_pred_dist_samples,
            y_pred_quantiles=y_pred_quantiles,
            quantiles_q_values=quantiles_q_values,
            y_pred_lower_bound=y_pred_lower_bound,
            y_pred_upper_bound=y_pred_upper_bound,
            interval_alpha=interval_alpha
        )

        # Print results
        print("Evaluation Results:", results)

    def run_test_default_metrics(self):
        """
        Test default metrics (mae, rmse) with minimal arguments.
        """
        y_true = pd.Series([1, 2, 3])
        y_pred = pd.Series([1, 2, 4])
        self.evaluator.metrics_to_calculate = ['mae', 'rmse']
        results = self.evaluator.evaluate(y_pred, y_true)
        assert 'mae' in results and 'rmse' in results
        print("Default metrics test passed:", results)

    def run_test_missing_required_args(self):
        """
        Test error handling for missing required arguments for certain metrics.
        """
        y_true = pd.Series([1, 2, 3])
        y_pred = pd.Series([1, 2, 4])
        self.evaluator.metrics_to_calculate = ['mase']
        try:
            self.evaluator.evaluate(y_pred, y_true)
        except ValueError as e:
            print("Caught expected error for missing y_train (MASE):", e)

        self.evaluator.metrics_to_calculate = ['crps']
        try:
            self.evaluator.evaluate(y_pred, y_true)
        except ValueError as e:
            print("Caught expected error for missing y_pred_dist_samples (CRPS):", e)

        self.evaluator.metrics_to_calculate = ['quantile_loss']
        try:
            self.evaluator.evaluate(y_pred, y_true)
        except ValueError as e:
            print("Caught expected error for missing quantile args (QuantileLoss):", e)

        self.evaluator.metrics_to_calculate = ['interval_score']
        try:
            self.evaluator.evaluate(y_pred, y_true)
        except ValueError as e:
            print("Caught expected error for missing interval bounds (IntervalScore):", e)

    def run_test_unknown_metric(self):
        """
        Test error handling for unknown metric name.
        """
        y_true = pd.Series([1, 2, 3])
        y_pred = pd.Series([1, 2, 4])
        self.evaluator.metrics_to_calculate = ['qpwegrhvf']
        try:
            self.evaluator.evaluate(y_pred, y_true)
        except ValueError as e:
            print("Caught expected error for unknown metric:", e)

    def run_test_length_mismatch(self):
        """
        Test error handling for mismatched prediction/true value lengths.
        """
        y_true = pd.Series([1, 2, 3])
        y_pred = pd.Series([1, 2])
        self.evaluator.metrics_to_calculate = ['mae']
        try:
            self.evaluator.evaluate(y_pred, y_true)
        except ValueError as e:
            print("Caught expected error for length mismatch:", e)

    def run_test_custom_config(self):
        """
        Test initialization with custom config.
        """
        config = {'metrics_to_calculate': ['mae'], 'target_col': 'target'}
        evaluator = Evaluator(config)
        assert evaluator.metrics_to_calculate == ['mae']
        assert evaluator.target_col_name == 'target'
        print("Custom config test passed.")

# Example usage:
if __name__ == "__main__":
    test = EvaluatorTest()
    test.run_tests()
    test.run_test_default_metrics()
    test.run_test_missing_required_args()
    test.run_test_unknown_metric()
    test.run_test_length_mismatch()
    test.run_test_custom_config()