import numpy as np
import pandas as pd
from ..pipeline.evaluator import Evaluator

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
        y_train = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])  # training data for MASE

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
        print("Evaluation Results:", results)

        # Additional test with different data
        y_true2 = pd.Series([10, 20, 30, 40])
        y_predictions2 = pd.Series([12, 18, 33, 39])
        y_train2 = pd.Series([5, 10, 15, 20, 25, 30, 35, 40])

        y_pred_dist_samples2 = np.array([
            [11, 13, 12],
            [19, 21, 20],
            [29, 31, 30],
            [41, 39, 40]
        ])

        y_pred_quantiles2 = np.array([
            [11, 13],
            [19, 21],
            [29, 31],
            [39, 41]
        ])
        quantiles_q_values2 = [0.25, 0.75]

        y_pred_lower_bound2 = np.array([9, 17, 28, 38])
        y_pred_upper_bound2 = np.array([13, 19, 34, 42])
        interval_alpha2 = 0.2

        results2 = self.evaluator.evaluate(
            y_predictions2,
            y_true2,
            y_train=y_train2,
            y_pred_dist_samples=y_pred_dist_samples2,
            y_pred_quantiles=y_pred_quantiles2,
            quantiles_q_values=quantiles_q_values2,
            y_pred_lower_bound=y_pred_lower_bound2,
            y_pred_upper_bound=y_pred_upper_bound2,
            interval_alpha=interval_alpha2
        )
        print("Second Evaluation Results:", results2)

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

if __name__ == "__main__":
    test = EvaluatorTest()
    test.run_tests()
    test.run_test_default_metrics()
    test.run_test_missing_required_args()
    test.run_test_unknown_metric()
    test.run_test_length_mismatch()
    test.run_test_custom_config()