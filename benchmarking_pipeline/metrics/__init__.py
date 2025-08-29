"""
Metrics package for evaluating time series forecasting models.

This package provides various evaluation metrics for assessing the performance of
time series forecasting models, including both point forecasts and probabilistic forecasts.

Available Metrics:
- crps: Continuous Ranked Probability Score for probabilistic forecasts
- interval_score: Interval score for prediction intervals
- mae: Mean Absolute Error for point forecasts
- rmse: Root Mean Square Error for point forecasts
- mape: Mean Absolute Percentage Error for point forecasts
- smape: Symmetric Mean Absolute Percentage Error for point forecasts

Usage:
    from benchmarking_pipeline.metrics import crps, interval_score
    
    # Compute CRPS for probabilistic forecasts
    crps_score = crps.compute_crps(y_true, y_pred_samples)
    
    # Compute interval score for prediction intervals
    interval_score = interval_score.compute_interval_score(y_true, lower, upper)
"""

__version__ = "1.0.0"
__description__ = "Evaluation metrics for time series forecasting models"
