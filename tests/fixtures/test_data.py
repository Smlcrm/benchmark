"""
Test data fixtures for benchmarking pipeline tests.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def create_sample_univariate_data(n_samples=100, start_date='2023-01-01'):
    """Create sample univariate time series data for testing."""
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start=start, periods=n_samples, freq='D')
    
    # Create realistic time series with trend and seasonality
    trend = np.linspace(0, 10, n_samples)
    seasonality = 5 * np.sin(2 * np.pi * np.arange(n_samples) / 7)  # Weekly seasonality
    noise = np.random.normal(0, 0.5, n_samples)
    
    values = trend + seasonality + noise
    
    return pd.DataFrame({
        'timestamp': dates,
        'y': values
    })


def create_sample_multivariate_data(n_samples=100, start_date='2023-01-01'):
    """Create sample multivariate time series data for testing."""
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start=start, periods=n_samples, freq='D')
    
    # Create multiple targets with different patterns
    target_1 = 10 + 0.1 * np.arange(n_samples) + 2 * np.sin(2 * np.pi * np.arange(n_samples) / 7)
    target_2 = 20 - 0.05 * np.arange(n_samples) + 3 * np.cos(2 * np.pi * np.arange(n_samples) / 30)
    
    # Create features
    feature_1 = 100 + np.random.normal(0, 10, n_samples)
    feature_2 = 50 + 0.5 * np.arange(n_samples) + np.random.normal(0, 5, n_samples)
    
    return pd.DataFrame({
        'timestamp': dates,
        'target_1': target_1,
        'target_2': target_2,
        'feature_1': feature_1,
        'feature_2': feature_2
    })


def create_sample_missing_data(n_samples=100, start_date='2023-01-01', missing_ratio=0.1):
    """Create sample data with missing values for testing."""
    data = create_sample_univariate_data(n_samples, start_date)
    
    # Randomly introduce missing values
    n_missing = int(n_samples * missing_ratio)
    missing_indices = np.random.choice(n_samples, n_missing, replace=False)
    data.loc[missing_indices, 'y'] = np.nan
    
    return data


def create_sample_irregular_data(n_samples=100, start_date='2023-01-01'):
    """Create sample data with irregular timestamps for testing."""
    start = pd.to_datetime(start_date)
    
    # Create irregular timestamps (some days missing, some with multiple observations)
    timestamps = []
    current_date = start
    
    for i in range(n_samples):
        if np.random.random() < 0.1:  # 10% chance to skip a day
            current_date += timedelta(days=2)
        elif np.random.random() < 0.05:  # 5% chance to add multiple observations
            for _ in range(np.random.randint(2, 5)):
                timestamps.append(current_date + timedelta(hours=np.random.randint(0, 24)))
            current_date += timedelta(days=1)
        else:
            timestamps.append(current_date)
            current_date += timedelta(days=1)
    
    # Create values
    values = 10 + 0.1 * np.arange(len(timestamps)) + np.random.normal(0, 0.5, len(timestamps))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'y': values
    })


def create_sample_long_horizon_data(n_samples=1000, start_date='2020-01-01'):
    """Create sample data for long-horizon forecasting tests."""
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start=start, periods=n_samples, freq='H')  # Hourly data
    
    # Create long-term patterns
    trend = 0.001 * np.arange(n_samples)
    yearly_seasonality = 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 365))
    weekly_seasonality = 2 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 7))
    daily_seasonality = 1 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    
    values = trend + yearly_seasonality + weekly_seasonality + daily_seasonality + np.random.normal(0, 0.1, n_samples)
    
    return pd.DataFrame({
        'timestamp': dates,
        'y': values
    })


def create_sample_categorical_data(n_samples=100, start_date='2023-01-01'):
    """Create sample data with categorical features for testing."""
    data = create_sample_univariate_data(n_samples, start_date)
    
    # Add categorical features
    categories = ['A', 'B', 'C', 'D']
    data['category'] = np.random.choice(categories, n_samples)
    data['binary'] = np.random.choice([0, 1], n_samples)
    
    return data


def create_sample_external_regressors(n_samples=100, start_date='2023-01-01'):
    """Create sample data with external regressors for testing."""
    data = create_sample_univariate_data(n_samples, start_date)
    
    # Add external regressors
    data['temperature'] = 20 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 2, n_samples)
    data['humidity'] = 60 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 365) + np.random.normal(0, 5, n_samples)
    data['holiday'] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])  # 10% holidays
    
    return data


def create_sample_anomaly_data(n_samples=100, start_date='2023-01-01', anomaly_ratio=0.05):
    """Create sample data with anomalies for testing."""
    data = create_sample_univariate_data(n_samples, start_date)
    
    # Introduce anomalies
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Create different types of anomalies
        if np.random.random() < 0.5:
            # Spike anomaly
            data.loc[idx, 'y'] = data.loc[idx, 'y'] + 10 * np.random.choice([-1, 1])
        else:
            # Level shift anomaly
            data.loc[idx:, 'y'] = data.loc[idx:, 'y'] + 5 * np.random.choice([-1, 1])
    
    return data


def get_test_dataset_configs():
    """Get a list of test dataset configurations."""
    return [
        {
            'name': 'univariate_simple',
            'data_func': create_sample_univariate_data,
            'params': {'n_samples': 100},
            'type': 'univariate'
        },
        {
            'name': 'multivariate_simple',
            'data_func': create_sample_multivariate_data,
            'params': {'n_samples': 100},
            'type': 'multivariate'
        },
        {
            'name': 'missing_data',
            'data_func': create_sample_missing_data,
            'params': {'n_samples': 100, 'missing_ratio': 0.1},
            'type': 'univariate'
        },
        {
            'name': 'long_horizon',
            'data_func': create_sample_long_horizon_data,
            'params': {'n_samples': 1000},
            'type': 'univariate'
        }
    ]
