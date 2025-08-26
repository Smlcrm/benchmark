"""
Unit tests for evaluator functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from benchmarking_pipeline.metrics.crps import CRPS
from benchmarking_pipeline.metrics.interval_score import IntervalScore
from benchmarking_pipeline.metrics.mae import MAE
from benchmarking_pipeline.metrics.rmse import RMSE


class TestCRPSScore:
    """Test cases for CRPS (Continuous Ranked Probability Score) functionality."""
    
    @pytest.mark.unit
    def test_crps_score_basic(self):
        """Test basic CRPS score calculation."""
        crps = CRPS()
        
        # Simple test case
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        # Generate samples: (n_samples, n_predictions)
        y_pred_dist_samples = np.random.normal(y_pred, 0.1, (100, 3))
        
        score = crps(y_true, y_pred, y_pred_dist_samples=y_pred_dist_samples)
        assert isinstance(score, np.ndarray)
        assert len(score) == 3
    
    @pytest.mark.unit
    def test_crps_score_with_uncertainty(self):
        """Test CRPS score with uncertainty in predictions."""
        crps = CRPS()
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        # High uncertainty
        y_pred_dist_samples = np.random.normal(y_pred, 1.0, (100, 3))
        
        score = crps(y_true, y_pred, y_pred_dist_samples=y_pred_dist_samples)
        assert isinstance(score, np.ndarray)
        assert len(score) == 3
        # Higher uncertainty should lead to higher CRPS scores
        assert np.all(score > 0)
    
    @pytest.mark.unit
    def test_crps_score_missing_samples(self):
        """Test CRPS score with missing samples."""
        crps = CRPS()
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        # Some samples are NaN
        y_pred_dist_samples = np.random.normal(y_pred, 0.1, (100, 3))
        y_pred_dist_samples[0, 0] = np.nan
        
        score = crps(y_true, y_pred, y_pred_dist_samples=y_pred_dist_samples)
        assert isinstance(score, np.ndarray)
        assert len(score) == 3
    
    @pytest.mark.unit
    def test_crps_score_multivariate(self):
        """Test CRPS score with multivariate data."""
        crps = CRPS()
        
        y_true = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        # Create simple samples manually to avoid numpy broadcasting issues
        # Shape: (n_timesteps, n_targets, n_samples)
        y_pred_dist_samples = np.zeros((3, 2, 100))
        for i in range(3):
            for j in range(2):
                y_pred_dist_samples[i, j, :] = np.random.normal(y_pred[i, j], 0.1, 100)
        
        score = crps(y_true, y_pred, y_pred_dist_samples=y_pred_dist_samples)
        assert isinstance(score, np.ndarray)
        assert len(score) == 2  # One score per series


class TestIntervalScore:
    """Test cases for Interval Score functionality."""
    
    @pytest.mark.unit
    def test_interval_score_basic(self):
        """Test basic interval score calculation."""
        interval_score = IntervalScore()
        
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])
        
        score = interval_score(y_true, lower, upper)
        assert isinstance(score, np.ndarray)
        assert len(score) == 3
    
    @pytest.mark.unit
    def test_interval_score_perfect_coverage(self):
        """Test interval score with perfect coverage."""
        interval_score = IntervalScore()
        
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5]
        )
        
        score = interval_score(y_true, lower, upper)
        # Perfect coverage should give lower scores
        assert np.all(score >= 0)
    
    @pytest.mark.unit
    def test_interval_score_poor_coverage(self):
        """Test interval score with poor coverage."""
        interval_score = IntervalScore()
        
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([2.0, 3.0, 4.0])  # All values above true
        upper = np.array([3.0, 4.0, 5.0])
        
        score = interval_score(y_true, lower, upper)
        # Poor coverage should give higher scores
        assert np.all(score > 0)
    
    @pytest.mark.unit
    def test_interval_score_missing_bounds(self):
        """Test interval score with missing bounds."""
        interval_score = IntervalScore()
        
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, np.nan, 2.5])
        upper = np.array([1.5, 2.5, np.nan])
        
        score = interval_score(y_true, lower, upper)
        assert isinstance(score, np.ndarray)
        assert len(score) == 3
    
    @pytest.mark.parametrize("alpha", [0.1, 0.05, 0.01])
    @pytest.mark.unit
    def test_interval_score_different_alpha(self, alpha):
        """Test interval score with different alpha values."""
        interval_score = IntervalScore(alpha=alpha)
        
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])
        
        score = interval_score(y_true, lower, upper)
        assert isinstance(score, np.ndarray)
        assert len(score) == 3


class TestMAE:
    """Test cases for MAE (Mean Absolute Error) functionality."""
    
    @pytest.mark.unit
    def test_mae_basic(self):
        """Test basic MAE calculation."""
        mae = MAE()
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        score = mae(y_true, y_pred)
        assert score == 0.0  # Perfect predictions
    
    @pytest.mark.unit
    def test_mae_with_error(self):
        """Test MAE calculation with prediction errors."""
        mae = MAE()
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])  # Off by 0.5
        
        score = mae(y_true, y_pred)
        assert score == 0.5  # Average absolute error


class TestRMSE:
    """Test cases for RMSE (Root Mean Square Error) functionality."""
    
    @pytest.mark.unit
    def test_rmse_basic(self):
        """Test basic RMSE calculation."""
        rmse = RMSE()
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        score = rmse(y_true, y_pred)
        assert score == 0.0  # Perfect predictions
    
    @pytest.mark.unit
    def test_rmse_with_error(self):
        """Test RMSE calculation with prediction errors."""
        rmse = RMSE()
        
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])  # Off by 0.5
        
        score = rmse(y_true, y_pred)
        assert score == 0.5  # Root mean square error
