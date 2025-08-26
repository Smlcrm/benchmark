"""
Unit tests for evaluator functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from benchmarking_pipeline.metrics.crps import crps_score
from benchmarking_pipeline.metrics.interval_score import interval_score


class TestCRPSScore:
    """Test cases for CRPS (Continuous Ranked Probability Score) functionality."""
    
    def test_crps_score_basic(self):
        """Test basic CRPS score calculation."""
        # Simple test case
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        score = crps_score(y_true, y_pred)
        assert score == 0.0  # Perfect prediction should give 0 CRPS
    
    def test_crps_score_with_uncertainty(self):
        """Test CRPS score with prediction uncertainty."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])  # Slight offset
        
        score = crps_score(y_true, y_pred)
        assert score > 0.0  # Should be positive for imperfect predictions
        assert score < 1.0  # Should be reasonable magnitude
    
    def test_crps_score_different_lengths(self):
        """Test that CRPS handles different input lengths appropriately."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0])  # Different length
        
        with pytest.raises(ValueError):
            crps_score(y_true, y_pred)
    
    def test_crps_score_empty_inputs(self):
        """Test CRPS score with empty inputs."""
        y_true = np.array([])
        y_pred = np.array([])
        
        with pytest.raises(ValueError):
            crps_score(y_true, y_pred)
    
    def test_crps_score_nan_handling(self):
        """Test CRPS score handles NaN values appropriately."""
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            crps_score(y_true, y_pred)


class TestIntervalScore:
    """Test cases for interval score functionality."""
    
    def test_interval_score_basic(self):
        """Test basic interval score calculation."""
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])
        
        score = interval_score(y_true, lower, upper)
        assert score >= 0.0  # Interval score should be non-negative
    
    def test_interval_score_perfect_coverage(self):
        """Test interval score when all true values fall within intervals."""
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        
        score = interval_score(y_true, lower, upper)
        # Should be relatively low for good coverage
        assert score < 10.0
    
    def test_interval_score_poor_coverage(self):
        """Test interval score when true values fall outside intervals."""
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([1.1, 2.1, 3.1])  # All true values below lower bound
        upper = np.array([1.2, 2.2, 3.2])
        
        score = interval_score(y_true, lower, upper)
        assert score > 0.0  # Should be positive for poor coverage
    
    def test_interval_score_invalid_bounds(self):
        """Test interval score with invalid bounds (lower > upper)."""
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([2.0, 3.0, 4.0])  # Lower > upper
        upper = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            interval_score(y_true, lower, upper)
    
    def test_interval_score_different_lengths(self):
        """Test that interval score handles different input lengths appropriately."""
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5])
        upper = np.array([1.5, 2.5])
        
        with pytest.raises(ValueError):
            interval_score(y_true, lower, upper)
    
    @pytest.mark.parametrize("alpha", [0.1, 0.05, 0.01])
    def test_interval_score_different_alpha(self, alpha):
        """Test interval score with different alpha values."""
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.5, 1.5, 2.5])
        upper = np.array([1.5, 2.5, 3.5])
        
        score = interval_score(y_true, lower, upper, alpha=alpha)
        assert score >= 0.0
        # Different alpha values should give different scores
        assert isinstance(score, (int, float))
