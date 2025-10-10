import numpy as np
import pytest


class TestCalcSlope:
    """Test suite for calc_slope function."""
    
    def test_positive_slope(self):
        """Test with perfect linear data (slope should be 1.0)."""
        window = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        slope = calc_slope(window, window_size=5)
        assert np.isclose(slope, 1.0), "Slope should be 1.0 for linear increase"
    
    def test_negative_slope(self):
        """Test with negative slope."""
        window = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        slope = calc_slope(window, window_size=5)
        assert np.isclose(slope, -1.0), "Slope should be -1.0 for linear decrease"
    
    def test_zero_slope(self):
        """Test with flat data (slope should be 0.0)."""
        window = np.array([3.0, 3.0, 3.0, 3.0])
        slope = calc_slope(window, window_size=4)
        assert np.isclose(slope, 0.0), "Slope should be 0.0 for flat data"
    
    def test_insufficient_data(self):
        """Test with insufficient data (should return NaN)."""
        window = np.array([1.0, 2.0])
        slope = calc_slope(window, window_size=5)
        assert np.isnan(slope), "Should return NaN when window is smaller than window_size"
