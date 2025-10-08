import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from csv_analyzer import CSVAnalyzer  # Adjust import based on your module name


@pytest.fixture
def analyzer():
    """Create a CSVAnalyzer instance with default thresholds."""
    thresholds = {
        'max_variance': 1.5,
        'max_std': 2.0,
        'max_slope': 0.5
    }
    return CSVAnalyzer(thresholds)


@pytest.fixture
def sample_voltage_data():
    """Create sample voltage data for testing - typical range 15-30V."""
    return np.array([24.0, 24.5, 24.2, 24.8, 24.3, 24.7, 24.1, 24.6])


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing - typical voltage range."""
    return pd.DataFrame({
        'voltage': [24.0, 24.5, 24.2, 24.8, 24.3, 24.7, 24.1, 24.6],
        'timestamp': range(8),
        'segment': [1, 1, 1, 1, 2, 2, 2, 2],
        'label': ['steady_state'] * 8
    })


@pytest.fixture
def steady_state_dataframe():
    """Create a DataFrame with steady state data - very stable voltage."""
    return pd.DataFrame({
        'voltage': [24.0, 24.1, 24.0, 24.1, 24.0, 24.1, 24.0, 24.1, 24.0, 24.1],
        'timestamp': range(10),
        'segment': [1] * 10,
        'label': ['steady_state'] * 10
    })


@pytest.fixture
def high_variance_dataframe():
    """Create a DataFrame with high variance data - unstable voltage."""
    return pd.DataFrame({
        'voltage': [15, 30, 18, 28, 16, 29, 17, 27, 20, 25],  # High variance ~25-30
        'timestamp': range(10),
        'segment': [1] * 10,
        'label': ['steady_state'] * 10
    })


class TestCalculateBasicMetrics:
    """Tests for calculate_basic_metrics method."""
    
    def test_basic_metrics_normal_data(self, analyzer, sample_voltage_data):
        metrics = analyzer.calculate_basic_metrics(sample_voltage_data)
        
        assert metrics['n_points'] == 8
        assert metrics['mean_voltage'] == pytest.approx(24.4, rel=1e-4)
        assert metrics['median_voltage'] == pytest.approx(24.4, rel=1e-4)
        assert 'std' in metrics
        assert 'variance' in metrics
        assert metrics['min_voltage'] == 24.0
        assert metrics['max_voltage'] == 24.8
        assert metrics['range'] == pytest.approx(0.8, rel=1e-4)
    
    def test_basic_metrics_empty_array(self, analyzer):
        metrics = analyzer.calculate_basic_metrics(np.array([]))
        assert metrics == {}
    
    def test_basic_metrics_single_value(self, analyzer):
        metrics = analyzer.calculate_basic_metrics(np.array([24.0]))
        
        assert metrics['n_points'] == 1
        assert metrics['mean_voltage'] == 24.0
        assert metrics['std'] == 0.0
        assert metrics['variance'] == 0.0
        assert metrics['cv'] == 0.0
    
    def test_basic_metrics_percentiles(self, analyzer):
        data = np.array([15, 18, 20, 22, 24, 26, 28, 30])
        metrics = analyzer.calculate_basic_metrics(data)
        
        assert metrics['q1'] == pytest.approx(19.5, rel=1e-2)
        assert metrics['q3'] == pytest.approx(27.5, rel=1e-2)
        assert metrics['iqr'] == pytest.approx(8.0, rel=1e-2)
    
    def test_cv_calculation(self, analyzer):
        data = np.array([20, 22, 24, 26, 28])
        metrics = analyzer.calculate_basic_metrics(data)
        
        expected_cv = (np.std(data) / np.mean(data)) * 100
        assert metrics['cv'] == pytest.approx(expected_cv, rel=1e-4)
    
    def test_realistic_voltage_range(self, analyzer):
        """Test with realistic voltage values in 15-30V range."""
        data = np.array([23.5, 24.0, 23.8, 24.2, 23.9, 24.1])
        metrics = analyzer.calculate_basic_metrics(data)
        
        assert 23.0 < metrics['mean_voltage'] < 25.0
        assert metrics['min_voltage'] >= 15.0
        assert metrics['max_voltage'] <= 30.0
        assert metrics['variance'] < 1.5  # Should be low for steady state


class TestCalculateSlopeMetrics:
    """Tests for calculate_slope_metrics method."""
    
    def test_slope_with_trend(self, analyzer):
        # Data with positive trend (ramping voltage)
        data = np.array([20.0, 21.0, 22.0, 23.0, 24.0])
        metrics = analyzer.calculate_slope_metrics(data)
        
        assert metrics['slope'] == pytest.approx(1.0, rel=1e-4)
        assert metrics['abs_slope'] == pytest.approx(1.0, rel=1e-4)
        assert metrics['r_squared'] == pytest.approx(1.0, rel=1e-4)
    
    def test_slope_flat_data(self, analyzer):
        data = np.array([24.0, 24.0, 24.0, 24.0])
        metrics = analyzer.calculate_slope_metrics(data)
        
        assert metrics['slope'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['abs_slope'] == pytest.approx(0.0, abs=1e-10)
    
    def test_slope_single_point(self, analyzer):
        data = np.array([24.0])
        metrics = analyzer.calculate_slope_metrics(data)
        
        assert metrics['slope'] == 0
        assert metrics['abs_slope'] == 0
        assert metrics['r_squared'] == 0
    
    def test_slope_empty_array(self, analyzer):
        data = np.array([])
        metrics = analyzer.calculate_slope_metrics(data)
        
        assert metrics['slope'] == 0
        assert metrics['abs_slope'] == 0
        assert metrics['r_squared'] == 0
    
    def test_slope_negative_trend(self, analyzer):
        data = np.array([28.0, 26.0, 24.0, 22.0, 20.0])
        metrics = analyzer.calculate_slope_metrics(data)
        
        assert metrics['slope'] == pytest.approx(-2.0, rel=1e-4)
        assert metrics['abs_slope'] == pytest.approx(2.0, rel=1e-4)
    
    def test_slope_steady_state_small_variations(self, analyzer):
        """Test that steady state has near-zero slope."""
        data = np.array([24.0, 24.1, 24.0, 24.1, 24.0, 24.1])
        metrics = analyzer.calculate_slope_metrics(data)
        
        assert abs(metrics['slope']) < 0.1  # Very small slope for steady state


class TestCheckThreshold:
    """Tests for check_threshold method."""
    
    def test_value_below_threshold(self, analyzer):
        failed, reason = analyzer.check_threshold(
            value=0.5,
            min_threshold=1.0,
            max_threshold=5.0,
            metric_name='Variance'
        )
        
        assert failed is True
        assert 'Variance' in reason
        assert '0.5' in reason
        assert '1.0' in reason
    
    def test_value_above_threshold(self, analyzer):
        failed, reason = analyzer.check_threshold(
            value=6.0,
            min_threshold=1.0,
            max_threshold=5.0,
            metric_name='Std'
        )
        
        assert failed is True
        assert 'Std' in reason
        assert '6.0' in reason
    
    def test_value_within_threshold(self, analyzer):
        failed, reason = analyzer.check_threshold(
            value=1.2,
            min_threshold=0.5,
            max_threshold=2.0,
            metric_name='IQR'
        )
        
        assert failed is False
        assert reason is None
    
    def test_value_at_boundary(self, analyzer):
        # Test at exact boundary (should pass due to rounding)
        failed, reason = analyzer.check_threshold(
            value=2.0,
            min_threshold=0.5,
            max_threshold=2.0,
            metric_name='Metric'
        )
        
        assert failed is False
    
    def test_rounding_behavior(self, analyzer):
        # Value that's very close but should round to within bounds
        failed, reason = analyzer.check_threshold(
            value=2.00001,
            min_threshold=0.5,
            max_threshold=2.0,
            metric_name='Metric',
            round_digits=4
        )
        
        assert failed is False
    
    def test_realistic_variance_threshold(self, analyzer):
        """Test with realistic variance values."""
        # Variance of 1.3 should be within threshold of 1.5
        failed, reason = analyzer.check_threshold(
            value=1.3,
            min_threshold=0.0,
            max_threshold=1.5,
            metric_name='Variance'
        )
        
        assert failed is False


class TestCheckDynamicThresholds:
    """Tests for check_dynamic_thresholds method."""
    
    def test_all_thresholds_pass(self, analyzer):
        metrics = {
            'variance': 0.8,
            'std': 1.2,
            'slope': 0.05,
            'iqr': 0.9
        }
        thresholds = {
            'min_variance': 0.1, 'max_variance': 1.5,
            'min_std': 0.2, 'max_std': 2.0,
            'min_slope': -0.5, 'max_slope': 0.5,
            'min_iqr': 0.1, 'max_iqr': 1.5
        }
        
        should_flag, reasons = analyzer.check_dynamic_thresholds(metrics, thresholds)
        
        assert should_flag is False
        assert len(reasons) == 0
    
    def test_all_four_thresholds_fail(self, analyzer):
        metrics = {
            'variance': 10.0,  # Too high
            'std': 8.0,        # Too high
            'slope': 2.0,      # Too high
            'iqr': 5.0         # Too high
        }
        thresholds = {
            'min_variance': 0.1, 'max_variance': 1.5,
            'min_std': 0.2, 'max_std': 2.0,
            'min_slope': -0.5, 'max_slope': 0.5,
            'min_iqr': 0.1, 'max_iqr': 1.5
        }
        
        should_flag, reasons = analyzer.check_dynamic_thresholds(metrics, thresholds)
        
        assert should_flag is True
        assert len(reasons) == 4
    
    def test_three_thresholds_fail_no_flag(self, analyzer):
        metrics = {
            'variance': 10.0,  # Too high
            'std': 8.0,        # Too high
            'slope': 2.0,      # Too high
            'iqr': 0.5         # OK
        }
        thresholds = {
            'min_variance': 0.1, 'max_variance': 1.5,
            'min_std': 0.2, 'max_std': 2.0,
            'min_slope': -0.5, 'max_slope': 0.5,
            'min_iqr': 0.1, 'max_iqr': 1.5
        }
        
        should_flag, reasons = analyzer.check_dynamic_thresholds(metrics, thresholds)
        
        # Should NOT flag unless all 4 fail
        assert should_flag is False
        assert len(reasons) == 0
    
    def test_realistic_steady_state_values(self, analyzer):
        """Test with realistic steady state metrics."""
        metrics = {
            'variance': 0.3,   # Low variance - good
            'std': 0.6,        # Low std - good
            'slope': 0.01,     # Near zero slope - good
            'iqr': 0.4         # Low IQR - good
        }
        thresholds = {
            'min_variance': 0.0, 'max_variance': 1.5,
            'min_std': 0.0, 'max_std': 2.0,
            'min_slope': -0.5, 'max_slope': 0.5,
            'min_iqr': 0.0, 'max_iqr': 1.5
        }
        
        should_flag, reasons = analyzer.check_dynamic_thresholds(metrics, thresholds)
        
        assert should_flag is False


class TestCheckFixedThresholds:
    """Tests for check_fixed_thresholds method."""
    
    def test_all_pass(self, analyzer):
        metrics = {
            'variance': 0.8,
            'std': 1.2,
            'abs_slope': 0.05
        }
        
        should_flag, reasons = analyzer.check_fixed_thresholds(metrics)
        
        assert should_flag is False
        assert len(reasons) == 0
    
    def test_all_three_fail(self, analyzer):
        metrics = {
            'variance': 10.0,  # > 1.5
            'std': 8.0,        # > 2.0
            'abs_slope': 2.0   # > 0.5
        }
        
        should_flag, reasons = analyzer.check_fixed_thresholds(metrics)
        
        assert should_flag is True
        assert len(reasons) == 3
    
    def test_two_fail_no_flag(self, analyzer):
        metrics = {
            'variance': 10.0,   # Fails
            'std': 8.0,         # Fails
            'abs_slope': 0.1    # Passes
        }
        
        should_flag, reasons = analyzer.check_fixed_thresholds(metrics)
        
        assert should_flag is False
        assert len(reasons) == 0
    
    def test_realistic_good_steady_state(self, analyzer):
        """Test with realistic good steady state values."""
        metrics = {
            'variance': 0.5,    # At threshold
            'std': 0.8,         # Slightly below threshold
            'abs_slope': 0.01   # Well below threshold
        }
        
        should_flag, reasons = analyzer.check_fixed_thresholds(metrics)
        
        assert should_flag is False
    
    def test_realistic_bad_steady_state(self, analyzer):
        """Test with realistic bad steady state values."""
        metrics = {
            'variance': 25.0,   # Way over threshold
            'std': 5.5,         # Way over threshold
            'abs_slope': 1.2    # Way over threshold
        }
        
        should_flag, reasons = analyzer.check_fixed_thresholds(metrics)
        
        assert should_flag is True
        assert len(reasons) == 3


class TestProcessLabelMetrics:
    """Tests for process_label_metrics method."""
    
    def test_process_steady_state_with_flags(self, analyzer, sample_dataframe):
        grouping = {'ofp': 'test_ofp', 'test_case': 'test1'}
        
        # Create data that will fail thresholds
        df = sample_dataframe.copy()
        df['voltage'] = [10, 20, 30, 40, 50, 60, 70, 80]  # High variance
        
        metrics = analyzer.process_label_metrics(df, 'steady_state', grouping)
        
        assert metrics is not None
        assert metrics['label'] == 'steady_state'
        assert 'flagged' in metrics
        assert 'flags' in metrics
        assert 'flag_reasons' in metrics
    
    def test_process_non_steady_state(self, analyzer, sample_dataframe):
        grouping = {'ofp': 'test_ofp', 'test_case': 'test1'}
        df = sample_dataframe.copy()
        df['label'] = 'ramp_up'
        
        metrics = analyzer.process_label_metrics(df, 'ramp_up', grouping)
        
        assert metrics is not None
        assert metrics['label'] == 'ramp_up'
        # Non-steady state shouldn't have flagging fields
        assert 'flagged' not in metrics or metrics.get('flagged') is False
    
    def test_process_empty_label(self, analyzer, sample_dataframe):
        grouping = {'ofp': 'test_ofp', 'test_case': 'test1'}
        df = sample_dataframe.copy()
        
        metrics = analyzer.process_label_metrics(df, 'nonexistent_label', grouping)
        
        assert metrics is None
    
    def test_process_with_dynamic_thresholds(self, analyzer, sample_dataframe):
        grouping = {'ofp': 'test_ofp', 'test_case': 'test1'}
        dynamic_thresholds = {
            'test_ofp_test1': {
                'min_variance': 0.0001,
                'max_variance': 0.001,
                'min_std': 0.01,
                'max_std': 0.05,
                'min_slope': -0.0001,
                'max_slope': 0.0001,
                'min_iqr': 0.01,
                'max_iqr': 0.05
            }
        }
        
        metrics = analyzer.process_label_metrics(
            sample_dataframe, 
            'steady_state', 
            grouping,
            dynamic_thresholds
        )
        
        assert metrics is not None
        assert 'flagged' in metrics


class TestAnalyzeCSVIntegration:
    """Integration tests - minimal, just verifying the pieces connect properly."""
    
    def test_analyze_csv_integration(self, analyzer, sample_dataframe):
        """Simple integration test to verify analyze_csv connects all methods."""
        mock_path = MagicMock(spec=Path)
        mock_path.name = 'test.csv'
        mock_path.parent.name = 'dc1'
        mock_path.parent.parent.name = 'test_case'
        
        with patch.object(analyzer, 'parse_filename', return_value={'ofp': 'test', 'test_case': 'case1'}), \
             patch.object(analyzer, 'classify_segments', return_value=sample_dataframe), \
             patch('pandas.read_csv', return_value=sample_dataframe):
            
            result = analyzer.analyze_csv(mock_path)
        
        # Just verify the method executes and returns expected structure
        assert result is not None
        assert len(result) == 3  # results, df, grouping
        results, df, grouping = result
        assert isinstance(results, list)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(grouping, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
