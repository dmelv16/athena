import pandas as pd
import numpy as np
from scipy.stats import linregress
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CSVAnalyzer:
    """Refactored CSV analyzer with modular methods for better testing."""
    
    def __init__(self, steady_state_thresholds: Dict[str, float]):
        self.steady_state_thresholds = steady_state_thresholds
        self.failed_files = []
    

    

    
    def calculate_basic_metrics(self, voltage_values: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistical metrics for voltage values.
        
        :param voltage_values: Array of voltage measurements
        :type voltage_values: np.ndarray
        :return: Dictionary containing statistical metrics
        :rtype: Dict[str, float]
        """
        if len(voltage_values) == 0:
            return {}
        
        mean_val = np.mean(voltage_values)
        std_val = np.std(voltage_values)
        
        return {
            'n_points': len(voltage_values),
            'mean_voltage': mean_val,
            'median_voltage': np.median(voltage_values),
            'std': std_val,
            'variance': np.var(voltage_values),
            'min_voltage': np.min(voltage_values),
            'max_voltage': np.max(voltage_values),
            'range': np.max(voltage_values) - np.min(voltage_values),
            'q1': np.percentile(voltage_values, 25),
            'q3': np.percentile(voltage_values, 75),
            'iqr': np.percentile(voltage_values, 75) - np.percentile(voltage_values, 25),
            'cv': (std_val / mean_val * 100) if mean_val != 0 else 0
        }
    
    def calculate_slope_metrics(self, voltage_values: np.ndarray) -> Dict[str, float]:
        """Calculate slope and r-squared metrics.
        
        :param voltage_values: Array of voltage measurements
        :type voltage_values: np.ndarray
        :return: Dictionary containing slope, abs_slope, and r_squared
        :rtype: Dict[str, float]
        """
        if len(voltage_values) <= 1:
            return {
                'slope': 0,
                'abs_slope': 0,
                'r_squared': 0
            }
        
        try:
            result = linregress(range(len(voltage_values)), voltage_values)
            return {
                'slope': result.slope,
                'abs_slope': abs(result.slope),
                'r_squared': result.rvalue ** 2
            }
        except Exception:
            return {
                'slope': 0,
                'abs_slope': 0,
                'r_squared': 0
            }
    
    def check_threshold(
        self, 
        value: float, 
        min_threshold: float, 
        max_threshold: float,
        metric_name: str,
        round_digits: int = 4
    ) -> Tuple[bool, Optional[str]]:
        """Check if a value is within threshold bounds.
        
        :param value: Value to check
        :type value: float
        :param min_threshold: Minimum acceptable value
        :type min_threshold: float
        :param max_threshold: Maximum acceptable value
        :type max_threshold: float
        :param metric_name: Name of metric for error message
        :type metric_name: str
        :param round_digits: Number of digits to round to, defaults to 4
        :type round_digits: int, optional
        :return: Tuple of (failed: bool, reason: Optional[str])
        :rtype: Tuple[bool, Optional[str]]
        """
        value_rounded = round(value, round_digits)
        min_rounded = round(min_threshold, round_digits)
        max_rounded = round(max_threshold, round_digits)
        
        if value_rounded < min_rounded:
            reason = f"{metric_name} {value:.3f} < {min_threshold:.3f} (dynamic)"
            return True, reason
        elif value_rounded > max_rounded:
            reason = f"{metric_name} {value:.3f} > {max_threshold:.3f} (dynamic)"
            return True, reason
        
        return False, None
    
    def check_dynamic_thresholds(
        self, 
        metrics: Dict[str, float], 
        thresholds: Dict[str, float],
        round_digits: int = 4
    ) -> Tuple[bool, List[str]]:
        """Check metrics against dynamic thresholds.
        
        :param metrics: Dictionary of calculated metrics
        :type metrics: Dict[str, float]
        :param thresholds: Dictionary of threshold values
        :type thresholds: Dict[str, float]
        :param round_digits: Number of digits to round to, defaults to 4
        :type round_digits: int, optional
        :return: Tuple of (should_flag: bool, reasons: List[str])
        :rtype: Tuple[bool, List[str]]
        """
        failed_checks = 0
        reasons = []
        
        # Check variance
        if 'min_variance' in thresholds and 'max_variance' in thresholds:
            failed, reason = self.check_threshold(
                metrics['variance'], 
                thresholds['min_variance'], 
                thresholds['max_variance'],
                'Variance',
                round_digits
            )
            if failed:
                failed_checks += 1
                reasons.append(reason)
        
        # Check std
        if 'min_std' in thresholds and 'max_std' in thresholds:
            failed, reason = self.check_threshold(
                metrics['std'], 
                thresholds['min_std'], 
                thresholds['max_std'],
                'Std',
                round_digits
            )
            if failed:
                failed_checks += 1
                reasons.append(reason)
        
        # Check slope
        if 'min_slope' in thresholds and 'max_slope' in thresholds:
            slope_rounded = round(metrics['slope'], round_digits)
            min_rounded = round(thresholds['min_slope'], round_digits)
            max_rounded = round(thresholds['max_slope'], round_digits)
            
            if slope_rounded < min_rounded or slope_rounded > max_rounded:
                failed_checks += 1
                if slope_rounded < min_rounded:
                    reasons.append(f"Slope {metrics['slope']:.4f} < {thresholds['min_slope']:.4f} (dynamic)")
                else:
                    reasons.append(f"Slope {metrics['slope']:.4f} > {thresholds['max_slope']:.4f} (dynamic)")
        
        # Check IQR
        if 'min_iqr' in thresholds and 'max_iqr' in thresholds:
            failed, reason = self.check_threshold(
                metrics['iqr'], 
                thresholds['min_iqr'], 
                thresholds['max_iqr'],
                'IQR',
                round_digits
            )
            if failed:
                failed_checks += 1
                reasons.append(reason)
        
        # Only flag if ALL 4 thresholds failed
        should_flag = failed_checks == 4
        return should_flag, reasons if should_flag else []
    
    def check_fixed_thresholds(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Check metrics against fixed thresholds.
        
        :param metrics: Dictionary of calculated metrics
        :type metrics: Dict[str, float]
        :return: Tuple of (should_flag: bool, reasons: List[str])
        :rtype: Tuple[bool, List[str]]
        """
        failed_checks = 0
        reasons = []
        
        if metrics['variance'] > self.steady_state_thresholds['max_variance']:
            failed_checks += 1
            reasons.append(f"Variance {metrics['variance']:.3f} > {self.steady_state_thresholds['max_variance']}")
        
        if metrics['std'] > self.steady_state_thresholds['max_std']:
            failed_checks += 1
            reasons.append(f"Std {metrics['std']:.3f} > {self.steady_state_thresholds['max_std']}")
        
        if metrics['abs_slope'] > self.steady_state_thresholds['max_slope']:
            failed_checks += 1
            reasons.append(f"Slope {metrics['abs_slope']:.4f} > {self.steady_state_thresholds['max_slope']}")
        
        # Flag if all 3 checks fail
        should_flag = failed_checks >= 3
        return should_flag, reasons if should_flag else []
    
    def process_label_metrics(
        self, 
        df: pd.DataFrame, 
        label: str, 
        grouping: Dict[str, str],
        dynamic_thresholds: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Optional[Dict]:
        """Process metrics for a specific label.
        
        :param df: DataFrame containing the data
        :type df: pd.DataFrame
        :param label: Label to filter by
        :type label: str
        :param grouping: Grouping information (OFP, test_case, etc.)
        :type grouping: Dict[str, str]
        :param dynamic_thresholds: Optional dynamic thresholds, defaults to None
        :type dynamic_thresholds: Optional[Dict[str, Dict[str, float]]], optional
        :return: Dictionary of metrics or None if no data
        :rtype: Optional[Dict]
        """
        label_data = df[df['label'] == label]
        voltage_values = label_data['voltage'].values
        
        if len(voltage_values) == 0:
            return None
        
        # Calculate all metrics
        metrics = {**grouping, 'label': label}
        metrics.update(self.calculate_basic_metrics(voltage_values))
        metrics.update(self.calculate_slope_metrics(voltage_values))
        
        # Only flag anomalies for steady state
        if label == 'steady_state':
            should_flag = False
            flag_reasons = []
            
            if dynamic_thresholds:
                group_key = f"{grouping.get('ofp', 'NA')}_{grouping.get('test_case', 'NA')}"
                if group_key in dynamic_thresholds:
                    should_flag, flag_reasons = self.check_dynamic_thresholds(
                        metrics, 
                        dynamic_thresholds[group_key]
                    )
            else:
                should_flag, flag_reasons = self.check_fixed_thresholds(metrics)
            
            metrics['flagged'] = should_flag
            metrics['flags'] = 'all_thresholds_failed' if should_flag else ''
            metrics['flag_reasons'] = '; '.join(flag_reasons)
        
        return metrics
    
    def analyze_csv(
        self, 
        csv_path: Path, 
        dynamic_thresholds: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Optional[Tuple[List[Dict], pd.DataFrame, Dict[str, str]]]:
        """Main method to analyze a CSV file.
        
        :param csv_path: Path to the CSV file
        :type csv_path: Path
        :param dynamic_thresholds: Optional dynamic thresholds dictionary, defaults to None
        :type dynamic_thresholds: Optional[Dict[str, Dict[str, float]]], optional
        :return: Tuple of (results, dataframe, grouping) or None if failed
        :rtype: Optional[Tuple[List[Dict], pd.DataFrame, Dict[str, str]]]
        """
        try:
            filename = csv_path.name
            grouping = self.parse_filename(filename)
            
            # Add folder info directly in this method
            parent_path = csv_path.parent
            grouping['dc_folder'] = parent_path.name
            grouping['test_case_folder'] = parent_path.parent.name
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Check required columns directly in this method
            required_columns = ['voltage', 'timestamp', 'segment']
            if not all(col in df.columns for col in required_columns):
                self.failed_files.append((filename, "Missing required columns"))
                return None
            
            # Classify segments using actual functions
            df = self.classify_segments(df)
            
            # Calculate metrics for each label type
            results = []
            for label in df['label'].unique():
                metrics = self.process_label_metrics(df, label, grouping, dynamic_thresholds)
                if metrics is not None:
                    results.append(metrics)
            
            return results, df, grouping
            
        except Exception as e:
            self.failed_files.append((csv_path.name, str(e)))
            return None
    
    # Placeholder methods (would be implemented in full class)
    def parse_filename(self, filename: str) -> Dict[str, str]:
        """Parse filename to extract grouping information.
        
        :param filename: Name of the file to parse
        :type filename: str
        :return: Dictionary containing parsed grouping information
        :rtype: Dict[str, str]
        """
        # Implementation would go here
        return {'ofp': 'example', 'test_case': 'test'}
    
    def classify_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify segments in the dataframe.
        
        :param df: DataFrame containing segment data
        :type df: pd.DataFrame
        :return: DataFrame with classified segments
        :rtype: pd.DataFrame
        """
        # Implementation would go here
        return df



# unit test

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
from csv_analyzer import CSVAnalyzer  # Adjust import based on your module name


@pytest.fixture
def analyzer():
    """Create a CSVAnalyzer instance with default thresholds."""
    thresholds = {
        'max_variance': 0.5,
        'max_std': 0.7,
        'max_slope': 0.001
    }
    return CSVAnalyzer(thresholds)


@pytest.fixture
def sample_voltage_data():
    """Create sample voltage data for testing."""
    return np.array([12.0, 12.1, 12.05, 12.08, 12.03, 12.07, 12.02, 12.06])


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'voltage': [12.0, 12.1, 12.05, 12.08, 12.03, 12.07, 12.02, 12.06],
        'timestamp': range(8),
        'segment': [1, 1, 1, 1, 2, 2, 2, 2],
        'label': ['steady_state'] * 8
    })


class TestCalculateBasicMetrics:
    """Tests for calculate_basic_metrics method."""
    
    def test_basic_metrics_normal_data(self, analyzer, sample_voltage_data):
        metrics = analyzer.calculate_basic_metrics(sample_voltage_data)
        
        assert metrics['n_points'] == 8
        assert metrics['mean_voltage'] == pytest.approx(12.05125, rel=1e-4)
        assert metrics['median_voltage'] == pytest.approx(12.055, rel=1e-4)
        assert 'std' in metrics
        assert 'variance' in metrics
        assert metrics['min_voltage'] == 12.0
        assert metrics['max_voltage'] == 12.1
        assert metrics['range'] == pytest.approx(0.1, rel=1e-4)
    
    def test_basic_metrics_empty_array(self, analyzer):
        metrics = analyzer.calculate_basic_metrics(np.array([]))
        assert metrics == {}
    
    def test_basic_metrics_single_value(self, analyzer):
        metrics = analyzer.calculate_basic_metrics(np.array([12.0]))
        
        assert metrics['n_points'] == 1
        assert metrics['mean_voltage'] == 12.0
        assert metrics['std'] == 0.0
        assert metrics['variance'] == 0.0
        assert metrics['cv'] == 0.0
    
    def test_basic_metrics_percentiles(self, analyzer):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        metrics = analyzer.calculate_basic_metrics(data)
        
        assert metrics['q1'] == pytest.approx(3.25, rel=1e-2)
        assert metrics['q3'] == pytest.approx(7.75, rel=1e-2)
        assert metrics['iqr'] == pytest.approx(4.5, rel=1e-2)
    
    def test_cv_calculation(self, analyzer):
        data = np.array([10, 12, 14, 16, 18])
        metrics = analyzer.calculate_basic_metrics(data)
        
        expected_cv = (np.std(data) / np.mean(data)) * 100
        assert metrics['cv'] == pytest.approx(expected_cv, rel=1e-4)


class TestCalculateSlopeMetrics:
    """Tests for calculate_slope_metrics method."""
    
    def test_slope_with_trend(self, analyzer):
        # Data with positive trend
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = analyzer.calculate_slope_metrics(data)
        
        assert metrics['slope'] == pytest.approx(1.0, rel=1e-4)
        assert metrics['abs_slope'] == pytest.approx(1.0, rel=1e-4)
        assert metrics['r_squared'] == pytest.approx(1.0, rel=1e-4)
    
    def test_slope_flat_data(self, analyzer):
        data = np.array([12.0, 12.0, 12.0, 12.0])
        metrics = analyzer.calculate_slope_metrics(data)
        
        assert metrics['slope'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['abs_slope'] == pytest.approx(0.0, abs=1e-10)
    
    def test_slope_single_point(self, analyzer):
        data = np.array([12.0])
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
        data = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        metrics = analyzer.calculate_slope_metrics(data)
        
        assert metrics['slope'] == pytest.approx(-1.0, rel=1e-4)
        assert metrics['abs_slope'] == pytest.approx(1.0, rel=1e-4)


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
            value=3.0,
            min_threshold=1.0,
            max_threshold=5.0,
            metric_name='IQR'
        )
        
        assert failed is False
        assert reason is None
    
    def test_value_at_boundary(self, analyzer):
        # Test at exact boundary (should pass due to rounding)
        failed, reason = analyzer.check_threshold(
            value=5.0,
            min_threshold=1.0,
            max_threshold=5.0,
            metric_name='Metric'
        )
        
        assert failed is False
    
    def test_rounding_behavior(self, analyzer):
        # Value that's very close but should round to within bounds
        failed, reason = analyzer.check_threshold(
            value=5.00001,
            min_threshold=1.0,
            max_threshold=5.0,
            metric_name='Metric',
            round_digits=4
        )
        
        assert failed is False


class TestCheckDynamicThresholds:
    """Tests for check_dynamic_thresholds method."""
    
    def test_all_thresholds_pass(self, analyzer):
        metrics = {
            'variance': 0.3,
            'std': 0.5,
            'slope': 0.0005,
            'iqr': 0.2
        }
        thresholds = {
            'min_variance': 0.1, 'max_variance': 0.5,
            'min_std': 0.2, 'max_std': 0.8,
            'min_slope': -0.001, 'max_slope': 0.001,
            'min_iqr': 0.1, 'max_iqr': 0.4
        }
        
        should_flag, reasons = analyzer.check_dynamic_thresholds(metrics, thresholds)
        
        assert should_flag is False
        assert len(reasons) == 0
    
    def test_all_four_thresholds_fail(self, analyzer):
        metrics = {
            'variance': 10.0,  # Too high
            'std': 5.0,        # Too high
            'slope': 0.1,      # Too high
            'iqr': 8.0         # Too high
        }
        thresholds = {
            'min_variance': 0.1, 'max_variance': 0.5,
            'min_std': 0.2, 'max_std': 0.8,
            'min_slope': -0.001, 'max_slope': 0.001,
            'min_iqr': 0.1, 'max_iqr': 0.4
        }
        
        should_flag, reasons = analyzer.check_dynamic_thresholds(metrics, thresholds)
        
        assert should_flag is True
        assert len(reasons) == 4
    
    def test_three_thresholds_fail_no_flag(self, analyzer):
        metrics = {
            'variance': 10.0,  # Too high
            'std': 5.0,        # Too high
            'slope': 0.1,      # Too high
            'iqr': 0.2         # OK
        }
        thresholds = {
            'min_variance': 0.1, 'max_variance': 0.5,
            'min_std': 0.2, 'max_std': 0.8,
            'min_slope': -0.001, 'max_slope': 0.001,
            'min_iqr': 0.1, 'max_iqr': 0.4
        }
        
        should_flag, reasons = analyzer.check_dynamic_thresholds(metrics, thresholds)
        
        # Should NOT flag unless all 4 fail
        assert should_flag is False
        assert len(reasons) == 0


class TestCheckFixedThresholds:
    """Tests for check_fixed_thresholds method."""
    
    def test_all_pass(self, analyzer):
        metrics = {
            'variance': 0.2,
            'std': 0.3,
            'abs_slope': 0.0005
        }
        
        should_flag, reasons = analyzer.check_fixed_thresholds(metrics)
        
        assert should_flag is False
        assert len(reasons) == 0
    
    def test_all_three_fail(self, analyzer):
        metrics = {
            'variance': 10.0,  # > 0.5
            'std': 5.0,        # > 0.7
            'abs_slope': 0.1   # > 0.001
        }
        
        should_flag, reasons = analyzer.check_fixed_thresholds(metrics)
        
        assert should_flag is True
        assert len(reasons) == 3
    
    def test_two_fail_no_flag(self, analyzer):
        metrics = {
            'variance': 10.0,   # Fails
            'std': 5.0,         # Fails
            'abs_slope': 0.0001 # Passes
        }
        
        should_flag, reasons = analyzer.check_fixed_thresholds(metrics)
        
        assert should_flag is False
        assert len(reasons) == 0


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
    """Integration tests for the full analyze_csv method."""
    
    @patch('pathlib.Path.open')
    @patch.object(CSVAnalyzer, 'parse_filename')
    @patch.object(CSVAnalyzer, 'classify_segments')
    def test_analyze_csv_success(self, mock_classify, mock_parse, mock_open, analyzer, sample_dataframe):
        # Setup mocks
        mock_parse.return_value = {'ofp': 'test', 'test_case': 'case1'}
        mock_classify.return_value = sample_dataframe
        
        csv_path = Path('/data/test_case/dc1/test.csv')
        
        with patch('pandas.read_csv', return_value=sample_dataframe):
            result = analyzer.analyze_csv(csv_path)
        
        assert result is not None
        results, df, grouping = result
        assert len(results) > 0
        assert isinstance(df, pd.DataFrame)
        # Verify folder info was added directly in analyze_csv
        assert grouping['dc_folder'] == 'dc1'
        assert grouping['test_case_folder'] == 'test_case'
    
    @patch('pandas.read_csv')
    @patch.object(CSVAnalyzer, 'parse_filename')
    def test_analyze_csv_missing_columns(self, mock_parse, mock_read, analyzer):
        mock_parse.return_value = {'ofp': 'test', 'test_case': 'case1'}
        mock_read.return_value = pd.DataFrame({'only_voltage': [1, 2, 3]})
        
        csv_path = Path('/data/test.csv')
        result = analyzer.analyze_csv(csv_path)
        
        assert result is None
        assert len(analyzer.failed_files) > 0
        # Verify the error message mentions missing columns
        assert any('Missing required columns' in error[1] for error in analyzer.failed_files)
    
    @patch.object(CSVAnalyzer, 'parse_filename')
    @patch.object(CSVAnalyzer, 'classify_segments')
    def test_analyze_csv_with_all_columns(self, mock_classify, mock_parse, analyzer, sample_dataframe):
        mock_parse.return_value = {'ofp': 'test', 'test_case': 'case1'}
        mock_classify.return_value = sample_dataframe
        
        csv_path = Path('/root/my_test_case/dc2/data.csv')
        
        with patch('pandas.read_csv', return_value=sample_dataframe):
            result = analyzer.analyze_csv(csv_path)
        
        assert result is not None
        results, df, grouping = result
        # Check that columns were validated inline
        assert 'voltage' in df.columns
        assert 'timestamp' in df.columns
        assert 'segment' in df.columns
    
    @patch.object(CSVAnalyzer, 'parse_filename')
    @patch.object(CSVAnalyzer, 'classify_segments')
    def test_analyze_csv_folder_structure(self, mock_classify, mock_parse, analyzer, sample_dataframe):
        mock_parse.return_value = {'ofp': 'ofp_123', 'test_case': 'tc_456'}
        mock_classify.return_value = sample_dataframe
        
        csv_path = Path('/base/test_case_folder/dc_folder/file.csv')
        
        with patch('pandas.read_csv', return_value=sample_dataframe):
            result = analyzer.analyze_csv(csv_path)
        
        assert result is not None
        results, df, grouping = result
        # Verify folder extraction happened inline
        assert grouping['dc_folder'] == 'dc_folder'
        assert grouping['test_case_folder'] == 'test_case_folder'
        # Verify parse_filename data is still there
        assert grouping['ofp'] == 'ofp_123'
        assert grouping['test_case'] == 'tc_456'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
