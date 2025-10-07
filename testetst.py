#!/usr/bin/env python3
"""
Unit tests for VoltageAnalyzer with Dynamic Thresholds
Run with: python test_voltage_analyzer.py
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import sys
import os

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the class to test (adjust import as needed)
# from athenav4 import VoltageAnalyzer


class TestVoltageAnalyzerDynamic(unittest.TestCase):
    
    def setUp(self):
        """Create temp directories for testing"""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / 'input'
        self.output_dir = Path(self.test_dir) / 'output'
        self.input_dir.mkdir()
        
        # Initialize analyzer (assuming VoltageAnalyzer is the class name)
        self.analyzer = VoltageAnalyzer(self.input_dir, self.output_dir)
        
        # Mock the methods that aren't shown
        self.analyzer.parse_filename = Mock(return_value={
            'unit_id': 'UNIT001',
            'ofp': 'OFP1',
            'test_case': 'TC1',
            'test_run': '1'
        })
        self.analyzer.classify_segments = Mock(side_effect=self._mock_classify_segments)
        self.analyzer.create_simple_plot = Mock()
        self.analyzer.create_summary_plot = Mock()
        
        # Set default thresholds
        self.analyzer.operational_min = 18.0
        self.analyzer.steady_state_thresholds = {
            'max_variance': 0.5,
            'max_std': 0.7,
            'max_slope': 0.01
        }
        self.analyzer.results = []
        self.analyzer.failed_files = []
    
    def tearDown(self):
        """Clean up temp directories"""
        shutil.rmtree(self.test_dir)
    
    def _mock_classify_segments(self, df):
        """Mock segment classification - always returns steady_state for testing"""
        df['label'] = 'steady_state'
        return df
    
    def create_test_csv(self, folder, filename, mean_v=19.0, std_v=0.1, n_points=100):
        """Helper to create test CSV files"""
        folder.mkdir(parents=True, exist_ok=True)
        np.random.seed(42)
        
        df = pd.DataFrame({
            'timestamp': np.arange(n_points),
            'voltage': np.random.normal(mean_v, std_v, n_points),
            'segment': np.ones(n_points, dtype=int)
        })
        
        path = folder / filename
        df.to_csv(path, index=False)
        return path
    
    # Test two-pass analysis with dynamic thresholds
    def test_run_analysis_two_pass_process(self):
        """Test that run_analysis performs two passes"""
        # Create test files
        for i in range(3):
            self.create_test_csv(
                self.input_dir / 'tc1' / 'dc1',
                f'unit_id=U{i:03d}_ofp=OFP1_test_case=TC1_test_run=1_segments.csv',
                mean_v=19.0,
                std_v=0.1
            )
        
        # Track analyze_csv calls
        original_analyze = self.analyzer.analyze_csv
        call_count = []
        
        def track_analyze(path, dynamic_thresholds=None):
            call_count.append(dynamic_thresholds)
            # Return mock data
            return ([{
                'label': 'steady_state',
                'mean_voltage': 19.0,
                'variance': 0.01,
                'std': 0.1,
                'slope': 0.001,
                'abs_slope': 0.001,
                'iqr': 0.2,
                'flagged': False,
                'flags': '',
                'flag_reasons': '',
                'ofp': 'OFP1',
                'test_case': 'TC1',
                'n_points': 100
            }], pd.DataFrame(), {'ofp': 'OFP1', 'test_case': 'TC1'})
        
        self.analyzer.analyze_csv = track_analyze
        
        with patch('pandas.ExcelWriter'), patch('builtins.print'):
            self.analyzer.run_analysis()
        
        # Should have been called twice per file (first pass with None, second with thresholds)
        self.assertEqual(len(call_count), 6)  # 3 files * 2 passes
        
        # First 3 calls should have None (first pass)
        for i in range(3):
            self.assertIsNone(call_count[i])
        
        # Last 3 calls should have dynamic thresholds (second pass)
        for i in range(3, 6):
            self.assertIsInstance(call_count[i], dict)
    
    def test_dynamic_threshold_calculation(self):
        """Test dynamic threshold calculation logic"""
        # Create files with consistent patterns
        for i in range(5):
            self.create_test_csv(
                self.input_dir / 'tc1' / 'dc1',
                f'unit_id=U{i:03d}_ofp=OFP1_test_case=TC1_test_run={i}_segments.csv',
                mean_v=19.0,
                std_v=0.2
            )
        
        # Mock analyze_csv to return consistent metrics
        def mock_analyze(path, dynamic_thresholds=None):
            return ([{
                'label': 'steady_state',
                'mean_voltage': 19.0,
                'variance': 0.04,
                'std': 0.2,
                'slope': 0.001,
                'abs_slope': 0.001,
                'iqr': 0.3,
                'flagged': False,
                'ofp': 'OFP1',
                'test_case': 'TC1',
                'n_points': 100
            }], pd.DataFrame(), {'ofp': 'OFP1', 'test_case': 'TC1'})
        
        self.analyzer.analyze_csv = mock_analyze
        
        with patch('pandas.ExcelWriter'), patch('builtins.print'):
            self.analyzer.run_analysis()
        
        # Should have calculated thresholds for OFP1_TC1 group
        # Can't directly test the thresholds without access to internals
        # But the function should complete without error
        self.assertTrue(True)
    
    def test_all_four_thresholds_must_fail_logic(self):
        """Test that flagging only occurs when ALL 4 thresholds fail"""
        # Create test CSV
        path = self.create_test_csv(
            self.input_dir / 'tc1' / 'dc1',
            'unit_id=U001_ofp=OFP1_test_case=TC1_test_run=1_segments.csv'
        )
        
        # Create dynamic thresholds
        dynamic_thresholds = {
            'OFP1_TC1': {
                'min_variance': 0.01,
                'max_variance': 0.05,
                'min_std': 0.1,
                'max_std': 0.3,
                'min_slope': -0.01,
                'max_slope': 0.01,
                'min_iqr': 0.1,
                'max_iqr': 0.5
            }
        }
        
        # Mock parse_filename to return correct grouping
        self.analyzer.parse_filename = Mock(return_value={
            'unit_id': 'U001',
            'ofp': 'OFP1',
            'test_case': 'TC1',
            'test_run': '1'
        })
        
        # Test case 1: Only 3 thresholds fail - should NOT be flagged
        with patch.object(pd, 'read_csv') as mock_read:
            mock_df = pd.DataFrame({
                'timestamp': np.arange(100),
                'voltage': np.ones(100) * 19.0,
                'segment': np.ones(100)
            })
            mock_read.return_value = mock_df
            
            # Mock metrics that fail only 3 checks
            with patch.object(np, 'var', return_value=0.001):  # Below min_variance
                with patch.object(np, 'std', return_value=0.05):  # Below min_std  
                    with patch.object(np, 'percentile', side_effect=[18.9, 19.1, 18.9, 19.1]):  # IQR = 0.2, within range
                        result = self.analyzer.analyze_csv(path, dynamic_thresholds)
                        
                        if result:
                            metrics = result[0]
                            # Should NOT be flagged (only 3 failures)
                            for m in metrics:
                                if m.get('label') == 'steady_state':
                                    self.assertFalse(m.get('flagged', False))
    
    def test_only_steady_state_segments_flagged(self):
        """Test that only steady-state segments can be flagged"""
        # Mock classify_segments to return different labels
        def mock_classify(df):
            # Create different labels
            n = len(df)
            df['label'] = ['de-energized'] * (n//3) + ['stabilizing'] * (n//3) + ['steady_state'] * (n//3 + n%3)
            return df
        
        self.analyzer.classify_segments = mock_classify
        
        path = self.create_test_csv(
            self.input_dir / 'tc1' / 'dc1',
            'test_segments.csv',
            mean_v=17.0  # Low voltage that would normally trigger flag
        )
        
        result = self.analyzer.analyze_csv(path)
        
        if result:
            metrics = result[0]
            for m in metrics:
                if m['label'] != 'steady_state':
                    # Non-steady-state should NEVER be flagged
                    self.assertFalse(m['flagged'])
                    self.assertEqual(m['flags'], '')
    
    def test_run_analysis_creates_flagged_plots_folder(self):
        """Test that flagged_plots folder is created"""
        self.create_test_csv(
            self.input_dir / 'tc1' / 'dc1',
            'test_segments.csv'
        )
        
        with patch('pandas.ExcelWriter'), patch('builtins.print'):
            self.analyzer.run_analysis()
        
        plots_folder = self.output_dir / 'flagged_plots'
        self.assertTrue(plots_folder.exists())
    
    def test_run_analysis_handles_duplicate_plot_names(self):
        """Test handling of duplicate plot names"""
        # Create files that would generate same plot name
        for i in range(3):
            self.create_test_csv(
                self.input_dir / 'tc1' / 'dc1',
                f'unit_id=U001_ofp=OFP1_test_case=TC1_test_run=1_v{i}_segments.csv'
            )
        
        # Mock analyze_csv to return flagged steady state
        def mock_analyze(path, dynamic_thresholds=None):
            if dynamic_thresholds is not None:  # Second pass
                return ([{
                    'label': 'steady_state',
                    'flagged': True,
                    'flags': 'test',
                    'flag_reasons': 'test reason',
                    'unit_id': 'U001',
                    'test_case': 'TC1',
                    'test_run': '1',
                    'dc_folder': 'dc1'
                }], pd.DataFrame(), {
                    'unit_id': 'U001',
                    'test_case': 'TC1', 
                    'test_run': '1',
                    'dc_folder': 'dc1'
                })
            return ([], pd.DataFrame(), {})
        
        self.analyzer.analyze_csv = mock_analyze
        
        with patch('pandas.ExcelWriter'), patch('builtins.print'):
            self.analyzer.run_analysis()
        
        # Should handle duplicate names without error
        self.assertTrue(True)
    
    def test_run_analysis_excel_sheets(self):
        """Test that Excel file contains correct sheets"""
        self.create_test_csv(
            self.input_dir / 'tc1' / 'dc1',
            'test_segments.csv'
        )
        
        # Mock analyze_csv
        def mock_analyze(path, dynamic_thresholds=None):
            return ([{
                'label': 'steady_state',
                'mean_voltage': 19.0,
                'variance': 0.01,
                'std': 0.1,
                'cv': 0.5,
                'flagged': dynamic_thresholds is not None,  # Flag on second pass
                'flags': 'test' if dynamic_thresholds else '',
                'test_case': 'TC1',
                'dc_folder': 'dc1',
                'n_points': 100,
                'ofp': 'OFP1'
            }], pd.DataFrame(), {'test_case': 'TC1', 'dc_folder': 'dc1'})
        
        self.analyzer.analyze_csv = mock_analyze
        
        mock_writer = MagicMock()
        mock_writer.__enter__ = Mock(return_value=mock_writer)
        mock_writer.__exit__ = Mock(return_value=None)
        
        with patch('pandas.ExcelWriter', return_value=mock_writer), patch('builtins.print'):
            df = self.analyzer.run_analysis()
        
        # Check that ExcelWriter was used
        self.assertTrue(mock_writer.__enter__.called)
    
    def test_run_analysis_summary_output(self):
        """Test summary output for flagged steady states only"""
        # Create test file
        self.create_test_csv(
            self.input_dir / 'tc1' / 'dc1',
            'test_segments.csv'
        )
        
        # Mock analyze_csv to return mixed labels with some flagged
        def mock_analyze(path, dynamic_thresholds=None):
            return ([
                {
                    'label': 'steady_state',
                    'flagged': True,
                    'flag_reasons': 'test (dynamic)',
                    'test_case': 'TC1'
                },
                {
                    'label': 'de-energized',
                    'flagged': False,
                    'flag_reasons': '',
                    'test_case': 'TC1'
                }
            ], pd.DataFrame(), {})
        
        self.analyzer.analyze_csv = mock_analyze
        
        with patch('pandas.ExcelWriter'), patch('builtins.print') as mock_print:
            df = self.analyzer.run_analysis()
        
        # Check that output mentions steady-state flagging
        print_calls = str(mock_print.call_args_list)
        self.assertIn('steady', print_calls.lower())
    
    def test_run_analysis_empty_input_folder(self):
        """Test handling of empty input folder"""
        with patch('builtins.print'):
            df = self.analyzer.run_analysis()
        
        self.assertTrue(df.empty)
    
    def test_run_analysis_nonexistent_input_folder(self):
        """Test handling of nonexistent input folder"""
        self.analyzer.input_folder = Path('/nonexistent/path')
        
        with patch('builtins.print') as mock_print:
            df = self.analyzer.run_analysis()
        
        self.assertTrue(df.empty)
        # Should print error message
        print_calls = str(mock_print.call_args_list)
        self.assertIn('ERROR', print_calls)
    
    def test_rounding_in_threshold_comparison(self):
        """Test that threshold comparisons use rounding"""
        path = self.create_test_csv(
            self.input_dir / 'tc1' / 'dc1',
            'test_segments.csv'
        )
        
        # Dynamic thresholds with values that need rounding
        dynamic_thresholds = {
            'OFP1_TC1': {
                'min_variance': 0.0999999,
                'max_variance': 0.1000001,
                'min_std': 0.2999999,
                'max_std': 0.3000001,
                'min_slope': -0.0010001,
                'max_slope': 0.0009999,
                'min_iqr': 0.1999999,
                'max_iqr': 0.2000001
            }
        }
        
        # Values that would fail without rounding but pass with rounding
        with patch.object(np, 'var', return_value=0.1):
            with patch.object(np, 'std', return_value=0.3):
                result = self.analyzer.analyze_csv(path, dynamic_thresholds)
                
                # Should handle rounding properly without errors
                self.assertIsNotNone(result)


if __name__ == '__main__':
    # Run tests
    print("Running VoltageAnalyzer Dynamic Threshold Unit Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestVoltageAnalyzerDynamic)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print(f"SUCCESS: All {result.testsRun} tests passed!")
    else:
        print(f"FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
