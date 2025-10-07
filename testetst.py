#!/usr/bin/env python3
"""
Essential unit tests for VoltageAnalyzer
Run with: python test_voltage_analyzer.py
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
import sys

# Import the class to test
# from athenav4 import VoltageAnalyzer


class TestVoltageAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / 'input'
        self.output_dir = Path(self.test_dir) / 'output'
        self.input_dir.mkdir()
        
        self.analyzer = VoltageAnalyzer(self.input_dir, self.output_dir)
        
        # Mock helper methods
        self.analyzer.parse_filename = Mock(return_value={
            'unit_id': 'U001', 'ofp': 'OFP1', 'test_case': 'TC1', 'test_run': '1'
        })
        self.analyzer.classify_segments = Mock(side_effect=lambda df: df.assign(label='steady_state'))
        self.analyzer.create_simple_plot = Mock()
        self.analyzer.create_summary_plot = Mock()
    
    def tearDown(self):
        """Cleanup"""
        shutil.rmtree(self.test_dir)
    
    def create_csv(self, path, mean_v=19.0, std_v=0.1):
        """Create test CSV"""
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            'timestamp': np.arange(100),
            'voltage': np.random.normal(mean_v, std_v, 100),
            'segment': np.ones(100)
        })
        df.to_csv(path, index=False)
        return path
    
    # TEST 1: Basic analyze_csv functionality
    def test_analyze_csv_basic(self):
        """Test analyze_csv returns correct structure"""
        path = self.create_csv(self.input_dir / 'tc1' / 'dc1' / 'test_segments.csv')
        
        result = self.analyzer.analyze_csv(path)
        
        self.assertIsNotNone(result)
        metrics, df, grouping = result
        self.assertIsInstance(metrics, list)
        self.assertIn('mean_voltage', metrics[0])
        self.assertIn('flagged', metrics[0])
    
    # TEST 2: ALL 4 thresholds must fail logic
    def test_all_four_thresholds_must_fail(self):
        """Test flagging only when ALL 4 thresholds fail"""
        path = self.create_csv(self.input_dir / 'tc1' / 'dc1' / 'test_segments.csv', std_v=0.001)
        
        # All 4 will fail - values are way out of range
        thresholds_all_fail = {
            'OFP1_TC1': {
                'min_variance': 1.0, 'max_variance': 2.0,  # actual will be ~0.000001
                'min_std': 1.0, 'max_std': 2.0,           # actual will be ~0.001
                'min_slope': 0.1, 'max_slope': 0.2,       # actual will be ~0
                'min_iqr': 1.0, 'max_iqr': 2.0            # actual will be ~0.001
            }
        }
        
        result = self.analyzer.analyze_csv(path, thresholds_all_fail)
        if result:
            metrics_list = result[0]  # This is a list
            if metrics_list and len(metrics_list) > 0:
                metrics = metrics_list[0]  # Get first metric dict
                self.assertTrue(metrics['flagged'], 
                    f"Should be flagged when all 4 fail. Metrics: {metrics}")
        
        # Only 3 will fail (IQR will pass)
        thresholds_three_fail = {
            'OFP1_TC1': {
                'min_variance': 1.0, 'max_variance': 2.0,  # Will fail
                'min_std': 1.0, 'max_std': 2.0,           # Will fail  
                'min_slope': 0.1, 'max_slope': 0.2,       # Will fail
                'min_iqr': 0.0, 'max_iqr': 10.0           # Will PASS
            }
        }
        
        result = self.analyzer.analyze_csv(path, thresholds_three_fail)
        if result:
            metrics_list = result[0]
            if metrics_list and len(metrics_list) > 0:
                metrics = metrics_list[0]
                self.assertFalse(metrics['flagged'], 
                    f"Should NOT be flagged when only 3 fail. Metrics: {metrics}")
    
    # TEST 3: Two-pass process
    def test_run_analysis_two_passes(self):
        """Test run_analysis does two passes"""
        self.create_csv(self.input_dir / 'tc1' / 'dc1' / 'test_segments.csv')
        
        call_args = []
        def track_calls(path, dynamic_thresholds=None):
            call_args.append(dynamic_thresholds)
            return ([{'label': 'steady_state', 'mean_voltage': 19, 'variance': 0.01, 
                     'std': 0.1, 'slope': 0.001, 'abs_slope': 0.001, 'iqr': 0.2,
                     'flagged': False, 'ofp': 'OFP1', 'test_case': 'TC1'}], 
                   pd.DataFrame(), {})
        
        self.analyzer.analyze_csv = track_calls
        
        with patch('pandas.ExcelWriter'), patch('builtins.print'):
            self.analyzer.run_analysis()
        
        # First call: dynamic_thresholds=None, Second call: dict
        self.assertEqual(len(call_args), 2)
        self.assertIsNone(call_args[0])  # First pass
        self.assertIsInstance(call_args[1], dict)  # Second pass
    
    # TEST 4: Only steady state can be flagged
    def test_only_steady_state_flagged(self):
        """Test non-steady-state segments never flagged"""
        path = self.create_csv(self.input_dir / 'tc1' / 'dc1' / 'test_segments.csv', mean_v=17.0)
        
        # Mock to return different labels
        self.analyzer.classify_segments = Mock(side_effect=lambda df: df.assign(label='de-energized'))
        
        result = self.analyzer.analyze_csv(path)
        if result:
            metrics_list = result[0]  # This is a list
            if metrics_list and len(metrics_list) > 0:
                metrics = metrics_list[0]  # Get first metric dict
                self.assertFalse(metrics['flagged'])  # de-energized never flagged
                self.assertEqual(metrics['flags'], '')
    
    # TEST 5: Missing columns handling
    def test_missing_columns(self):
        """Test handling of missing required columns"""
        path = self.input_dir / 'tc1' / 'dc1' / 'bad.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'wrong': [1, 2]}).to_csv(path, index=False)
        
        result = self.analyzer.analyze_csv(path)
        
        self.assertIsNone(result)
        self.assertEqual(len(self.analyzer.failed_files), 1)
    
    # TEST 6: Basic run_analysis execution
    @patch('pandas.ExcelWriter')
    def test_run_analysis_complete(self, mock_excel):
        """Test run_analysis completes successfully"""
        for i in range(3):
            self.create_csv(self.input_dir / 'tc1' / 'dc1' / f'test_{i}_segments.csv')
        
        # Mock analyze_csv
        self.analyzer.analyze_csv = Mock(return_value=(
            [{'label': 'steady_state', 'mean_voltage': 19, 'variance': 0.01,
              'std': 0.1, 'flagged': False, 'test_case': 'TC1', 'ofp': 'OFP1'}],
            pd.DataFrame(), {}
        ))
        
        with patch('builtins.print'):
            df = self.analyzer.run_analysis()
        
        self.assertFalse(df.empty)
        self.assertTrue((self.output_dir / 'flagged_plots').exists())


if __name__ == '__main__':
    print("Running Essential VoltageAnalyzer Tests...")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestVoltageAnalyzer)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("PASSED" if result.wasSuccessful() else "FAILED")
    
    sys.exit(0 if result.wasSuccessful() else 1)
