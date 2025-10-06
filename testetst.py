#!/usr/bin/env python3
"""
Unit tests for SimplifiedVoltageAnalyzer
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
import os

# Add parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the class to test
from athena import SimplifiedVoltageAnalyzer


class TestVoltageAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Create temp directories for testing"""
        self.test_dir = tempfile.mkdtemp()
        self.input_dir = Path(self.test_dir) / 'input'
        self.output_dir = Path(self.test_dir) / 'output'
        self.input_dir.mkdir()
        self.analyzer = SimplifiedVoltageAnalyzer(self.input_dir, self.output_dir)
    
    def tearDown(self):
        """Clean up temp directories"""
        shutil.rmtree(self.test_dir)
    
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
    
    # Test initialization
    def test_init_creates_output_folder(self):
        """Test output folder is created"""
        self.assertTrue(self.output_dir.exists())
    
    def test_init_sets_thresholds(self):
        """Test thresholds are set correctly"""
        self.assertEqual(self.analyzer.operational_min, 18.0)
        self.assertEqual(self.analyzer.operational_max, 29.0)
        self.assertEqual(self.analyzer.deenergized_max, 2.0)
    
    # Test filename parsing
    def test_parse_filename_simple(self):
        """Test parsing simple filename"""
        filename = "unit_id=U001_test_case=baseline_test_run=1_segments.csv"
        result = self.analyzer.parse_filename(filename)
        
        self.assertEqual(result['unit_id'], 'U001')
        self.assertEqual(result['test_case'], 'baseline')
        self.assertEqual(result['test_run'], '1')
    
    def test_parse_filename_compound_values(self):
        """Test parsing filename with compound values"""
        filename = "unit_id=UNIT_001_test_case=stress_test_test_run=2_segments.csv"
        result = self.analyzer.parse_filename(filename)
        
        self.assertEqual(result['unit_id'], 'UNIT_001')
        self.assertEqual(result['test_case'], 'stress_test')
        self.assertEqual(result['test_run'], '2')
    
    # Test classification functions
    def test_is_deenergized(self):
        """Test de-energized classification"""
        voltages = np.ones(100) * 1.5  # Low voltage
        labels = np.ones(100, dtype=int)
        timestamps = np.arange(100)
        
        mask = self.analyzer.is_deenergized(voltages, labels, timestamps)
        self.assertTrue(np.all(mask))
    
    def test_is_stabilizing(self):
        """Test stabilizing classification"""
        voltages = np.linspace(5, 25, 100)  # Ramping up
        labels = np.ones(100, dtype=int)
        timestamps = np.arange(100)
        
        mask = self.analyzer.is_stabilizing(voltages, labels, timestamps, slope_cutoff=0.1)
        self.assertTrue(np.all(mask))
    
    def test_is_steadystate(self):
        """Test steady state classification"""
        voltages = np.ones(100) * 22.0 + np.random.normal(0, 0.01, 100)
        labels = np.ones(100, dtype=int)
        timestamps = np.arange(100)
        
        mask = self.analyzer.is_steadystate(voltages, labels, timestamps)
        self.assertTrue(np.all(mask))
    
    # Test segment classification
    def test_classify_segments(self):
        """Test full segment classification"""
        df = pd.DataFrame({
            'timestamp': np.arange(300),
            'voltage': np.concatenate([
                np.ones(100) * 1.0,      # De-energized
                np.linspace(2, 20, 100), # Stabilizing
                np.ones(100) * 22.0      # Steady state
            ]),
            'segment': np.repeat([0, 1, 2], 100)
        })
        
        result = self.analyzer.classify_segments(df.copy())
        
        self.assertIn('label', result.columns)
        labels = result['label'].unique()
        self.assertIn('de-energized', labels)
        self.assertIn('steady_state', labels)
    
    # Test CSV analysis
    def test_analyze_csv_valid_file(self):
        """Test analyzing valid CSV file"""
        path = self.create_test_csv(
            self.input_dir / 'tc1' / 'dc1',
            'unit_id=U001_test_case=test_test_run=1_segments.csv',
            mean_v=19.5
        )
        
        result = self.analyzer.analyze_csv(path)
        
        self.assertIsNotNone(result)
        metrics, df, grouping = result
        
        self.assertIsInstance(metrics, list)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(grouping, dict)
        
        # Check metrics
        self.assertTrue(len(metrics) > 0)
        self.assertIn('mean_voltage', metrics[0])
        self.assertIn('flagged', metrics[0])
    
    def test_analyze_csv_missing_columns(self):
        """Test handling missing columns"""
        folder = self.input_dir / 'tc1' / 'dc1'
        folder.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame({'bad_column': [1, 2, 3]})
        path = folder / 'bad.csv'
        df.to_csv(path, index=False)
        
        result = self.analyzer.analyze_csv(path)
        
        self.assertIsNone(result)
        self.assertEqual(len(self.analyzer.failed_files), 1)
    
    def test_analyze_csv_low_voltage_flag(self):
        """Test low voltage flagging"""
        path = self.create_test_csv(
            self.input_dir / 'tc1' / 'dc1',
            'test_segments.csv',
            mean_v=17.0,  # Below 18V
            std_v=0.01
        )
        
        result = self.analyzer.analyze_csv(path)
        
        if result:
            metrics, _, _ = result
            ss_metrics = [m for m in metrics if m['label'] == 'steady_state']
            if ss_metrics:
                self.assertTrue(ss_metrics[0]['flagged'])
                self.assertIn('low_voltage', ss_metrics[0]['flags'])
    
    def test_analyze_csv_high_variance_flag(self):
        """Test high variance flagging"""
        path = self.create_test_csv(
            self.input_dir / 'tc1' / 'dc1',
            'test_segments.csv',
            mean_v=22.0,
            std_v=2.0  # High variance
        )
        
        result = self.analyzer.analyze_csv(path)
        
        if result:
            metrics, _, _ = result
            ss_metrics = [m for m in metrics if m['label'] == 'steady_state']
            if ss_metrics:
                self.assertTrue(ss_metrics[0]['flagged'])
    
    # Test plots (just verify they don't crash)
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_simple_plot(self, mock_close, mock_savefig):
        """Test plot creation doesn't crash"""
        df = pd.DataFrame({
            'timestamp': np.arange(100),
            'voltage': np.random.normal(19, 0.1, 100),
            'segment': np.ones(100),
            'label': ['steady_state'] * 100
        })
        
        grouping = {'unit_id': 'U001', 'test_case': 'test', 'test_run': '1', 'dc_folder': 'dc1'}
        
        self.analyzer.create_simple_plot(df, grouping, Path('test.png'))
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_create_summary_plot(self, mock_close, mock_savefig):
        """Test summary plot creation"""
        df = pd.DataFrame([
            {'label': 'steady_state', 'flagged': True, 'mean_voltage': 17.5, 
             'variance': 1.5, 'test_case': 'tc1'}
        ])
        
        self.analyzer.create_summary_plot(df)
        
        mock_savefig.assert_called_once()
    
    # Test run_analysis
    def test_run_analysis_no_files(self):
        """Test run_analysis with no files"""
        with patch('builtins.print'):
            df = self.analyzer.run_analysis()
        
        self.assertTrue(df.empty)
    
    @patch('pandas.ExcelWriter')
    @patch.object(SimplifiedVoltageAnalyzer, 'create_summary_plot')
    def test_run_analysis_with_files(self, mock_plot, mock_excel):
        """Test run_analysis with files"""
        # Create test files
        self.create_test_csv(
            self.input_dir / 'tc1' / 'dc1',
            'test1_segments.csv'
        )
        self.create_test_csv(
            self.input_dir / 'tc1' / 'dc2',
            'test2_segments.csv'
        )
        
        with patch('builtins.print'):
            df = self.analyzer.run_analysis()
        
        self.assertFalse(df.empty)
        self.assertIn('mean_voltage', df.columns)
        self.assertIn('flagged', df.columns)
    
    # Test edge cases
    def test_empty_dataframe(self):
        """Test handling empty DataFrame"""
        df = pd.DataFrame(columns=['timestamp', 'voltage', 'segment'])
        
        result = self.analyzer.classify_segments(df)
        
        self.assertTrue(result.empty)
        self.assertIn('label', result.columns)
    
    def test_single_point(self):
        """Test handling single data point"""
        df = pd.DataFrame({
            'timestamp': [0],
            'voltage': [19.0],
            'segment': [0]
        })
        
        result = self.analyzer.classify_segments(df)
        
        self.assertEqual(len(result), 1)
        self.assertIn('label', result.columns)
    
    def test_zero_variance(self):
        """Test handling zero variance data"""
        path = self.create_test_csv(
            self.input_dir / 'tc1' / 'dc1',
            'flat_segments.csv',
            mean_v=19.0,
            std_v=0.0  # Zero variance
        )
        
        result = self.analyzer.analyze_csv(path)
        
        if result:
            metrics, _, _ = result
            for m in metrics:
                self.assertEqual(m['variance'], 0.0)
                self.assertEqual(m['std'], 0.0)


if __name__ == '__main__':
    # Run tests
    print("Running SimplifiedVoltageAnalyzer Unit Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestVoltageAnalyzer)
    
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
