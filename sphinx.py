"""
Voltage Analysis Module

This module provides comprehensive voltage analysis capabilities with cluster-based
segment classification and anomaly detection for electrical systems.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from scipy.stats import zscore, linregress
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class SimplifiedVoltageAnalyzer:
    """
    Voltage analyzer using cluster-based classification with comprehensive anomaly detection.
    
    This analyzer processes voltage time series data from segmented CSV files,
    classifies segments into operational states (de-energized, stabilizing, 
    steady-state, unidentified), and flags anomalies based on statistical thresholds.
    
    Attributes
    ----------
    input_folder : pathlib.Path
        Directory containing CSV files organized in test_case/dc[1,2] structure
    output_folder : pathlib.Path
        Directory for analysis outputs (plots, Excel reports)
    deenergized_max : float
        Maximum voltage threshold for de-energized state (default: 2.0V)
    operational_min : float
        Minimum voltage for operational state (default: 18.0V)
    operational_max : float
        Maximum voltage for normal operation (default: 29.0V)
    steady_state_thresholds : dict
        Thresholds for steady-state anomaly detection:
        - max_variance: Maximum allowed variance (default: 1.0)
        - max_std: Maximum allowed standard deviation (default: 1.0)
        - max_slope: Maximum allowed slope magnitude (default: 0.05)
        - outlier_threshold: Z-score threshold (default: 3, currently unused)
    results : list
        Accumulated analysis results across all files
    failed_files : list
        List of tuples (filename, error_message) for failed analyses
        
    Examples
    --------
    >>> analyzer = SimplifiedVoltageAnalyzer('data/segments', 'results/')
    >>> df = analyzer.run_analysis()
    >>> print(f"Analyzed {len(df)} segments, found {df['flagged'].sum()} anomalies")
    
    Notes
    -----
    The analyzer expects CSV files with columns: 'voltage', 'timestamp', 'segment'
    Files should follow naming convention with key=value pairs separated by underscores
    """
    
    def __init__(self, input_folder='greedygaussv4', output_folder='voltage_analysis'):
        """
        Initialize the voltage analyzer.
        
        Parameters
        ----------
        input_folder : str or pathlib.Path, optional
            Path to input directory containing CSV files (default: 'greedygaussv4')
        output_folder : str or pathlib.Path, optional
            Path to output directory for results (default: 'voltage_analysis')
            
        Side Effects
        ------------
        Creates output_folder if it doesn't exist
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Voltage thresholds for outlier detection
        self.deenergized_max = 2.0  
        self.operational_min = 18.0  
        self.operational_max = 29.0
        
        # Only flag anomalies in steady state
        self.steady_state_thresholds = {
            'max_variance': 1.0,
            'max_std': 1.0,
            'max_slope': 0.05,
            'outlier_threshold': 3  # z-score for steady state only
        }
        
        self.results = []
        self.failed_files = []
    
    def parse_filename(self, filename):
        """
        Parse filename to extract metadata using key=value format.
        
        Handles multi-word keys with underscores by temporarily replacing
        known compound keys, splitting on underscores, then restoring.
        
        Parameters
        ----------
        filename : str
            CSV filename to parse (e.g., 'unit_id=UNIT001_test_case=baseline_test_run=1_segments.csv')
            
        Returns
        -------
        dict
            Parsed metadata with keys like 'unit_id', 'test_case', 'test_run', etc.
            
        Examples
        --------
        >>> analyzer = SimplifiedVoltageAnalyzer()
        >>> metadata = analyzer.parse_filename('unit_id=U001_test_case=stress_test_run=2_segments.csv')
        >>> print(metadata)
        {'unit_id': 'U001', 'test_case': 'stress', 'test_run': '2'}
        
        Notes
        -----
        Handles compound keys like 'unit_id', 'test_case', 'test_run' that contain underscores
        """
        # Remove .csv and _segments
        base = filename.replace('.csv', '').replace('_segments', '')
        
        # Handle known multi-word keys
        base = base.replace('unit_id=', 'unitid=')
        base = base.replace('test_case=', 'testcase=')
        base = base.replace('test_run=', 'testrun=')
        
        parts = {}
        segments = base.split('_')
        
        current_key = None
        current_value = []
        
        for segment in segments:
            if '=' in segment:
                # Save previous if exists
                if current_key and current_value:
                    parts[current_key] = '_'.join(current_value)
                
                key, value = segment.split('=', 1)
                current_key = key
                current_value = [value] if value else []
            else:
                if current_key:
                    current_value.append(segment)
        
        # Save last pair
        if current_key and current_value:
            parts[current_key] = '_'.join(current_value)
        
        # Restore original names
        if 'unitid' in parts:
            parts['unit_id'] = parts.pop('unitid')
        if 'testcase' in parts:
            parts['test_case'] = parts.pop('testcase')
        if 'testrun' in parts:
            parts['test_run'] = parts.pop('testrun')
        
        return parts
    
    def is_deenergized(self, voltages, labels, timestamps, slope_threshold=0.1, mean_threshold=5):
        """
        Identify de-energized segments using cluster analysis.
        
        A segment is considered de-energized if it has low mean voltage and 
        minimal slope. Also checks for de-energized clusters between other
        de-energized clusters.
        
        Parameters
        ----------
        voltages : numpy.ndarray
            Array of voltage values
        labels : numpy.ndarray
            Array of segment labels/cluster IDs
        timestamps : numpy.ndarray
            Array of timestamp values
        slope_threshold : float, optional
            Maximum absolute slope for de-energized state (default: 0.1)
        mean_threshold : float, optional
            Maximum mean voltage for de-energized state (default: 5V)
            
        Returns
        -------
        numpy.ndarray
            Boolean mask where True indicates de-energized points
            
        Notes
        -----
        Uses linear regression to calculate cluster slopes.
        Performs interpolation check for clusters between de-energized clusters.
        """
        from scipy.stats import linregress
        
        unique_labels = np.unique(labels)
        deenergized_clusters = np.zeros(len(unique_labels), dtype=bool)
        cluster_mean_voltages = []
        
        for ix, lab in enumerate(unique_labels):
            cluster_mask = labels == lab
            cluster_voltages = voltages[cluster_mask]
            cluster_timestamps = timestamps[cluster_mask]
            
            if len(cluster_voltages) > 1:
                try:
                    cluster_mean_voltage = np.mean(cluster_voltages)
                    cluster_abs_slope = np.abs(
                        linregress(cluster_timestamps, cluster_voltages).slope
                    )
                    
                    if (cluster_abs_slope < slope_threshold and 
                        cluster_mean_voltage < mean_threshold):
                        deenergized_clusters[ix] = True
                    cluster_mean_voltages.append(cluster_mean_voltage)
                except:
                    cluster_mean_voltages.append(np.mean(cluster_voltages))
            else:
                cluster_mean_voltages.append(np.mean(cluster_voltages))
        
        # Check for de-energized clusters between other de-energized clusters
        if len(deenergized_clusters) > 2:
            for i in range(1, len(deenergized_clusters) - 1):
                if (deenergized_clusters[i-1] and deenergized_clusters[i+1] and 
                    cluster_mean_voltages[i] < mean_threshold):
                    deenergized_clusters[i] = True
        
        # Create mask for all points
        deenergized_mask = np.zeros(len(voltages), dtype=bool)
        for ix, lab in enumerate(unique_labels):
            if deenergized_clusters[ix]:
                deenergized_mask[labels == lab] = True
        
        return deenergized_mask
    
    def is_stabilizing(self, voltages, labels, timestamps, slope_cutoff=1):
        """
        Identify stabilizing segments based on slope magnitude.
        
        Segments with high absolute slope are considered to be in a 
        stabilizing/transitional state.
        
        Parameters
        ----------
        voltages : numpy.ndarray
            Array of voltage values
        labels : numpy.ndarray
            Array of segment labels/cluster IDs
        timestamps : numpy.ndarray
            Array of timestamp values
        slope_cutoff : float, optional
            Minimum absolute slope for stabilizing state (default: 1)
            
        Returns
        -------
        numpy.ndarray
            Boolean mask where True indicates stabilizing points
            
        Notes
        -----
        Uses linear regression to calculate cluster slopes.
        Only clusters with >1 point are evaluated for slope.
        """
        from scipy.stats import linregress
        
        unique_labels = np.unique(labels)
        stabilizing_clusters = np.zeros(len(unique_labels), dtype=bool)
        
        for ix, lab in enumerate(unique_labels):
            cluster_mask = labels == lab
            cluster_voltages = voltages[cluster_mask]
            cluster_timestamps = timestamps[cluster_mask]
            
            if len(cluster_voltages) > 1:
                try:
                    cluster_abs_slope = np.abs(
                        linregress(cluster_timestamps, cluster_voltages).slope
                    )
                    if cluster_abs_slope > slope_cutoff:
                        stabilizing_clusters[ix] = True
                except:
                    pass
        
        # Create mask for all points
        stabilizing_mask = np.zeros(len(voltages), dtype=bool)
        for ix, lab in enumerate(unique_labels):
            if stabilizing_clusters[ix]:
                stabilizing_mask[labels == lab] = True
        
        return stabilizing_mask
    
    def is_steadystate(self, voltages, labels, timestamps, slope_threshold=0.1, mean_threshold=20):
        """
        Identify steady-state segments using cluster analysis.
        
        A segment is considered steady-state if it has high mean voltage and
        minimal slope, indicating stable operation.
        
        Parameters
        ----------
        voltages : numpy.ndarray
            Array of voltage values
        labels : numpy.ndarray
            Array of segment labels/cluster IDs
        timestamps : numpy.ndarray
            Array of timestamp values
        slope_threshold : float, optional
            Maximum absolute slope for steady state (default: 0.1)
        mean_threshold : float, optional
            Minimum mean voltage for steady state (default: 20V)
            
        Returns
        -------
        numpy.ndarray
            Boolean mask where True indicates steady-state points
            
        Notes
        -----
        Performs interpolation check for clusters between steady-state clusters.
        Uses linear regression for slope calculation.
        """
        from scipy.stats import linregress
        
        unique_labels = np.unique(labels)
        steadystate_clusters = np.zeros(len(unique_labels), dtype=bool)
        cluster_mean_voltages = []
        
        for ix, lab in enumerate(unique_labels):
            cluster_mask = labels == lab
            cluster_voltages = voltages[cluster_mask]
            cluster_timestamps = timestamps[cluster_mask]
            
            if len(cluster_voltages) > 1:
                try:
                    cluster_mean_voltage = np.mean(cluster_voltages)
                    cluster_abs_slope = np.abs(
                        linregress(cluster_timestamps, cluster_voltages).slope
                    )
                    
                    if (cluster_abs_slope < slope_threshold and 
                        cluster_mean_voltage > mean_threshold):
                        steadystate_clusters[ix] = True
                    cluster_mean_voltages.append(cluster_mean_voltage)
                except:
                    cluster_mean_voltages.append(np.mean(cluster_voltages))
            else:
                cluster_mean_voltages.append(np.mean(cluster_voltages))
        
        # Check for steady state clusters between other steady state clusters
        if len(steadystate_clusters) > 2:
            for i in range(1, len(steadystate_clusters) - 1):
                if (steadystate_clusters[i-1] and steadystate_clusters[i+1] and 
                    cluster_mean_voltages[i] > mean_threshold):
                    steadystate_clusters[i] = True
        
        # Create mask for all points
        steadystate_mask = np.zeros(len(voltages), dtype=bool)
        for ix, lab in enumerate(unique_labels):
            if steadystate_clusters[ix]:
                steadystate_mask[labels == lab] = True
        
        return steadystate_mask
    
    def classify_segments(self, df):
        """
        Classify all segments in a DataFrame using cluster analysis.
        
        Applies classification functions to determine operational state
        for each data point based on cluster characteristics.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with columns: 'voltage', 'segment', 'timestamp'
            
        Returns
        -------
        pandas.DataFrame
            Input DataFrame with added 'label' column containing:
            'de-energized', 'stabilizing', 'steady_state', or 'unidentified'
            
        Notes
        -----
        Classification priority: de-energized > stabilizing > steady_state > unidentified
        Each point can only have one label.
        """
        # Get arrays
        voltages = df['voltage'].to_numpy()
        labels = df['segment'].to_numpy()
        timestamps = df['timestamp'].to_numpy()
        
        # Get masks from classification functions
        deenergized_mask = self.is_deenergized(voltages, labels, timestamps)
        stabilizing_mask = self.is_stabilizing(voltages, labels, timestamps)
        steadystate_mask = self.is_steadystate(voltages, labels, timestamps)
        
        # Apply labels based on masks
        df['label'] = 'unidentified'
        df.loc[deenergized_mask, 'label'] = 'de-energized'
        df.loc[stabilizing_mask, 'label'] = 'stabilizing'
        df.loc[steadystate_mask, 'label'] = 'steady_state'
        
        return df
    
    def analyze_csv(self, csv_path):
        """
        Analyze a single CSV file with comprehensive voltage statistics.
        
        Processes a voltage time series file, classifies segments, calculates
        statistical metrics, and flags anomalies based on operational state.
        
        Parameters
        ----------
        csv_path : pathlib.Path
            Path to CSV file containing voltage data
            
        Returns
        -------
        tuple or None
            If successful, returns (results, df, grouping):
            - results : list of dict
                Statistical metrics for each segment label including:
                mean_voltage, std, variance, min/max, percentiles, slope,
                r_squared, skewness, kurtosis, flagging information
            - df : pandas.DataFrame
                Processed DataFrame with classified segments
            - grouping : dict
                Parsed filename metadata and folder information
            Returns None if analysis fails.
            
        Side Effects
        ------------
        - Appends to self.results for later plot generation
        - Appends to self.failed_files if processing fails
        
        Notes
        -----
        Only steady-state segments are evaluated for anomalies.
        Other states (de-energized, stabilizing, unidentified) are never flagged.
        
        Examples
        --------
        >>> analyzer = SimplifiedVoltageAnalyzer()
        >>> result = analyzer.analyze_csv(Path('data/test_segments.csv'))
        >>> if result:
        ...     metrics, df, metadata = result
        ...     print(f"Found {len(metrics)} segment types")
        """
        try:
            filename = csv_path.name
            grouping = self.parse_filename(filename)
            
            # Add folder info
            parent_path = csv_path.parent
            grouping['dc_folder'] = parent_path.name  # dc1 or dc2
            grouping['test_case_folder'] = parent_path.parent.name
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Check required columns
            if not all(col in df.columns for col in ['voltage', 'timestamp', 'segment']):
                self.failed_files.append((filename, "Missing required columns"))
                return None
            
            # Classify segments using actual functions
            df = self.classify_segments(df)
            
            # Calculate metrics for each label type
            results = []
            for label in df['label'].unique():
                label_data = df[df['label'] == label]
                voltage_values = label_data['voltage'].values
                
                if len(voltage_values) == 0:
                    continue
                
                # Comprehensive metrics for all
                metrics = {
                    **grouping,
                    'label': label,
                    'n_points': len(voltage_values),
                    'mean_voltage': np.mean(voltage_values),
                    'median_voltage': np.median(voltage_values),
                    'std': np.std(voltage_values),
                    'variance': np.var(voltage_values),
                    'min_voltage': np.min(voltage_values),
                    'max_voltage': np.max(voltage_values),
                    'range': np.max(voltage_values) - np.min(voltage_values),
                    'q1': np.percentile(voltage_values, 25),
                    'q3': np.percentile(voltage_values, 75),
                    'iqr': np.percentile(voltage_values, 75) - np.percentile(voltage_values, 25),
                    'cv': (np.std(voltage_values) / np.mean(voltage_values) * 100) if np.mean(voltage_values) != 0 else 0
                }
                
                # Calculate slope and r-squared if enough points
                if len(label_data) > 1:
                    try:
                        result = linregress(range(len(voltage_values)), voltage_values)
                        metrics['slope'] = result.slope
                        metrics['abs_slope'] = abs(result.slope)
                        metrics['r_squared'] = result.rvalue ** 2
                    except:
                        metrics['slope'] = 0
                        metrics['abs_slope'] = 0
                        metrics['r_squared'] = 0
                else:
                    metrics['slope'] = 0
                    metrics['abs_slope'] = 0
                    metrics['r_squared'] = 0
                
                # Calculate skewness and kurtosis
                if len(voltage_values) > 3:
                    metrics['skewness'] = stats.skew(voltage_values)
                    metrics['kurtosis'] = stats.kurtosis(voltage_values)
                else:
                    metrics['skewness'] = 0
                    metrics['kurtosis'] = 0
                
                # Only flag anomalies for steady state
                if label == 'steady_state':
                    flags = []
                    reasons = []
                    
                    # CHECK 1: Voltage below 18V for steady state
                    if metrics['mean_voltage'] < self.operational_min:
                        flags.append('low_voltage')
                        reasons.append(f"Mean voltage {metrics['mean_voltage']:.2f}V < 18V")
                    
                    # CHECK 2: High variance
                    if metrics['variance'] > self.steady_state_thresholds['max_variance']:
                        flags.append('high_variance')
                        reasons.append(f"Variance {metrics['variance']:.3f} > {self.steady_state_thresholds['max_variance']}")
                    
                    # CHECK 3: High std
                    if metrics['std'] > self.steady_state_thresholds['max_std']:
                        flags.append('high_std')
                        reasons.append(f"Std {metrics['std']:.3f} > {self.steady_state_thresholds['max_std']}")
                    
                    # CHECK 4: Excessive slope
                    if metrics['abs_slope'] > self.steady_state_thresholds['max_slope']:
                        flags.append('excessive_slope')
                        reasons.append(f"Slope {metrics['abs_slope']:.4f} > {self.steady_state_thresholds['max_slope']}")
                    
                    # NO MORE Z-SCORE CHECKING - removed because it flags tight steady states
                    metrics['n_outliers_zscore'] = 0
                    metrics['max_zscore'] = 0
                    metrics['outlier_indices'] = []
                    metrics['n_outliers_iqr'] = 0
                    
                    metrics['flagged'] = len(flags) > 0
                    metrics['flags'] = ', '.join(flags) if flags else ''
                    metrics['flag_reasons'] = '; '.join(reasons) if reasons else ''
                    
                else:
                    # Other labels (de-energized, stabilizing, unidentified) are NEVER flagged
                    # Just track their statistics
                    metrics['flagged'] = False
                    metrics['flags'] = ''
                    metrics['flag_reasons'] = ''
                    metrics['n_outliers_zscore'] = 0
                    metrics['n_outliers_iqr'] = 0
                    metrics['max_zscore'] = 0
                    metrics['outlier_indices'] = []
                
                results.append(metrics)
            
            # Store for plot access
            self.results.extend(results)
            
            return results, df, grouping
            
        except Exception as e:
            self.failed_files.append((csv_path.name, str(e)))
            return None
    
    def create_simple_plot(self, df, grouping, output_path):
        """
        Create diagnostic plot showing voltage data and flagging reasons.
        
        Generates a two-panel plot with voltage time series on top and
        segment visualization on bottom, including flagging information.
        
        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with voltage data and segment labels
        grouping : dict
            Metadata from filename parsing
        output_path : pathlib.Path
            Path where plot should be saved
            
        Side Effects
        ------------
        - Saves plot to output_path as PNG file
        - Uses data from self.results for flagging information
        
        Notes
        -----
        Plot uses scatter points without connecting lines.
        Title color is red for flagged files, black otherwise.
        Reference lines show operational thresholds.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        
        colors = {
            'de-energized': 'gray',
            'stabilizing': 'orange',
            'steady_state': 'green',
            'unidentified': 'purple'
        }
        
        # Main voltage plot - SCATTER ONLY (no lines)
        for label in df['label'].unique():
            label_data = df[df['label'] == label]
            ax1.scatter(label_data['timestamp'], label_data['voltage'],
                       color=colors.get(label, 'black'),
                       label=label, s=15, alpha=0.7, edgecolors='none')
        
        # Add reference lines
        ax1.axhline(y=self.operational_min, color='red', linestyle='--', alpha=0.5, 
                   label='18V threshold', linewidth=2)
        ax1.axhline(y=self.deenergized_max, color='gray', linestyle='--', alpha=0.3)
        
        # Get flagging info for steady state only
        flagged_info = []
        if 'steady_state' in df['label'].values:
            ss_data = df[df['label'] == 'steady_state']
            
            # Get flag reasons from our results
            for result in self.results:
                if (result.get('label') == 'steady_state' and 
                    result.get('unit_id') == grouping.get('unit_id') and
                    result.get('test_run') == grouping.get('test_run')):
                    
                    if result.get('flagged'):
                        flagged_info.append(f"FLAGGED: {result.get('flag_reasons', 'Unknown reason')}")
                    
                    # Add statistics text
                    stats_text = (f"Mean: {result.get('mean_voltage', 0):.2f}V | "
                                f"Std: {result.get('std', 0):.3f} | "
                                f"Var: {result.get('variance', 0):.3f} | "
                                f"Slope: {result.get('abs_slope', 0):.4f}")
                    flagged_info.append(stats_text)
                    break
        
        # Title with all info
        title = (f"Unit: {grouping.get('unit_id', 'NA')} | "
                f"Test Case: {grouping.get('test_case', 'NA')} | "
                f"Run: {grouping.get('test_run', 'NA')} | "
                f"DC: {grouping.get('dc_folder', 'NA')}")
        
        if flagged_info:
            title += f"\n{' | '.join(flagged_info)}"
        
        ax1.set_title(title, fontsize=11, color='red' if flagged_info else 'black')
        ax1.set_xlabel('Timestamp', fontsize=10)
        ax1.set_ylabel('Voltage (V)', fontsize=10)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Segment visualization at bottom
        segment_colors = plt.cm.tab10(np.linspace(0, 1, len(df['segment'].unique())))
        for i, seg in enumerate(df['segment'].unique()):
            seg_data = df[df['segment'] == seg]
            ax2.scatter(seg_data['timestamp'], [seg]*len(seg_data), 
                       color=segment_colors[i], s=10, alpha=0.7, label=f'Seg {seg}')
        
        ax2.set_xlabel('Timestamp', fontsize=10)
        ax2.set_ylabel('Segment ID', fontsize=10)
        ax2.set_title('Original Segment Clustering', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
    
    def create_summary_plot(self, all_results_df):
        """
        Create summary visualization of all flagged anomalies.
        
        Generates a 2x2 grid of plots showing flagged items by test case,
        label type, voltage distribution, and variance distribution.
        
        Parameters
        ----------
        all_results_df : pandas.DataFrame
            Combined results DataFrame from all analyzed files
            
        Side Effects
        ------------
        Saves plot to self.output_folder/summary_flagged.png
        
        Notes
        -----
        Only creates plot if there are flagged items.
        Includes threshold reference lines where applicable.
        """
        if all_results_df.empty or 'flagged' not in all_results_df.columns:
            print("No data to create summary plot")
            return
            
        if not any(all_results_df['flagged']):
            print("No flagged items to plot")
            return
        
        flagged = all_results_df[all_results_df['flagged']]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Flagged items by test case
        ax = axes[0, 0]
        if 'test_case' in flagged.columns:
            test_cases = flagged['test_case'].value_counts()
            ax.bar(range(len(test_cases)), test_cases.values, color='coral')
            ax.set_xticks(range(len(test_cases)))
            ax.set_xticklabels(test_cases.index, rotation=45, ha='right', fontsize=8)
            ax.set_title('Flagged Items by Test Case')
            ax.set_ylabel('Count')
        
        # Plot 2: Flagged items by label type
        ax = axes[0, 1]
        label_counts = flagged['label'].value_counts()
        colors_map = {'steady_state': 'green', 'voltage_outlier': 'red', 'unidentified': 'purple'}
        colors = [colors_map.get(x, 'gray') for x in label_counts.index]
        ax.bar(range(len(label_counts)), label_counts.values, color=colors)
        ax.set_xticks(range(len(label_counts)))
        ax.set_xticklabels(label_counts.index, rotation=45, ha='right')
        ax.set_title('Flagged Items by Label Type')
        ax.set_ylabel('Count')
        
        # Plot 3: Voltage distribution of flagged steady states
        ax = axes[1, 0]
        ss_flagged = flagged[flagged['label'] == 'steady_state']
        if not ss_flagged.empty:
            ax.hist(ss_flagged['mean_voltage'], bins=20, edgecolor='black', color='green', alpha=0.7)
            ax.axvline(x=self.operational_min, color='red', linestyle='--', alpha=0.5)
            ax.axvline(x=self.operational_max, color='red', linestyle='--', alpha=0.5)
            ax.set_title('Voltage Distribution - Flagged Steady States')
            ax.set_xlabel('Mean Voltage (V)')
            ax.set_ylabel('Count')
        
        # Plot 4: Variance distribution
        ax = axes[1, 1]
        if 'variance' in flagged.columns:
            ax.hist(flagged['variance'], bins=20, edgecolor='black', color='blue', alpha=0.7)
            ax.axvline(x=self.steady_state_thresholds['max_variance'], 
                      color='red', linestyle='--', alpha=0.5, 
                      label=f"Threshold: {self.steady_state_thresholds['max_variance']}")
            ax.set_title('Variance Distribution - Flagged Items')
            ax.set_xlabel('Variance')
            ax.set_ylabel('Count')
            ax.legend()
        
        plt.suptitle('Summary of Flagged Anomalies', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_folder / 'summary_flagged.png'
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"Summary plot saved to: {output_path}")
    
    def run_analysis(self):
        """
        Execute complete voltage analysis pipeline on all CSV files.
        
        Processes all CSV files in the input folder structure, generates
        plots for flagged files, creates Excel reports, and produces
        summary visualizations.
        
        Returns
        -------
        pandas.DataFrame
            Combined results DataFrame containing:
            - All segment metrics and statistics
            - Flagging information for anomalies
            - Metadata from filenames and folder structure
            Returns empty DataFrame if no files are processed.
            
        Side Effects
        ------------
        - Creates output folder structure
        - Generates PNG plots for flagged files in output_folder/flagged_plots/
        - Saves Excel workbook to output_folder/analysis_results.xlsx with sheets:
            - All_Results: Complete analysis data
            - Flagged_Only: Filtered to show only anomalies
            - Summary_Stats: Aggregated statistics by test case and label
            - DC_Comparison: Comparison between dc1 and dc2
        - Creates summary plot at output_folder/summary_flagged.png
        - Prints progress and summary statistics to console
        
        Notes
        -----
        Expected folder structure:
        input_folder/
        ├── test_case_1/
        │   ├── dc1/*_segments.csv
        │   └── dc2/*_segments.csv
        └── test_case_2/
            ├── dc1/*_segments.csv
            └── dc2/*_segments.csv
            
        Examples
        --------
        >>> analyzer = SimplifiedVoltageAnalyzer('data/', 'results/')
        >>> results_df = analyzer.run_analysis()
        >>> print(f"Analysis complete: {len(results_df)} segments processed")
        
        See Also
        --------
        analyze_csv : Single file analysis
        create_simple_plot : Individual file visualization
        create_summary_plot : Summary visualization
        """
        print("="*60)
        print("VOLTAGE ANALYSIS WITH CLUSTER CLASSIFICATION")
        print("="*60)
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
        
        # Collect all CSV files
        csv_files = []
        for test_case_folder in self.input_folder.iterdir():
            if test_case_folder.is_dir():
                for dc_folder in ['dc1', 'dc2']:
                    dc_path = test_case_folder / dc_folder
                    if dc_path.exists():
                        found_csvs = list(dc_path.glob('*_segments.csv'))
                        if found_csvs:
                            csv_files.extend(found_csvs)
                            print(f"  Found {len(found_csvs)} files in {test_case_folder.name}/{dc_folder}")
        
        if not csv_files:
            print("No CSV files found!")
            return pd.DataFrame()
        
        print(f"\nTotal CSV files to process: {len(csv_files)}")
        
        # Process each CSV
        all_results = []
        flagged_files = []
        
        for csv_path in tqdm(csv_files, desc="Processing"):
            result = self.analyze_csv(csv_path)
            if result:
                file_results, df, grouping = result
                all_results.extend(file_results)
                
                # Check if any segment is flagged
                if any(r.get('flagged', False) for r in file_results):
                    flagged_files.append((df, grouping, csv_path))
        
        # Create results dataframe
        results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
        
        # Create output folders - SIMPLIFIED STRUCTURE
        plots_folder = self.output_folder / 'flagged_plots'
        plots_folder.mkdir(exist_ok=True)
        
        # Generate plots only for flagged files
        if flagged_files:
            print(f"\nGenerating plots for {len(flagged_files)} flagged files...")
            for df, grouping, csv_path in tqdm(flagged_files, desc="Creating plots"):
                # Simple filename including DC folder
                plot_name = (f"{grouping.get('unit_id', 'NA')}_"
                           f"{grouping.get('test_case', 'NA')}_"
                           f"run{grouping.get('test_run', 'NA')}_"
                           f"{grouping.get('dc_folder', 'NA')}.png")
                plot_path = plots_folder / plot_name
                self.create_simple_plot(df, grouping, plot_path)
        else:
            print("\nNo flagged files found - no plots to generate")
        
        # Save Excel report
        if not results_df.empty:
            excel_path = self.output_folder / 'analysis_results.xlsx'
            with pd.ExcelWriter(excel_path) as writer:
                # All results
                results_df.to_excel(writer, sheet_name='All_Results', index=False)
                
                # Only flagged
                if 'flagged' in results_df.columns:
                    flagged_df = results_df[results_df['flagged'] == True]
                    if not flagged_df.empty:
                        flagged_df.to_excel(writer, sheet_name='Flagged_Only', index=False)
                        
                        # Summary statistics by test case and label
                        summary = results_df.groupby(['test_case', 'label']).agg({
                            'mean_voltage': ['mean', 'std', 'min', 'max'],
                            'variance': ['mean', 'max'],
                            'n_points': 'sum',
                            'flagged': 'sum'
                        }).round(3)
                        summary.to_excel(writer, sheet_name='Summary_Stats')
                        
                        # DC comparison
                        if 'dc_folder' in results_df.columns:
                            dc_summary = results_df.groupby(['dc_folder', 'label']).agg({
                                'mean_voltage': ['mean', 'std'],
                                'flagged': 'sum',
                                'n_points': 'sum'
                            }).round(3)
                            dc_summary.to_excel(writer, sheet_name='DC_Comparison')
            
            print(f"\nExcel report saved to: {excel_path}")
            
            # Create summary plot
            self.create_summary_plot(results_df)
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Total segments analyzed: {len(results_df)}")
        
        if 'flagged' in results_df.columns:
            n_flagged = results_df['flagged'].sum()
            print(f"Flagged segments: {n_flagged}")
            
            if n_flagged > 0:
                print("\nFlagged breakdown by label:")
                for label in results_df[results_df['flagged']]['label'].unique():
                    count = len(results_df[(results_df['flagged']) & (results_df['label'] == label)])
                    print(f"  {label}: {count}")
                
                print("\nFlagged breakdown by test case:")
                if 'test_case' in results_df.columns:
                    for tc in results_df[results_df['flagged']]['test_case'].unique():
                        count = len(results_df[(results_df['flagged']) & (results_df['test_case'] == tc)])
                        print(f"  {tc}: {count}")
        
        if self.failed_files:
            print(f"\n{len(self.failed_files)} files failed to process")
            for filename, error in self.failed_files[:5]:
                print(f"  {filename}: {error}")
        
        print(f"\nOutputs:")
        print(f"  Excel: {self.output_folder}/analysis_results.xlsx")
        print(f"  Plots: {self.output_folder}/flagged_plots/")
        print(f"  Summary: {self.output_folder}/summary_flagged.png")
        
        return results_df


def main():
    """
    Main entry point for voltage analysis.
    
    Creates analyzer instance and runs complete analysis pipeline.
    
    Returns
    -------
    pandas.DataFrame
        Results DataFrame from analysis
        
    Examples
    --------
    >>> results = main()
    >>> print(f"Processed {len(results)} segments")
    """
    analyzer = SimplifiedVoltageAnalyzer(
        input_folder='greedygaussv4',
        output_folder='voltage_analysis'
    )
    
    results = analyzer.run_analysis()
    
    return results


if __name__ == "__main__":
    main()
