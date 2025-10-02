def analyze_csv(self, csv_path, dynamic_thresholds=None):
    """Analyze a single CSV file with comprehensive statistics."""
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
                
                # Use dynamic thresholds if provided, otherwise use fixed
                if dynamic_thresholds:
                    # Use dynamic thresholds for this OFP/test_case group
                    group_key = f"{grouping.get('ofp', 'NA')}_{grouping.get('test_case', 'NA')}"
                    if group_key in dynamic_thresholds:
                        thresh = dynamic_thresholds[group_key]
                        
                        # Always check voltage minimum
                        if metrics['mean_voltage'] < 18.0:
                            flags.append('low_voltage')
                            reasons.append(f"Mean voltage {metrics['mean_voltage']:.2f}V < 18V")
                        
                        # Check BOTH min and max for variance
                        if 'min_variance' in thresh and metrics['variance'] < thresh['min_variance']:
                            flags.append('low_variance')
                            reasons.append(f"Variance {metrics['variance']:.3f} < {thresh['min_variance']:.3f} (dynamic)")
                        if 'max_variance' in thresh and metrics['variance'] > thresh['max_variance']:
                            flags.append('high_variance')
                            reasons.append(f"Variance {metrics['variance']:.3f} > {thresh['max_variance']:.3f} (dynamic)")
                        
                        # Check BOTH min and max for std
                        if 'min_std' in thresh and metrics['std'] < thresh['min_std']:
                            flags.append('low_std')
                            reasons.append(f"Std {metrics['std']:.3f} < {thresh['min_std']:.3f} (dynamic)")
                        if 'max_std' in thresh and metrics['std'] > thresh['max_std']:
                            flags.append('high_std')
                            reasons.append(f"Std {metrics['std']:.3f} > {thresh['max_std']:.3f} (dynamic)")
                        
                        # Check BOTH min and max for abs_slope
                        if 'min_abs_slope' in thresh and metrics['abs_slope'] < thresh['min_abs_slope']:
                            flags.append('low_slope')
                            reasons.append(f"Slope {metrics['abs_slope']:.4f} < {thresh['min_abs_slope']:.4f} (dynamic)")
                        if 'max_abs_slope' in thresh and metrics['abs_slope'] > thresh['max_abs_slope']:
                            flags.append('excessive_slope')
                            reasons.append(f"Slope {metrics['abs_slope']:.4f} > {thresh['max_abs_slope']:.4f} (dynamic)")
                        
                        # Check BOTH min and max for IQR
                        if 'min_iqr' in thresh and metrics['iqr'] < thresh['min_iqr']:
                            flags.append('low_iqr')
                            reasons.append(f"IQR {metrics['iqr']:.3f} < {thresh['min_iqr']:.3f} (dynamic)")
                        if 'max_iqr' in thresh and metrics['iqr'] > thresh['max_iqr']:
                            flags.append('high_iqr')
                            reasons.append(f"IQR {metrics['iqr']:.3f} > {thresh['max_iqr']:.3f} (dynamic)")
                    else:
                        # No threshold for this group, just check voltage
                        if metrics['mean_voltage'] < 18.0:
                            flags.append('low_voltage')
                            reasons.append(f"Mean voltage {metrics['mean_voltage']:.2f}V < 18V")
                else:
                    # Use fixed thresholds (original logic)
                    if metrics['mean_voltage'] < self.operational_min:
                        flags.append('low_voltage')
                        reasons.append(f"Mean voltage {metrics['mean_voltage']:.2f}V < 18V")
                    
                    if metrics['variance'] > self.steady_state_thresholds['max_variance']:
                        flags.append('high_variance')
                        reasons.append(f"Variance {metrics['variance']:.3f} > {self.steady_state_thresholds['max_variance']}")
                    
                    if metrics['std'] > self.steady_state_thresholds['max_std']:
                        flags.append('high_std')
                        reasons.append(f"Std {metrics['std']:.3f} > {self.steady_state_thresholds['max_std']}")
                    
                    if metrics['abs_slope'] > self.steady_state_thresholds['max_slope']:
                        flags.append('excessive_slope')
                        reasons.append(f"Slope {metrics['abs_slope']:.4f} > {self.steady_state_thresholds['max_slope']}")
                
                # NO Z-SCORE CHECKING
                metrics['n_outliers_zscore'] = 0
                metrics['max_zscore'] = 0
                metrics['outlier_indices'] = []
                metrics['n_outliers_iqr'] = 0
                
                metrics['flagged'] = len(flags) > 0
                metrics['flags'] = ', '.join(flags) if flags else ''
                metrics['flag_reasons'] = '; '.join(reasons) if reasons else ''
                
            else:
                # Other labels (de-energized, stabilizing, unidentified) are NEVER flagged
                metrics['flagged'] = False
                metrics['flags'] = ''
                metrics['flag_reasons'] = ''
                metrics['n_outliers_zscore'] = 0
                metrics['n_outliers_iqr'] = 0
                metrics['max_zscore'] = 0
                metrics['outlier_indices'] = []
            
            results.append(metrics)
        
        return results, df, grouping
        
    except Exception as e:
        self.failed_files.append((csv_path.name, str(e)))
        return None

def run_analysis(self):
    """Run analysis on all CSV files with dynamic thresholds."""
    print("="*60)
    print("VOLTAGE ANALYSIS WITH DYNAMIC THRESHOLDS")
    print("="*60)
    print(f"Input folder: {self.input_folder}")
    print(f"Output folder: {self.output_folder}")
    
    # Check if input folder exists
    if not self.input_folder.exists():
        print(f"ERROR: Input folder {self.input_folder} does not exist!")
        return pd.DataFrame()
    
    # Collect all CSV files
    csv_files = []
    empty_folders = []
    
    for test_case_folder in self.input_folder.iterdir():
        if test_case_folder.is_dir():
            for dc_folder in ['dc1', 'dc2']:
                dc_path = test_case_folder / dc_folder
                if dc_path.exists():
                    found_csvs = list(dc_path.glob('*_segments.csv'))
                    if found_csvs:
                        csv_files.extend(found_csvs)
                        print(f"  Found {len(found_csvs)} files in {test_case_folder.name}/{dc_folder}")
                    else:
                        empty_folders.append(f"{test_case_folder.name}/{dc_folder}")
                else:
                    print(f"  No {dc_folder} folder in {test_case_folder.name}")
    
    # Report empty folders
    if empty_folders:
        print(f"\nEmpty folders (no CSV files):")
        for folder in empty_folders:
            print(f"  - {folder}")
    
    if not csv_files:
        print("\nNo CSV files found in any folder!")
        return pd.DataFrame()
    
    print(f"\nTotal CSV files to process: {len(csv_files)}")
    
    # FIRST PASS: Collect metrics with fixed thresholds to establish baseline
    print("\nFirst pass: Collecting metrics for dynamic threshold calculation...")
    all_metrics = []
    
    for csv_path in tqdm(csv_files, desc="Collecting metrics"):
        try:
            result = self.analyze_csv(csv_path, dynamic_thresholds=None)  # Use fixed thresholds
            if result:
                file_results, _, _ = result
                all_metrics.extend(file_results)
        except Exception as e:
            print(f"\nError in first pass for {csv_path.name}: {e}")
            continue
    
    if not all_metrics:
        print("No metrics collected in first pass!")
        return pd.DataFrame()
    
    # CALCULATE DYNAMIC THRESHOLDS (± 50% of standard deviation)
    print("\nCalculating dynamic thresholds from OFP/test_case groups...")
    metrics_df = pd.DataFrame(all_metrics)
    
    # Only use steady state for baseline
    steady_df = metrics_df[metrics_df['label'] == 'steady_state']
    
    dynamic_thresholds = {}
    if not steady_df.empty:
        for (ofp, test_case), group in steady_df.groupby(['ofp', 'test_case']):
            if len(group) >= 3:  # Need at least 3 samples
                thresholds = {}
                
                # Calculate thresholds for each metric (both upper and lower bounds)
                for metric in ['variance', 'std', 'abs_slope', 'iqr']:
                    if metric in group.columns:
                        mean_val = group[metric].mean()
                        std_val = group[metric].std()
                        
                        # Upper threshold = mean + (50% of std)
                        # Lower threshold = mean - (50% of std)
                        upper_threshold = mean_val + (0.5 * std_val)
                        lower_threshold = mean_val - (0.5 * std_val)
                        
                        # Store both thresholds
                        thresholds[f'max_{metric}'] = upper_threshold
                        thresholds[f'min_{metric}'] = max(0, lower_threshold)  # Ensure non-negative
                
                dynamic_thresholds[f"{ofp}_{test_case}"] = thresholds
                print(f"  {ofp}/{test_case}:")
                print(f"    Variance: [{thresholds.get('min_variance', 0):.3f}, {thresholds.get('max_variance', 0):.3f}]")
                print(f"    Std: [{thresholds.get('min_std', 0):.3f}, {thresholds.get('max_std', 0):.3f}]")
                print(f"    Slope: [{thresholds.get('min_abs_slope', 0):.4f}, {thresholds.get('max_abs_slope', 0):.4f}]")
                print(f"    IQR: [{thresholds.get('min_iqr', 0):.3f}, {thresholds.get('max_iqr', 0):.3f}]")
    else:
        print("  No steady state segments found for threshold calculation")
    
    # SECOND PASS: Process with dynamic thresholds
    print("\nSecond pass: Applying dynamic thresholds...")
    all_results = []
    flagged_files = []
    
    for csv_path in tqdm(csv_files, desc="Processing with dynamic thresholds"):
        try:
            result = self.analyze_csv(csv_path, dynamic_thresholds=dynamic_thresholds)
            if result:
                file_results, df, grouping = result
                all_results.extend(file_results)
                
                # Store results for plot access
                self.results.extend(file_results)
                
                # Check if any segment is flagged
                if any(r.get('flagged', False) for r in file_results):
                    flagged_files.append((df, grouping, csv_path))
        except Exception as e:
            print(f"\nError processing {csv_path.name}: {e}")
            self.failed_files.append((csv_path.name, str(e)))
            continue
    
    # Create results dataframe
    results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    
    if results_df.empty:
        print("\nNo data was successfully processed!")
        return results_df
    
    # Create output folders
    plots_folder = self.output_folder / 'flagged_plots'
    plots_folder.mkdir(exist_ok=True, parents=True)
    
    # Generate plots only for flagged files
    if flagged_files:
        print(f"\nGenerating plots for {len(flagged_files)} flagged files...")
        for df, grouping, csv_path in tqdm(flagged_files, desc="Creating plots"):
            try:
                plot_name = (f"{grouping.get('unit_id', 'NA')}_"
                           f"{grouping.get('test_case', 'NA')}_"
                           f"run{grouping.get('test_run', 'NA')}_"
                           f"{grouping.get('dc_folder', 'NA')}.png")
                plot_path = plots_folder / plot_name
                self.create_simple_plot(df, grouping, plot_path)
            except Exception as e:
                print(f"Error creating plot: {e}")
    else:
        print("\nNo flagged files found - no plots to generate")
    
    # Save Excel report
    excel_path = self.output_folder / 'analysis_results.xlsx'
    try:
        with pd.ExcelWriter(excel_path) as writer:
            # All results with all statistics
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
                    'cv': ['mean', 'max'],
                    'n_points': 'sum',
                    'flagged': 'sum'
                }).round(3)
                summary.to_excel(writer, sheet_name='Summary_Stats')
                
                # Dynamic thresholds sheet (updated to show both min and max)
                if dynamic_thresholds:
                    thresh_data = []
                    for group_key, thresholds in dynamic_thresholds.items():
                        ofp, test_case = group_key.split('_', 1)
                        # Group metrics by type for better display
                        for base_metric in ['variance', 'std', 'abs_slope', 'iqr']:
                            min_key = f'min_{base_metric}'
                            max_key = f'max_{base_metric}'
                            if min_key in thresholds and max_key in thresholds:
                                thresh_data.append({
                                    'ofp': ofp,
                                    'test_case': test_case,
                                    'metric': base_metric,
                                    'min_threshold': thresholds[min_key],
                                    'max_threshold': thresholds[max_key]
                                })
                    if thresh_data:
                        thresh_df = pd.DataFrame(thresh_data)
                        thresh_df.to_excel(writer, sheet_name='Dynamic_Thresholds', index=False)
                
                # DC comparison
                if 'dc_folder' in results_df.columns:
                    dc_summary = results_df.groupby(['dc_folder', 'label']).agg({
                        'mean_voltage': ['mean', 'std'],
                        'cv': 'mean',
                        'flagged': 'sum',
                        'n_points': 'sum'
                    }).round(3)
                    dc_summary.to_excel(writer, sheet_name='DC_Comparison')
        
        print(f"\nExcel report saved to: {excel_path}")
    except Exception as e:
        print(f"Error saving Excel: {e}")
    
    # Create summary plot
    try:
        self.create_summary_plot(results_df)
    except Exception as e:
        print(f"Error creating summary plot: {e}")
    
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
            
            # Show which were flagged by dynamic thresholds
            if dynamic_thresholds:
                dynamic_flagged = results_df[results_df['flag_reasons'].str.contains('dynamic', na=False)]
                if not dynamic_flagged.empty:
                    print(f"\nFlagged by dynamic thresholds: {len(dynamic_flagged)}")
    
    if self.failed_files:
        print(f"\n{len(self.failed_files)} files failed to process:")
        for filename, error in self.failed_files[:5]:
            print(f"  {filename}: {error}")
        if len(self.failed_files) > 5:
            print(f"  ... and {len(self.failed_files) - 5} more")
    
    print(f"\nOutputs:")
    print(f"  Excel: {self.output_folder}/analysis_results.xlsx")
    print(f"  Plots: {self.output_folder}/flagged_plots/")
    print(f"  Summary: {self.output_folder}/summary_flagged.png")
    
    return results_df
