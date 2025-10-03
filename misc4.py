def analyze_csv(self, csv_path, dynamic_thresholds=None):
    """Analyze a single CSV file - ONLY STEADY STATE for flagging."""
    try:
        filename = csv_path.name
        grouping = self.parse_filename(filename)  # This already includes 'save' and 'station'
        
        # Add folder info
        parent_path = csv_path.parent
        grouping['dc_folder'] = parent_path.name  # dc1 or dc2
        grouping['test_case_folder'] = parent_path.parent.name
        
        # Create unique identifier INCLUDING STATION AND SAVE
        grouping['unique_id'] = (f"{grouping.get('unit_id', 'NA')}_"
                                f"{grouping.get('save', 'NA')}_"
                                f"{grouping.get('station', 'NA')}_"  # INCLUDE STATION
                                f"{grouping.get('test_case', 'NA')}_"
                                f"{grouping.get('test_run', 'NA')}_"
                                f"{grouping.get('ofp', 'NA')}_"
                                f"{grouping.get('dc_folder', 'NA')}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Check required columns
        if not all(col in df.columns for col in ['voltage', 'timestamp', 'segment']):
            self.failed_files.append((filename, "Missing required columns"))
            return None
        
        # Classify segments into LABELS
        df = self.classify_segments(df)
        
        # ONLY ANALYZE STEADY STATE LABEL
        results = []
        
        # Check if steady_state exists
        if 'steady_state' not in df['label'].values:
            # No steady state in this file, return empty results
            return results, df, grouping
        
        # Get ALL steady_state data (combines all segments labeled as steady_state)
        steady_data = df[df['label'] == 'steady_state']
        voltage_values = steady_data['voltage'].values
        
        if len(voltage_values) == 0:
            return results, df, grouping
        
        # Calculate metrics for the ENTIRE steady_state label
        metrics = {
            **grouping,  # This includes 'save', 'station', and 'unique_id'
            'label': 'steady_state',
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
        if len(steady_data) > 1:
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
        
        # FLAG CHECKING FOR STEADY STATE
        flags = []
        reasons = []
        
        # Use dynamic thresholds if provided
        if dynamic_thresholds:
            # Use dynamic thresholds for this OFP/test_case group
            group_key = f"{grouping.get('ofp', 'NA')}_{grouping.get('test_case', 'NA')}"
            if group_key in dynamic_thresholds:
                thresh = dynamic_thresholds[group_key]
                
                # Check all 4 thresholds
                failed_checks = 0
                
                # Check variance
                if ('min_variance' in thresh and metrics['variance'] < thresh['min_variance']) or \
                   ('max_variance' in thresh and metrics['variance'] > thresh['max_variance']):
                    failed_checks += 1
                    if 'min_variance' in thresh and metrics['variance'] < thresh['min_variance']:
                        reasons.append(f"Variance {metrics['variance']:.3f} < {thresh['min_variance']:.3f} (dynamic)")
                    else:
                        reasons.append(f"Variance {metrics['variance']:.3f} > {thresh['max_variance']:.3f} (dynamic)")
                
                # Check std
                if ('min_std' in thresh and metrics['std'] < thresh['min_std']) or \
                   ('max_std' in thresh and metrics['std'] > thresh['max_std']):
                    failed_checks += 1
                    if 'min_std' in thresh and metrics['std'] < thresh['min_std']:
                        reasons.append(f"Std {metrics['std']:.3f} < {thresh['min_std']:.3f} (dynamic)")
                    else:
                        reasons.append(f"Std {metrics['std']:.3f} > {thresh['max_std']:.3f} (dynamic)")
                
                # Check abs_slope
                if ('min_abs_slope' in thresh and metrics['abs_slope'] < thresh['min_abs_slope']) or \
                   ('max_abs_slope' in thresh and metrics['abs_slope'] > thresh['max_abs_slope']):
                    failed_checks += 1
                    if 'min_abs_slope' in thresh and metrics['abs_slope'] < thresh['min_abs_slope']:
                        reasons.append(f"Slope {metrics['abs_slope']:.4f} < {thresh['min_abs_slope']:.4f} (dynamic)")
                    else:
                        reasons.append(f"Slope {metrics['abs_slope']:.4f} > {thresh['max_abs_slope']:.4f} (dynamic)")
                
                # Check IQR
                if ('min_iqr' in thresh and metrics['iqr'] < thresh['min_iqr']) or \
                   ('max_iqr' in thresh and metrics['iqr'] > thresh['max_iqr']):
                    failed_checks += 1
                    if 'min_iqr' in thresh and metrics['iqr'] < thresh['min_iqr']:
                        reasons.append(f"IQR {metrics['iqr']:.3f} < {thresh['min_iqr']:.3f} (dynamic)")
                    else:
                        reasons.append(f"IQR {metrics['iqr']:.3f} > {thresh['max_iqr']:.3f} (dynamic)")
                
                # Only flag if ALL 4 fail
                if failed_checks == 4:
                    flags.append('all_thresholds_failed')
                else:
                    reasons = []  # Clear reasons if not flagging
        else:
            # Use fixed thresholds
            failed_checks = 0
            
            if metrics['variance'] > self.steady_state_thresholds['max_variance']:
                failed_checks += 1
                reasons.append(f"Variance {metrics['variance']:.3f} > {self.steady_state_thresholds['max_variance']}")
            
            if metrics['std'] > self.steady_state_thresholds['max_std']:
                failed_checks += 1
                reasons.append(f"Std {metrics['std']:.3f} > {self.steady_state_thresholds['max_std']}")
            
            if metrics['abs_slope'] > self.steady_state_thresholds['max_slope']:
                failed_checks += 1
                reasons.append(f"Slope {metrics['abs_slope']:.4f} > {self.steady_state_thresholds['max_slope']}")
            
            # Only flag if all fail (3 for fixed since no IQR threshold)
            if failed_checks >= 3:
                flags.append('all_thresholds_failed')
            else:
                reasons = []
        
        # Set flagging results
        metrics['flagged'] = len(flags) > 0
        metrics['flags'] = ', '.join(flags) if flags else ''
        metrics['flag_reasons'] = '; '.join(reasons) if reasons else ''
        metrics['n_outliers_zscore'] = 0
        metrics['max_zscore'] = 0
        metrics['outlier_indices'] = []
        metrics['n_outliers_iqr'] = 0
        
        results.append(metrics)
        
        return results, df, grouping
        
    except Exception as e:
        self.failed_files.append((csv_path.name, str(e)))
        return None


def create_simple_plot(self, df, grouping, output_path):
    """Create a detailed plot showing WHY something was flagged."""
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
    
    # Get flagging info for steady state only - WITH ALL FIELDS INCLUDING STATION
    flagged_info = []
    found_flagged = False
    
    if 'steady_state' in df['label'].values:
        # Match INCLUDING STATION
        matching_results = [r for r in self.results 
                          if (r.get('label') == 'steady_state' and 
                              r.get('unit_id') == grouping.get('unit_id') and
                              r.get('save') == grouping.get('save') and
                              r.get('station') == grouping.get('station') and  # MATCH STATION
                              r.get('test_run') == grouping.get('test_run') and
                              r.get('dc_folder') == grouping.get('dc_folder') and
                              r.get('test_case') == grouping.get('test_case') and
                              r.get('ofp') == grouping.get('ofp'))]
        
        # Should only be ONE result for steady_state per file
        if matching_results:
            result = matching_results[0]  # There should only be one steady_state result
            if result.get('flagged'):
                found_flagged = True
                flagged_info.append(f"FLAGGED: {result.get('flag_reasons', 'Unknown reason')}")
                
                # Add statistics for flagged steady state
                stats_text = (f"Flagged Stats: Mean={result.get('mean_voltage', 0):.2f}V | "
                            f"Std={result.get('std', 0):.3f} | "
                            f"Var={result.get('variance', 0):.3f} | "
                            f"Slope={result.get('abs_slope', 0):.4f} | "
                            f"IQR={result.get('iqr', 0):.3f}")
                flagged_info.append(stats_text)
            else:
                # Not flagged, just show stats
                stats_text = (f"Stats: Mean={result.get('mean_voltage', 0):.2f}V | "
                            f"Std={result.get('std', 0):.3f} | "
                            f"Var={result.get('variance', 0):.3f} | "
                            f"Slope={result.get('abs_slope', 0):.4f}")
                flagged_info.append(stats_text)
    
    # Title with all info INCLUDING STATION
    title = (f"Unit: {grouping.get('unit_id', 'NA')} | "
            f"Save: {grouping.get('save', 'NA')} | "
            f"Station: {grouping.get('station', 'NA')} | "  # INCLUDE STATION
            f"Test Case: {grouping.get('test_case', 'NA')} | "
            f"Run: {grouping.get('test_run', 'NA')} | "
            f"DC: {grouping.get('dc_folder', 'NA')}")
    
    if flagged_info:
        title += f"\n{chr(10).join(flagged_info)}"
    
    ax1.set_title(title, fontsize=11, color='red' if found_flagged else 'black')
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


# In run_analysis, update plot naming to include station:
# Generate plots only for flagged files
if flagged_files:
    print(f"\nGenerating plots for {len(flagged_files)} files with flagged steady-state segments...")
    
    for df, grouping, csv_path in tqdm(flagged_files, desc="Creating plots"):
        try:
            # Include STATION in the filename to prevent duplicates
            plot_name = (f"{grouping.get('unit_id', 'NA')}_"
                        f"save{grouping.get('save', 'NA')}_"
                        f"station{grouping.get('station', 'NA')}_"  # INCLUDE STATION
                        f"{grouping.get('test_case', 'NA')}_"
                        f"run{grouping.get('test_run', 'NA')}_"
                        f"{grouping.get('dc_folder', 'NA')}.png")
            
            plot_path = plots_folder / plot_name
            self.create_simple_plot(df, grouping, plot_path)
            
        except Exception as e:
            print(f"Error creating plot for {csv_path.name}: {e}")
