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
                                f"{grouping.get('station', 'NA')}_"
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
                metrics['slope'] = result.slope  # Keep actual slope (can be negative)
                metrics['abs_slope'] = abs(result.slope)  # Still store for reference
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
        
        # Define minimum meaningful threshold (below this, don't flag as "too low")
        MIN_MEANINGFUL_VALUE = 0.001  # Adjust as needed
        
        # Use dynamic thresholds if provided
        if dynamic_thresholds:
            # Use dynamic thresholds for this OFP/test_case group
            group_key = f"{grouping.get('ofp', 'NA')}_{grouping.get('test_case', 'NA')}"
            if group_key in dynamic_thresholds:
                thresh = dynamic_thresholds[group_key]
                
                # Check all 4 thresholds
                failed_checks = 0
                
                # Check variance - don't flag if both value and threshold are near zero
                if 'min_variance' in thresh and 'max_variance' in thresh:
                    if thresh['max_variance'] > MIN_MEANINGFUL_VALUE:  # Only check if threshold is meaningful
                        if metrics['variance'] < thresh['min_variance']:
                            failed_checks += 1
                            reasons.append(f"Variance {metrics['variance']:.3f} < {thresh['min_variance']:.3f} (dynamic)")
                        elif metrics['variance'] > thresh['max_variance']:
                            failed_checks += 1
                            reasons.append(f"Variance {metrics['variance']:.3f} > {thresh['max_variance']:.3f} (dynamic)")
                
                # Check std - don't flag if both value and threshold are near zero
                if 'min_std' in thresh and 'max_std' in thresh:
                    if thresh['max_std'] > MIN_MEANINGFUL_VALUE:  # Only check if threshold is meaningful
                        if metrics['std'] < thresh['min_std']:
                            failed_checks += 1
                            reasons.append(f"Std {metrics['std']:.3f} < {thresh['min_std']:.3f} (dynamic)")
                        elif metrics['std'] > thresh['max_std']:
                            failed_checks += 1
                            reasons.append(f"Std {metrics['std']:.3f} > {thresh['max_std']:.3f} (dynamic)")
                
                # Check ACTUAL SLOPE (not absolute) - should work with negative slopes
                if 'min_slope' in thresh and 'max_slope' in thresh:
                    # For slope, use the actual value (can be negative)
                    if metrics['slope'] < thresh['min_slope'] or metrics['slope'] > thresh['max_slope']:
                        failed_checks += 1
                        if metrics['slope'] < thresh['min_slope']:
                            reasons.append(f"Slope {metrics['slope']:.4f} < {thresh['min_slope']:.4f} (dynamic)")
                        else:
                            reasons.append(f"Slope {metrics['slope']:.4f} > {thresh['max_slope']:.4f} (dynamic)")
                
                # Check IQR - don't flag if both value and threshold are near zero
                if 'min_iqr' in thresh and 'max_iqr' in thresh:
                    if thresh['max_iqr'] > MIN_MEANINGFUL_VALUE:  # Only check if threshold is meaningful
                        if metrics['iqr'] < thresh['min_iqr']:
                            failed_checks += 1
                            reasons.append(f"IQR {metrics['iqr']:.3f} < {thresh['min_iqr']:.3f} (dynamic)")
                        elif metrics['iqr'] > thresh['max_iqr']:
                            failed_checks += 1
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
            
            # For fixed thresholds, check absolute slope (backward compatibility)
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


# Also update the threshold calculation in run_analysis to use actual slope, not abs_slope:
def calculate_dynamic_thresholds_section():
    """
    This is the section in run_analysis that calculates dynamic thresholds.
    Update it to use 'slope' instead of 'abs_slope'
    """
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
                # Change from 'abs_slope' to 'slope' for proper negative/positive handling
                for metric in ['variance', 'std', 'slope', 'iqr']:  # Changed from 'abs_slope' to 'slope'
                    if metric in group.columns:
                        mean_val = group[metric].mean()
                        std_val = group[metric].std()
                        
                        # Upper threshold = mean + (50% of std)
                        # Lower threshold = mean - (50% of std)
                        upper_threshold = mean_val + (0.5 * std_val)
                        lower_threshold = mean_val - (0.5 * std_val)
                        
                        # For variance, std, and iqr, ensure non-negative
                        if metric in ['variance', 'std', 'iqr']:
                            lower_threshold = max(0, lower_threshold)
                        # For slope, allow negative values
                        
                        # Store both thresholds
                        thresholds[f'max_{metric}'] = upper_threshold
                        thresholds[f'min_{metric}'] = lower_threshold
                
                dynamic_thresholds[f"{ofp}_{test_case}"] = thresholds
                print(f"  {ofp}/{test_case}:")
                print(f"    Variance: [{thresholds.get('min_variance', 0):.3f}, {thresholds.get('max_variance', 0):.3f}]")
                print(f"    Std: [{thresholds.get('min_std', 0):.3f}, {thresholds.get('max_std', 0):.3f}]")
                print(f"    Slope: [{thresholds.get('min_slope', 0):.4f}, {thresholds.get('max_slope', 0):.4f}]")  # Changed from abs_slope
                print(f"    IQR: [{thresholds.get('min_iqr', 0):.3f}, {thresholds.get('max_iqr', 0):.3f}]")
    else:
        print("  No steady state segments found for threshold calculation")
