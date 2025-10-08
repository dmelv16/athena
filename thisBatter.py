    def run_analysis(self, data: pd.DataFrame):
        """
        Execute complete voltage analysis pipeline with dynamic thresholds.
        
        Performs a two-pass analysis:
        
        1. **First pass**: Collect metrics to establish baseline
        2. **Calculate dynamic thresholds**: ±50% of standard deviation for each OFP/test_case group  
        3. **Second pass**: Apply dynamic thresholds for anomaly detection
        
        :param data: DataFrame with columns from translate_model_segments
        :type data: pd.DataFrame
        :return: Combined results DataFrame containing all segment metrics and flagging info
        :rtype: pd.DataFrame
        
        :raises: ValueError if no voltage columns are found
        
        :Example:
        
        >>> analyzer = SimplifiedVoltageAnalyzer('output_folder')
        >>> data = pd.read_parquet('segmented_data.parquet')
        >>> results = analyzer.run_analysis(data)
        
        .. note::
           Dynamic thresholds are calculated as mean ± (50% * std) for each metric
           within OFP/test_case groups. Requires at least 3 samples per group.
           
        .. warning::
           Two-pass processing doubles computation time but improves accuracy
        """
        print("="*60)
        print("VOLTAGE ANALYSIS WITH DYNAMIC THRESHOLDS")
        print("="*60)
        print(f"Output folder: {self.output_folder}")
        
        # Identify voltage columns
        voltage_columns = []
        for col in ['voltage_28v_dc1_cal', 'voltage_28v_dc2_cal']:
            if col in data.columns:
                voltage_columns.append(col)
                print(f"Found voltage column: {col}")
        
        if not voltage_columns:
            print("ERROR: No voltage columns found!")
            return pd.DataFrame()
        
        # Get unique run_ids
        unique_runs = data['run_id'].unique()
        print(f"\nTotal groups to process: {len(unique_runs)}")
        
        # FIRST PASS: Collect metrics for baseline
        print("\nFirst pass: Collecting metrics for dynamic threshold calculation...")
            
        all_metrics = []
        
        for run_id in tqdm(unique_runs, desc="Collecting metrics"):
            group_df = data[data['run_id'] == run_id].copy()
            
            # Get grouping info
            first_row = group_df.iloc[0]
            grouping = {
                'run_id': run_id,
                'ofp': first_row.get('ofp', 'NA'),
                'test_case': first_row.get('test_case', 'NA'),
                'test_run': first_row.get('test_run', 'NA'),
                'save': first_row.get('save', 'NA'),
                'unit_id': first_row.get('unit_id', 'NA'),
                'station': first_row.get('station', 'NA'),
                'dc': first_row.get('dc', 'NA')
            }
            
            # Process each voltage column
            for voltage_col in voltage_columns:
                if voltage_col not in group_df.columns:
                    continue
                
                try:
                    # First pass without thresholds
                    file_results, _ = self.analyze_group(
                        group_df, voltage_col, grouping, dynamic_thresholds=None
                    )
                    all_metrics.extend(file_results)
                except Exception as e:
                    print(f"\nError in first pass for run_id {run_id}: {e}")
                    continue
        
        if not all_metrics:
            print("No metrics collected!")
            return pd.DataFrame()
        
        # Calculate dynamic thresholds
        print("\nCalculating dynamic thresholds from OFP/test_case groups...")
        metrics_df = pd.DataFrame(all_metrics)
        
        # Only use steady state for baseline
        steady_df = metrics_df[metrics_df['label'] == 'steady_state']
        
        dynamic_thresholds = {}
        if not steady_df.empty:
            for (ofp, test_case), group in steady_df.groupby(['ofp', 'test_case']):
                if len(group) >= 3:  # Need at least 3 samples
                    thresholds = {}
                    
                    # Calculate thresholds for each metric
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
                            thresholds[f'min_{metric}'] = max(0, lower_threshold)
                    
                    # For abs_slope, we need different keys
                    if 'abs_slope' in group.columns:
                        thresholds['max_slope'] = thresholds.pop('max_abs_slope', 0)
                        thresholds['min_slope'] = thresholds.pop('min_abs_slope', 0)
                    
                    dynamic_thresholds[f"{ofp}_{test_case}"] = thresholds
                    print(f"  {ofp}/{test_case}:")
                    print(f"    Variance: [{thresholds.get('min_variance', 0):.3f}, {thresholds.get('max_variance', 0):.3f}]")
                    print(f"    Std: [{thresholds.get('min_std', 0):.3f}, {thresholds.get('max_std', 0):.3f}]")
                    print(f"    Slope: [{thresholds.get('min_slope', 0):.4f}, {thresholds.get('max_slope', 0):.4f}]")
                    print(f"    IQR: [{thresholds.get('min_iqr', 0):.3f}, {thresholds.get('max_iqr', 0):.3f}]")
        else:
            print("  No steady state segments found for threshold calculation")
        
        # SECOND PASS: Process with dynamic thresholds
        print("\nSecond pass: Applying dynamic thresholds...")
        
        all_results = []
        flagged_files = []
        
        for run_id in tqdm(unique_runs, desc="Processing with thresholds"):
            group_df = data[data['run_id'] == run_id].copy()
            
            # Get grouping info
            first_row = group_df.iloc[0]
            grouping = {
                'run_id': run_id,
                'ofp': first_row.get('ofp', 'NA'),
                'test_case': first_row.get('test_case', 'NA'),
                'test_run': first_row.get('test_run', 'NA'),
                'save': first_row.get('save', 'NA'),
                'unit_id': first_row.get('unit_id', 'NA'),
                'station': first_row.get('station', 'NA'),
                'dc': first_row.get('dc', 'NA')
            }
            
            # Process each voltage column
            for voltage_col in voltage_columns:
                if voltage_col not in group_df.columns:
                    continue
                
                try:
                    # Process with dynamic thresholds
                    file_results, analyzed_df = self.analyze_group(
                        group_df, voltage_col, grouping, 
                        dynamic_thresholds=dynamic_thresholds
                    )
                    all_results.extend(file_results)
                    
                    # Check if any STEADY STATE segment is flagged
                    steady_state_flagged = any(
                        r.get('flagged', False) and r.get('label') == 'steady_state') 
                        for r in file_results
                    )
                    
                    if steady_state_flagged:
                        flagged_files.append((analyzed_df, voltage_col, grouping))
                        
                except Exception as e:
                    print(f"\nError processing run_id {run_id}: {e}")
                    self.failed_files.append((run_id, str(e)))
                    continue
        
        # Store results for use in plotting
        self.results = all_results
        
        # Create results dataframe
        results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
        
        if results_df.empty:
            print("\nNo data was successfully processed!")
            return results_df
        
        # Create output folders
        plots_folder = self.output_folder / 'flagged_plots'
        plots_folder.mkdir(exist_ok=True, parents=True)
        
        # Create flagged_data folder for CSV outputs
        flagged_data_folder = self.output_folder / 'flagged_data'
        flagged_data_folder.mkdir(exist_ok=True, parents=True)
        
        # Generate plots only for flagged files
        if flagged_files:
            print(f"\nGenerating plots for {len(flagged_files)} groups with flagged steady-state segments...")
            plot_counter = {}
            
            for df, voltage_col, grouping in tqdm(flagged_files, desc="Creating plots"):
                try:
                    dc_name = voltage_col.replace('voltage_28v_', '').replace('_cal', '')
                    base_name = (f"{grouping.get('unit_id', 'NA')}_"
                               f"{grouping.get('test_case', 'NA')}_"
                               f"run{grouping.get('test_run', 'NA')}_"
                               f"{dc_name}")
                    
                    # Handle duplicate names
                    if base_name in plot_counter:
                        plot_counter[base_name] += 1
                        plot_name = f"{base_name}_{plot_counter[base_name]}.png"
                    else:
                        plot_counter[base_name] = 0
                        plot_name = f"{base_name}.png"
                    
                    plot_path = plots_folder / plot_name
                    self.create_simple_plot(df, voltage_col, grouping, plot_path)
                except Exception as e:
                    print(f"Error creating plot: {e}")
        else:
            print("\nNo steady-state flagged segments found - no plots to generate")
        
        # Save CSV files
        print(f"\nSaving analysis results to CSV files in {flagged_data_folder}...")
