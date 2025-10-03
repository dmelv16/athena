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
                
                # Check if any STEADY STATE segment is flagged (FIXED: only steady state)
                steady_state_flagged = any(
                    r.get('flagged', False) and r.get('label') == 'steady_state' 
                    for r in file_results
                )
                
                if steady_state_flagged:
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
    
    # Generate plots only for flagged files (with unique names to prevent overwriting)
    if flagged_files:
        print(f"\nGenerating plots for {len(flagged_files)} files with flagged steady-state segments...")
        plot_counter = {}  # Track duplicate names
        
        for df, grouping, csv_path in tqdm(flagged_files, desc="Creating plots"):
            try:
                # Create base plot name
                base_name = (f"{grouping.get('unit_id', 'NA')}_"
                           f"{grouping.get('test_case', 'NA')}_"
                           f"run{grouping.get('test_run', 'NA')}_"
                           f"{grouping.get('dc_folder', 'NA')}")
                
                # Handle duplicate names by adding a counter
                if base_name in plot_counter:
                    plot_counter[base_name] += 1
                    plot_name = f"{base_name}_{plot_counter[base_name]}.png"
                else:
                    plot_counter[base_name] = 0
                    plot_name = f"{base_name}.png"
                
                plot_path = plots_folder / plot_name
                self.create_simple_plot(df, grouping, plot_path)
            except Exception as e:
                print(f"Error creating plot for {csv_path.name}: {e}")
        
        # Report actual number of plots created
        actual_plots = len(list(plots_folder.glob('*.png')))
        print(f"Actually created {actual_plots} plot files in {plots_folder}")
    else:
        print("\nNo steady-state flagged segments found - no plots to generate")
    
    # Save Excel report
    excel_path = self.output_folder / 'analysis_results.xlsx'
    try:
        with pd.ExcelWriter(excel_path) as writer:
            # All results with all statistics
            results_df.to_excel(writer, sheet_name='All_Results', index=False)
            
            # Only flagged steady state
            if 'flagged' in results_df.columns and 'label' in results_df.columns:
                flagged_steady = results_df[(results_df['flagged'] == True) & 
                                           (results_df['label'] == 'steady_state')]
                if not flagged_steady.empty:
                    flagged_steady.to_excel(writer, sheet_name='Flagged_Steady_State', index=False)
                
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
        # Count only steady state flags
        steady_flagged = results_df[(results_df['flagged']) & (results_df['label'] == 'steady_state')]
        n_steady_flagged = len(steady_flagged)
        print(f"Flagged steady-state segments: {n_steady_flagged}")
        
        if n_steady_flagged > 0:
            print("\nFlagged steady-state breakdown by test case:")
            if 'test_case' in steady_flagged.columns:
                for tc in steady_flagged['test_case'].unique():
                    count = len(steady_flagged[steady_flagged['test_case'] == tc])
                    print(f"  {tc}: {count}")
            
            # Show which were flagged by dynamic thresholds
            if dynamic_thresholds:
                dynamic_flagged = steady_flagged[steady_flagged['flag_reasons'].str.contains('dynamic', na=False)]
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
