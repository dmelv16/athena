def analyze_csv(self, csv_path, dynamic_thresholds=None):
    """Analyze a single CSV file with comprehensive statistics."""
    try:
        filename = csv_path.name
        grouping = self.parse_filename(filename)  # This already includes 'save'
        
        # Add folder info
        parent_path = csv_path.parent
        grouping['dc_folder'] = parent_path.name  # dc1 or dc2
        grouping['test_case_folder'] = parent_path.parent.name
        
        # Create unique identifier INCLUDING SAVE
        grouping['unique_id'] = f"{grouping.get('unit_id', 'NA')}_{grouping.get('save', 'NA')}_{grouping.get('test_case', 'NA')}_{grouping.get('test_run', 'NA')}_{grouping.get('ofp', 'NA')}_{grouping.get('dc_folder', 'NA')}"
        
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
            
            # Comprehensive metrics for all - grouping already includes 'save' and 'unique_id'
            metrics = {
                **grouping,  # This now includes 'save' and 'unique_id'
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
            
            # [Rest of the function stays the same...]


def create_simple_plot(self, df, grouping, output_path):
    """Create a detailed plot showing WHY something was flagged."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    colors = {
        'de-energized': 'gray',
        'stabilizing': 'orange',
        'steady_state': 'green',
        'unidentified': 'purple'
    }
    
    # Main voltage plot
    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        ax1.scatter(label_data['timestamp'], label_data['voltage'],
                   color=colors.get(label, 'black'),
                   label=label, s=15, alpha=0.7, edgecolors='none')
    
    # Add reference lines
    ax1.axhline(y=self.operational_min, color='red', linestyle='--', alpha=0.5, 
               label='18V threshold', linewidth=2)
    ax1.axhline(y=self.deenergized_max, color='gray', linestyle='--', alpha=0.3)
    
    # Get flagging info - match using ALL fields INCLUDING SAVE
    flagged_info = []
    found_flagged = False
    
    if 'steady_state' in df['label'].values:
        # Use either unique_id OR match all fields including save
        matching_results = [r for r in self.results 
                          if (r.get('label') == 'steady_state' and 
                              r.get('unit_id') == grouping.get('unit_id') and
                              r.get('save') == grouping.get('save') and  # ADD SAVE CHECK
                              r.get('test_run') == grouping.get('test_run') and
                              r.get('dc_folder') == grouping.get('dc_folder') and
                              r.get('test_case') == grouping.get('test_case') and
                              r.get('ofp') == grouping.get('ofp'))]
        
        # Or simpler if unique_id is available:
        # matching_results = [r for r in self.results 
        #                   if (r.get('unique_id') == grouping.get('unique_id') and
        #                       r.get('label') == 'steady_state')]
        
        for i, result in enumerate(matching_results):
            if result.get('flagged'):
                found_flagged = True
                flagged_info.append(f"FLAGGED Segment {i+1}: {result.get('flag_reasons', 'Unknown reason')}")
                
                stats_text = (f"Flagged Stats: Mean={result.get('mean_voltage', 0):.2f}V | "
                            f"Std={result.get('std', 0):.3f} | "
                            f"Var={result.get('variance', 0):.3f} | "
                            f"Slope={result.get('abs_slope', 0):.4f} | "
                            f"IQR={result.get('iqr', 0):.3f}")
                flagged_info.append(stats_text)
    
    # Title with all info INCLUDING SAVE
    title = (f"Unit: {grouping.get('unit_id', 'NA')} | "
            f"Save: {grouping.get('save', 'NA')} | "  # Display save in title
            f"Test Case: {grouping.get('test_case', 'NA')} | "
            f"Run: {grouping.get('test_run', 'NA')} | "
            f"DC: {grouping.get('dc_folder', 'NA')}")
    
    if flagged_info:
        title += f"\n{chr(10).join(flagged_info)}"
    
    ax1.set_title(title, fontsize=11, color='red' if found_flagged else 'black')
    # [Rest of plot code...]


# In run_analysis, for plot naming:
# Generate plots only for flagged files
if flagged_files:
    print(f"\nGenerating plots for {len(flagged_files)} files with flagged steady-state segments...")
    
    for df, grouping, csv_path in tqdm(flagged_files, desc="Creating plots"):
        try:
            # Use unique_id or include save in the filename
            plot_name = (f"{grouping.get('unit_id', 'NA')}_"
                        f"save{grouping.get('save', 'NA')}_"  # Include save in filename
                        f"{grouping.get('test_case', 'NA')}_"
                        f"run{grouping.get('test_run', 'NA')}_"
                        f"{grouping.get('dc_folder', 'NA')}.png")
            
            # Or use unique_id:
            # plot_name = f"{grouping.get('unique_id', csv_path.stem)}.png"
            
            plot_path = plots_folder / plot_name
            self.create_simple_plot(df, grouping, plot_path)
            
        except Exception as e:
            print(f"Error creating plot for {csv_path.name}: {e}")
