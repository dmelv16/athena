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
    
    # Get flagging info for steady state only - WITH SAVE TRACKING
    flagged_info = []
    found_flagged = False
    
    if 'steady_state' in df['label'].values:
        ss_data = df[df['label'] == 'steady_state']
        
        # Get flag reasons from our results - MATCH INCLUDING SAVE
        matching_results = [r for r in self.results 
                          if (r.get('label') == 'steady_state' and 
                              r.get('unit_id') == grouping.get('unit_id') and
                              r.get('save') == grouping.get('save') and  # IMPORTANT: Match save
                              r.get('test_run') == grouping.get('test_run') and
                              r.get('dc_folder') == grouping.get('dc_folder') and
                              r.get('test_case') == grouping.get('test_case') and
                              r.get('ofp') == grouping.get('ofp'))]
        
        # Process all matching steady state segments
        for i, result in enumerate(matching_results):
            if result.get('flagged'):
                found_flagged = True
                flagged_info.append(f"FLAGGED Segment {i+1}: {result.get('flag_reasons', 'Unknown reason')}")
                
                # Add statistics for flagged segment
                stats_text = (f"Flagged Stats: Mean={result.get('mean_voltage', 0):.2f}V | "
                            f"Std={result.get('std', 0):.3f} | "
                            f"Var={result.get('variance', 0):.3f} | "
                            f"Slope={result.get('abs_slope', 0):.4f} | "
                            f"IQR={result.get('iqr', 0):.3f}")
                flagged_info.append(stats_text)
        
        # If no flagged segments found but file was in flagged list, show first segment stats
        if not found_flagged and matching_results:
            result = matching_results[0]
            stats_text = (f"Stats: Mean={result.get('mean_voltage', 0):.2f}V | "
                        f"Std={result.get('std', 0):.3f} | "
                        f"Var={result.get('variance', 0):.3f} | "
                        f"Slope={result.get('abs_slope', 0):.4f}")
            flagged_info.append(stats_text)
    
    # Title with all info INCLUDING SAVE
    title = (f"Unit: {grouping.get('unit_id', 'NA')} | "
            f"Save: {grouping.get('save', 'NA')} | "  # Display save in title
            f"Test Case: {grouping.get('test_case', 'NA')} | "
            f"Run: {grouping.get('test_run', 'NA')} | "
            f"DC: {grouping.get('dc_folder', 'NA')}")
    
    if flagged_info:
        title += f"\n{chr(10).join(flagged_info)}"  # Use newlines for multiple flags
    
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
