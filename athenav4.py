def calculate_dynamic_thresholds(self, results_df):
    """
    Calculate dynamic thresholds based on OFP and test_case groupings.
    Threshold = mean + (0.5 * std) for each metric.
    """
    if results_df.empty or 'ofp' not in results_df.columns or 'test_case' not in results_df.columns:
        print("Cannot calculate dynamic thresholds - missing data or columns")
        return None
    
    # Only look at steady state segments for baseline
    steady_df = results_df[results_df['label'] == 'steady_state'].copy()
    
    if steady_df.empty:
        print("No steady state segments to calculate thresholds")
        return None
    
    # Group by OFP and test_case
    grouped_thresholds = {}
    
    for (ofp, test_case), group in steady_df.groupby(['ofp', 'test_case']):
        if len(group) < 3:  # Need at least 3 samples for meaningful statistics
            continue
        
        # Calculate mean and std for each metric
        metrics_to_threshold = ['variance', 'iqr', 'abs_slope', 'std']
        
        thresholds = {}
        for metric in metrics_to_threshold:
            if metric in group.columns:
                mean_val = group[metric].mean()
                std_val = group[metric].std()
                
                # Threshold = mean + (50% of std)
                # This means we flag anything outside mean ± 0.5*std
                threshold = mean_val + (0.5 * std_val)
                thresholds[metric] = threshold
                
                print(f"  {ofp}/{test_case} - {metric}: mean={mean_val:.4f}, std={std_val:.4f}, threshold={threshold:.4f}")
        
        grouped_thresholds[f"{ofp}_{test_case}"] = thresholds
    
    return grouped_thresholds

def check_against_dynamic_thresholds(self, row, dynamic_thresholds):
    """
    Check if a row exceeds dynamic thresholds for its OFP/test_case group.
    """
    if not dynamic_thresholds or row['label'] != 'steady_state':
        return False, []
    
    group_key = f"{row.get('ofp', 'NA')}_{row.get('test_case', 'NA')}"
    
    if group_key not in dynamic_thresholds:
        return False, []
    
    thresholds = dynamic_thresholds[group_key]
    flags = []
    
    for metric, threshold in thresholds.items():
        if metric in row and pd.notna(row[metric]):
            if row[metric] > threshold:
                flags.append(f"{metric}_exceeds_dynamic")
    
    return len(flags) > 0, flags


# After creating results_df, calculate dynamic thresholds
if not results_df.empty:
    print("\nCalculating dynamic thresholds based on OFP/test_case groups...")
    dynamic_thresholds = self.calculate_dynamic_thresholds(results_df)
    
    if dynamic_thresholds:
        # Apply dynamic thresholds to flag additional items
        print("\nApplying dynamic thresholds...")
        for idx, row in results_df.iterrows():
            is_flagged, dynamic_flags = self.check_against_dynamic_thresholds(row, dynamic_thresholds)
            
            if is_flagged:
                # Update the flagging
                if not results_df.at[idx, 'flagged']:
                    results_df.at[idx, 'flagged'] = True
                    results_df.at[idx, 'flags'] = ', '.join(dynamic_flags)
                else:
                    # Append to existing flags
                    existing_flags = results_df.at[idx, 'flags']
                    results_df.at[idx, 'flags'] = f"{existing_flags}, {', '.join(dynamic_flags)}"
