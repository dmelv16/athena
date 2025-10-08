import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from aeon.segmentation import GreedyGaussianSegmenter
import warnings
warnings.filterwarnings('ignore')

def process_power_data(parquet_path, output_file='segmentation_output.parquet'):
    """
    Process power data with greedy Gaussian segmentation
    Output only a parquet file with required columns
    """
    
    # Define columns
    group_cols = ['unit_id', 'save', 'ofp', 'station', 'test_case', 'test_run']
    voltage_cols = ['voltage_28v_dc1_cal', 'voltage_28v_dc2_cal']
    
    # First pass: identify unique groups
    print("Identifying unique groups...")
    parquet_file = pq.ParquetFile(parquet_path)
    unique_groups = set()
    
    for batch in parquet_file.iter_batches(batch_size=100000, columns=group_cols):
        df_batch = batch.to_pandas()
        groups = df_batch[group_cols].drop_duplicates()
        unique_groups.update([tuple(row) for _, row in groups.iterrows()])
    
    print(f"Found {len(unique_groups)} unique groups to process")
    
    # List to collect all results
    all_results = []
    
    # Process each group individually
    for group_idx, group_values in enumerate(unique_groups, 1):
        print(f"Processing group {group_idx}/{len(unique_groups)}: {dict(zip(group_cols, group_values))}")
        
        # Create filter expression for this group
        filters = [
            (col, '==', val) for col, val in zip(group_cols, group_values)
        ]
        
        # Read only this group's data
        group_df = read_group_data(parquet_path, filters, group_cols, voltage_cols)
        
        if group_df is None or len(group_df) < 10:
            continue
        
        # Sort by timestamp
        group_df = group_df.sort_values('timestamp')
        
        # Process each voltage column
        for voltage_idx, col in enumerate(voltage_cols):
            if col not in group_df.columns:
                continue
            
            # Extract DC identifier from column name (dc1 or dc2)
            dc = col.replace('voltage_28v_', '').replace('_cal', '')
            
            print(f"  Processing {dc}...")
            
            # Extract voltage data and timestamps
            voltage_data = group_df[col].values
            timestamps = group_df['timestamp'].values
            
            # Remove NaN values
            valid_mask = ~np.isnan(voltage_data)
            nan_count = np.sum(~valid_mask)
            
            if nan_count > 0:
                voltage_data = voltage_data[valid_mask]
                timestamps = timestamps[valid_mask]
            
            n_points = len(voltage_data)
            
            if n_points < 10:
                print(f"    Skipping {dc} - insufficient data after NaN removal ({n_points} points)")
                continue
            
            # Calculate features
            features = calculate_complete_features(voltage_data)
            
            # Perform segmentation
            try:
                # Initialize segmenter
                segmenter = GreedyGaussianSegmenter(k_max=5, max_shuffles=1)
                
                # Transpose features for segmentation
                features_transposed = features.T
                
                # Run segmentation
                segments = segmenter.fit_predict(features_transposed)
                
                print(f"    Found {len(np.unique(segments))} segments for {dc}")
                
                # Create run_id for this group+dc combination
                # run_id is unique for each group (ofp, test_case, test_run, save, unit_id, station) + dc
                run_id = f"{group_values[2]}_{group_values[4]}_{group_values[5]}_{group_values[1]}_{group_values[0]}_{group_values[3]}_{dc}"
                
                # Create dataframe for this segmentation result
                result_df = pd.DataFrame({
                    'ofp': group_values[2],
                    'dc': dc,
                    'test_case': group_values[4],
                    'test_run': group_values[5],
                    'save': group_values[1],
                    'unit_id': group_values[0],
                    'station': group_values[3],
                    'segment': segments,
                    'voltage': voltage_data,
                    'timestamp': timestamps,
                    'run_id': run_id
                })
                
                all_results.append(result_df)
                
            except Exception as e:
                print(f"    Error segmenting {dc}: {e}")
                continue
        
        # Clear memory
        del group_df
    
    # Combine all results and save to parquet
    if all_results:
        print("\nCombining all results...")
        final_df = pd.concat(all_results, ignore_index=True)
        
        print(f"Total rows in output: {len(final_df):,}")
        print(f"Unique run_ids: {final_df['run_id'].nunique()}")
        print(f"Columns in output: {list(final_df.columns)}")
        
        # Save to parquet
        print(f"\nSaving to {output_file}...")
        final_df.to_parquet(output_file, index=False, compression='snappy')
        print(f"Successfully saved {output_file}")
        
        # Print summary statistics
        print("\nSummary:")
        print(f"  Total records: {len(final_df):,}")
        print(f"  Unique groups (run_ids): {final_df['run_id'].nunique()}")
        print(f"  Average segments per run_id: {final_df.groupby('run_id')['segment'].nunique().mean():.1f}")
        
        return final_df
    else:
        print("No results to save")
        return None

def read_group_data(parquet_path, filters, group_cols, voltage_cols):
    """
    Read data for a specific group using filters
    """
    try:
        columns_to_read = ['timestamp'] + group_cols + voltage_cols
        df = pd.read_parquet(
            parquet_path,
            filters=filters,
            columns=columns_to_read,
            engine='pyarrow'
        )
        return df
    except Exception as e:
        print(f"Error reading group data: {e}")
        return None

def calculate_complete_features(data):
    """
    Calculate rolling averages AND slopes for 3 and 5 point windows
    Returns a 2D array with shape (n_points, 4) where columns are:
    [avg_3, avg_5, slope_3, slope_5]
    """
    n = len(data)
    
    # Pre-allocate arrays for all features
    avg_3 = np.zeros(n)
    avg_5 = np.zeros(n)
    slope_3 = np.zeros(n)
    slope_5 = np.zeros(n)
    
    # Vectorized rolling averages using convolution
    # 3-point average
    kernel_3 = np.ones(3) / 3
    if n >= 3:
        avg_3[1:n-1] = np.convolve(data, kernel_3, mode='valid')
    avg_3[0] = np.mean(data[:min(2, n)])
    avg_3[-1] = np.mean(data[max(-2, -n):])
    
    # 5-point average
    kernel_5 = np.ones(5) / 5
    if n >= 5:
        avg_5[2:n-2] = np.convolve(data, kernel_5, mode='valid')
        for i in range(min(2, n)):
            avg_5[i] = np.mean(data[:min(i+3, n)])
            if n-i-1 >= 0:
                avg_5[-(i+1)] = np.mean(data[max(-(i+3), -n):])
    else:
        avg_5[:] = np.mean(data)
    
    # Calculate slopes for 3 and 5 point windows
    # 3-point slope
    for i in range(n):
        if i == 0:
            slope_3[i] = (data[min(1, n-1)] - data[0]) if n > 1 else 0
        elif i == n-1:
            slope_3[i] = (data[-1] - data[max(-2, -n)]) if n > 1 else 0
        else:
            slope_3[i] = (data[min(i+1, n-1)] - data[max(i-1, 0)]) / 2
    
    # 5-point slope
    for i in range(n):
        start_idx = max(0, i-2)
        end_idx = min(n, i+3)
        window = data[start_idx:end_idx]
        if len(window) > 1:
            x = np.arange(len(window))
            slope_5[i] = np.polyfit(x, window, 1)[0]
        else:
            slope_5[i] = 0
    
    # Stack all features as columns
    features = np.column_stack([avg_3, avg_5, slope_3, slope_5])
    
    return features

# Main execution
if __name__ == "__main__":
    # Configuration
    PARQUET_FILE = "your_data.parquet"  # Replace with your input file path
    OUTPUT_FILE = "segmentation_output.parquet"  # Output parquet file
    
    # Run processing
    print("Starting power data segmentation...")
    print("Output: Parquet file with segmentation results only")
    print("-" * 50)
    
    result_df = process_power_data(PARQUET_FILE, OUTPUT_FILE)
    
    if result_df is not None:
        print("\nProcessing complete!")
        print(f"Output saved to: {OUTPUT_FILE}")
