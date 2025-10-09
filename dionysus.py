def main():
    """Process each voltage column completely separately."""
    DATA_PATH = "your_raw_data.parquet"
    
    # Get list of run_ids without loading all data
    run_id_df = pd.read_parquet(DATA_PATH, columns=['run_id'])
    unique_runs = run_id_df['run_id'].unique()
    
    # Limit for debugging
    DEBUG = True
    if DEBUG:
        unique_runs = unique_runs[:6]
    
    # Process each voltage column COMPLETELY SEPARATELY
    for voltage_col in ['voltage_28v_dc1_cal', 'voltage_28v_dc2_cal']:
        print(f"\n{'='*60}")
        print(f"Processing {voltage_col}")
        print('='*60)
        
        model = SteadyStateSegmenter()
        processed_data = []
        
        # Process this voltage column for all run_ids
        for run_id in tqdm(unique_runs, desc=f"Segmenting {voltage_col}"):
            filters = [('run_id', '==', run_id)]
            run_data = pd.read_parquet(DATA_PATH, filters=filters)
            
            if voltage_col in run_data.columns:
                # Check if this column has data
                if run_data[voltage_col].notna().any():
                    model.load_data(run_data, voltage_col)
                    model.predict()
                    
                    if hasattr(model, 'data'):
                        result = model.data.copy()
                        processed_data.append(result)
        
        if processed_data:
            # Combine data for this voltage column
            voltage_data = pd.concat(processed_data, ignore_index=True)
            
            # Create separate output folder for each DC
            dc_name = 'dc1' if 'dc1' in voltage_col else 'dc2'
            output_folder = f'voltage_analysis_output/{dc_name}'
            
            # Analyze this voltage column's data
            print(f"\nAnalyzing {voltage_col}...")
            analyzer = SimplifiedVoltageAnalyzer(output_folder)
            results = analyzer.run_analysis(voltage_data)
            
            # Save results specific to this DC
            results_file = f'{output_folder}/analysis_results_{dc_name}.parquet'
            results.to_parquet(results_file)
            print(f"Saved {voltage_col} results to {results_file}")
        else:
            print(f"No data found for {voltage_col}")
    
    print("\n" + "="*60)
    print("COMPLETE - Check voltage_analysis_output/dc1 and voltage_analysis_output/dc2")
    print("="*60)
