def load_test_case_requirements(self):
        """
        Load test case-specific requirements from TestCases folder structure.
        Each test case folder contains requirement CSVs (e.g., ps3_0070.csv).
        Only processes rows with TRUE or FALSE results.
        
        FIXED: Now correctly handles folder names with '&' by splitting them.
        """
        if not self.test_cases_folder.exists():
            print(f"Note: TestCases folder '{self.test_cases_folder}' does not exist")
            return []
        
        test_case_folders = [d for d in self.test_cases_folder.iterdir() if d.is_dir()]
        
        if not test_case_folders:
            print(f"No test case folders found in {self.test_cases_folder}")
            return []
        
        print(f"\nLoading test case requirements from {len(test_case_folders)} test case folders...")
        
        requirements_results = []
        
        for test_case_folder in test_case_folders:
            original_test_case_folder_name = test_case_folder.name  # e.g., "QS-007_02" or "QS-007_04&TW104_01"
            
            # -----------------------------------------------------------------
            # FIX: SPLIT THE FOLDER NAME BY '&'
            # -----------------------------------------------------------------
            original_ids_from_folder = [tc.strip() for tc in original_test_case_folder_name.split('&')]
            
            # Get grouped IDs just for the print statement
            grouped_ids_print = list(set([self.group_test_case_id(tc_id) for tc_id in original_ids_from_folder]))
            print(f"  Processing folder: {original_test_case_folder_name} -> Grouped IDs: {', '.join(grouped_ids_print)}")
            # -----------------------------------------------------------------
            
            # Find all CSV files in this test case folder
            csv_files = list(test_case_folder.glob("*.csv"))
            
            if not csv_files:
                continue
            
            for csv_file in csv_files:
                try:
                    # Requirement name is the CSV filename without extension
                    requirement_name = csv_file.stem  # e.g., "ps3_0070"
                    
                    # Read the CSV
                    df = pd.read_csv(csv_file)
                    
                    # Check for required columns
                    required_cols = ['unit_id', 'station', 'save', 'timestamp']
                    if not all(col in df.columns for col in required_cols):
                        print(f"    Warning: {csv_file.name} missing required columns")
                        continue
                    
                    # Check if the requirement column exists (column name = requirement name)
                    if requirement_name not in df.columns:
                        print(f"    Warning: {csv_file.name} missing column '{requirement_name}'")
                        continue
                    
                    # FILTER: Only keep rows with TRUE or FALSE in the requirement column
                    # Convert to string and uppercase for case-insensitive comparison
                    df[requirement_name] = df[requirement_name].astype(str).str.strip().str.upper()
                    
                    # Filter to only TRUE or FALSE rows
                    df_filtered = df[df[requirement_name].isin(['TRUE', 'FALSE'])].copy()
                    
                    # If no TRUE/FALSE rows exist, the requirement passed (no failures to report)
                    if df_filtered.empty:
                        # print(f"    Skipped {csv_file.name}: No TRUE/FALSE results (passed by default)") # Optional: reduce noise
                        continue
                    
                    rows_before = len(df)
                    rows_after = len(df_filtered)
                    print(f"    Loaded {csv_file.name}: {rows_after}/{rows_before} rows with TRUE/FALSE")
                    
                    # Process each filtered row
                    for _, row in df_filtered.iterrows():
                        unit_id = str(row.get('unit_id', '')).strip()
                        station = str(row.get('station', '')).strip()
                        save = str(row.get('save', '')).strip()
                        timestamp = row.get('timestamp')
                        ofp = str(row.get('ofp', '')).strip() if 'ofp' in df.columns else ''
                        
                        # Get the requirement result (already filtered to TRUE or FALSE)
                        result_value = row[requirement_name]
                        
                        # Skip if missing required fields
                        if not unit_id or not station or not save or pd.isna(timestamp):
                            continue
                        
                        # -----------------------------------------------------------------
                        # FIX: LOOP THROUGH ALL TEST CASES FROM THE FOLDER NAME
                        # -----------------------------------------------------------------
                        for original_test_case_id in original_ids_from_folder:
                            grouped_test_case_id = self.group_test_case_id(original_test_case_id)
                            
                            requirements_results.append({
                                'test_case_id': grouped_test_case_id,
                                'test_case_id_original': original_test_case_id,
                                'requirement_name': requirement_name,
                                'unit_id': unit_id,
                                'station': station,
                                'save': save,
                                'ofp': ofp,
                                'timestamp': timestamp,
                                'result': result_value,
                                'passed': result_value == 'TRUE',
                                'failed': result_value == 'FALSE'
                            })
                        # -----------------------------------------------------------------
                    
                except Exception as e:
                    print(f"    Error loading {csv_file.name}: {e}")
        
        print(f"  Total test case requirement results loaded: {len(requirements_results)}")
        
        self.test_case_requirements = requirements_results
        if requirements_results:
            self.df_test_case_requirements = pd.DataFrame(requirements_results)
            
            # Print summary statistics
            if len(requirements_results) > 0:
                total_passed = sum(1 for r in requirements_results if r['passed'])
                total_failed = sum(1 for r in requirements_results if r['failed'])
                print(f"    Summary: {total_passed} passed, {total_failed} failed")
        
        return requirements_results
