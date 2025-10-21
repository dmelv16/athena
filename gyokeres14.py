#!/usr/bin/env python3
"""
Enhanced Bus Monitor Analysis - Version 15
Includes:
1. Data word analysis with common error patterns
2. DC1/DC2 state checking for valid bus flips
3. Tracking flips with no data changes separately
4. Bus flip percentage vs total messages
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class StreamlinedBusMonitorDashboard:
    def __init__(self):
        """Initialize the Streamlined Bus Monitor Dashboard"""
        # CONFIGURATION - Change these paths as needed
        self.csv_folder = Path("./csv_data")  # <-- UPDATE THIS PATH
        self.lookup_csv_path = Path("./message_lookup.csv")  # <-- PATH TO LOOKUP CSV
        self.requirements_folder = Path("./requirements")  # <-- PATH TO REQUIREMENTS EXCEL FILES
        self.output_folder = Path("./bus_monitor_output")  # <-- OUTPUT FOLDER
        self.tca_folder = Path("./TCA")  # <-- PATH TO TCA FOLDER

        # Create output folder
        self.output_folder.mkdir(exist_ok=True)
        
        # Core data storage
        self.bus_flips = []
        self.bus_flips_no_changes = []  # Track flips with no data changes
        self.data_changes = []
        self.header_issues = []
        self.file_summary = []
        
        # Data word analysis storage
        self.data_word_issues = []  # Track all data word issues
        self.data_word_patterns = defaultdict(lambda: defaultdict(Counter))  # msg_type -> data_word -> error patterns

        # Add new storage for test case analysis
        self.test_cases = []
        self.test_case_bus_flips = []
        self.test_case_message_rates = []
        self.message_rates = []
        self.message_rate_by_test_case = []

        # Summary storage
        self.unit_summary = []
        self.station_summary = []
        self.save_summary = []
        
        # Requirements analysis storage
        self.requirements_at_risk = []
        self.requirements_summary = []
        
        # Statistics tracking
        self.total_messages_processed = 0
        self.messages_with_dc_on = 0
        self.invalid_dc_messages = 0
        
        # DataFrames
        self.df_flips = None
        self.df_flips_no_changes = None
        self.df_data_changes = None
        self.df_header_issues = None
        self.df_file_summary = None
        self.df_unit_summary = None
        self.df_station_summary = None
        self.df_save_summary = None
        self.df_station_save_matrix = None
        self.df_requirements_at_risk = None
        self.df_requirements_summary = None
        self.df_data_word_analysis = None
        self.df_data_word_patterns = None
        # Add new DataFrames
        self.df_test_cases = None
        self.df_test_case_summary = None
        self.df_test_case_bus_flips = None
        self.df_test_case_coverage = None
        self.df_message_rates = None
        self.df_message_rates_by_test_case = None
        self.df_message_rates_by_unit = None
        self.df_message_rates_by_station = None
        self.df_message_rates_by_save = None

        # Load message lookup
        self.message_header_lookup = self.load_message_lookup()
    
    def load_message_lookup(self):
        """Load the message type to header lookup table"""
        lookup = defaultdict(list)
        
        if self.lookup_csv_path.exists():
            try:
                df_lookup = pd.read_csv(self.lookup_csv_path)
                print(f"Loaded message lookup from {self.lookup_csv_path}")
                
                for msg_type, group in df_lookup.groupby('message_type'):
                    lookup[msg_type] = group['header'].tolist()
                
                print(f"  Loaded {len(lookup)} message types with header mappings")
                
            except Exception as e:
                print(f"Warning: Could not load message lookup: {e}")
        else:
            print(f"Note: No message lookup file found at {self.lookup_csv_path}")
            print("  Header validation will be skipped")
        
        return lookup
    
    def check_dc_states(self, row):
        """
        Check if DC1 or DC2 state is on (valid for bus flip detection)
        Returns True if at least one DC is on, False otherwise
        """
        dc1_on = False
        dc2_on = False
        
        if 'dc1_state' in row.index:
            dc1_val = str(row['dc1_state']).strip().upper()
            dc1_on = dc1_val in ['1', 'TRUE', 'ON', 'YES']
        
        if 'dc2_state' in row.index:
            dc2_val = str(row['dc2_state']).strip().upper()
            dc2_on = dc2_val in ['1', 'TRUE', 'ON', 'YES']
        
        return dc1_on or dc2_on
    
    def validate_header(self, msg_type: str, actual_header: str):
        """Validate if the actual header matches expected headers for message type"""
        if not self.message_header_lookup or msg_type is None:
            return True, []
        
        expected_headers = self.message_header_lookup.get(msg_type, [])
        
        if not expected_headers:
            return True, []
        
        actual_clean = str(actual_header).strip().upper() if pd.notna(actual_header) else ""
        expected_clean = [str(h).strip().upper() for h in expected_headers]
        
        is_valid = actual_clean in expected_clean
        
        return is_valid, expected_headers
    
    def parse_filename(self, filename: str):
        """Parse filename to extract unit_id, station, save, and station_num"""
        parts = filename.replace('.csv', '').split('_')
        if len(parts) >= 4:
            return {
                'unit_id': parts[0],
                'station': parts[1],
                'save': parts[2],
                'station_num': parts[3] if len(parts) > 3 else 'unknown',
                'filename': filename
            }
        return None
    
    def extract_message_type(self, decoded_desc: str):
        """Extract message type from decoded description"""
        if pd.isna(decoded_desc):
            return None, 0
            
        pattern = r'\((\d+)-\[([^\]]+)\]-(\d+)\)'
        match = re.search(pattern, str(decoded_desc))
        
        if match:
            msg_type = match.group(2)
            end_num = int(match.group(3))
            return msg_type, end_num
        
        return None, 0
    
    def extract_message_type_from_column(self, col_name: str):
        """
        Extract message type from requirements column name
        Examples: [19r] -> 19R, [27r-2] -> 27R, [19r-3] -> 19R
        """
        # Match pattern like [XXr] or [XXr-Y] or [XXt] or [XXt-Y]
        match = re.search(r'\[(\d+)([rt])', str(col_name).lower())
        if match:
            return f"{match.group(1)}{match.group(2).upper()}"
        return None
    
    def load_requirements_files(self):
        """
        Load all requirements Excel files and extract test results with pass/fail status
        """
        requirements_data = []
        
        if not self.requirements_folder.exists():
            print(f"Note: Requirements folder '{self.requirements_folder}' does not exist")
            return requirements_data
        
        excel_files = list(self.requirements_folder.glob("*_AllData.xlsx"))
        
        if not excel_files:
            print(f"No requirements Excel files found in {self.requirements_folder}")
            return requirements_data
        
        print(f"\nLoading {len(excel_files)} requirements files...")
        
        for excel_file in excel_files:
            try:
                # Extract requirement name from filename
                requirement_name = excel_file.stem.replace('_AllData', '')
                
                # Read Excel file
                df = pd.read_excel(excel_file)
                
                # Check for required columns
                required_cols = ['unit_id', 'save', 'station']
                if not all(col in df.columns for col in required_cols):
                    print(f"  Warning: {excel_file.name} missing required columns")
                    continue
                
                # Find message type columns (columns with brackets)
                msg_type_cols = [col for col in df.columns if '[' in str(col) and ']' in str(col)]
                
                # Process each row
                for _, row in df.iterrows():
                    # Get base information
                    unit_id = str(row.get('unit_id', '')).strip()
                    save = str(row.get('save', '')).strip()
                    station = str(row.get('station', '')).strip()
                    ofp = str(row.get('ofp', '')).strip() if 'ofp' in df.columns else ''
                    
                    # Extract message types and their test results
                    msg_types_tested = []
                    msg_types_passed = []
                    msg_types_failed = []
                    msg_types_mixed = []
                    test_results_detail = {}
                    
                    for col in msg_type_cols:
                        msg_type = self.extract_message_type_from_column(col)
                        if msg_type and pd.notna(row[col]):
                            cell_value = str(row[col]).strip().upper()
                            if cell_value:  # Has some test result
                                msg_types_tested.append(msg_type)
                                test_results_detail[msg_type] = cell_value
                                
                                # Categorize based on result
                                if 'TRUE' in cell_value and 'FALSE' not in cell_value and 'MIXED' not in cell_value:
                                    msg_types_passed.append(msg_type)
                                elif 'FALSE' in cell_value:
                                    msg_types_failed.append(msg_type)
                                elif 'MIXED' in cell_value:
                                    msg_types_mixed.append(msg_type)
                    
                    if unit_id and save and station:
                        requirements_data.append({
                            'requirement_name': requirement_name,
                            'unit_id': unit_id,
                            'save': save,
                            'station': station,
                            'ofp': ofp,
                            'msg_types_tested': msg_types_tested,
                            'msg_types_passed': msg_types_passed,
                            'msg_types_failed': msg_types_failed,
                            'msg_types_mixed': msg_types_mixed,
                            'msg_types_str': ', '.join(sorted(set(msg_types_tested))),
                            'test_results': test_results_detail,
                            'has_failures': len(msg_types_failed) > 0 or len(msg_types_mixed) > 0
                        })
                
                print(f"  Loaded {excel_file.name}: {len(df)} rows")
                
            except Exception as e:
                print(f"  Error loading {excel_file.name}: {e}")
        
        print(f"  Total requirements records loaded: {len(requirements_data)}")
        return requirements_data

    def analyze_failed_requirements_vs_bus_flips(self):
        """
        Analyze correlation between failed requirements tests and bus flip issues.
        This identifies which failed tests have corresponding bus flip problems.
        """
        # Load requirements data with test results
        requirements_data = self.load_requirements_files()
        
        if not requirements_data:
            print("No requirements data loaded")
            return
        
        # Filter to only requirements with failures
        failed_requirements = [req for req in requirements_data if req['has_failures']]
        
        if not failed_requirements:
            print("No failed requirements found")
            return
        
        print(f"\nFound {len(failed_requirements)} requirements with failures")
        
        # Check if we have bus flip data
        has_bus_flips = self.df_flips is not None and not self.df_flips.empty
        
        # Analyze correlation between failures and bus flips
        failure_analysis = []
        
        for req in failed_requirements:
            # Get all message types that failed or had mixed results
            problematic_msg_types = req['msg_types_failed'] + req['msg_types_mixed']
            
            for msg_type in problematic_msg_types:
                failure_record = {
                    'requirement_name': req['requirement_name'],
                    'unit_id': req['unit_id'],
                    'station': req['station'],
                    'save': req['save'],
                    'ofp': req['ofp'],
                    'msg_type': msg_type,
                    'test_result': 'FAILED' if msg_type in req['msg_types_failed'] else 'MIXED',
                    'full_result': req['test_results'].get(msg_type, ''),
                    'has_bus_flips': False,
                    'flip_count': 0,
                    'flip_types': '',
                    'data_words_affected': ''
                }
                
                # Check for bus flips if data available
                if has_bus_flips:
                    matching_flips = self.df_flips[
                        (self.df_flips['unit_id'] == req['unit_id']) &
                        (self.df_flips['station'] == req['station']) &
                        (self.df_flips['save'] == req['save']) &
                        (self.df_flips['msg_type'] == msg_type)
                    ]
                    
                    if not matching_flips.empty:
                        failure_record['has_bus_flips'] = True
                        failure_record['flip_count'] = len(matching_flips)
                        failure_record['flip_types'] = ', '.join(matching_flips['bus_transition'].unique())
                        
                        # Get affected data words
                        if 'changed_data_words' in matching_flips.columns:
                            all_words = []
                            for words in matching_flips['changed_data_words']:
                                if words and words != 'none':
                                    all_words.extend(words.split(', '))
                            unique_words = list(set(all_words))
                            failure_record['data_words_affected'] = ', '.join(unique_words[:10])
                            if len(unique_words) > 10:
                                failure_record['data_words_affected'] += f'... (+{len(unique_words)-10} more)'
                
                failure_analysis.append(failure_record)
        
        # Create DataFrame and sort by flip count
        self.df_failed_requirements_analysis = pd.DataFrame(failure_analysis)
        self.df_failed_requirements_analysis = self.df_failed_requirements_analysis.sort_values(
            ['has_bus_flips', 'flip_count'], ascending=[False, False]
        )
        
        # Create summary of failures with bus flips
        failures_with_flips = self.df_failed_requirements_analysis[
            self.df_failed_requirements_analysis['has_bus_flips'] == True
        ]
        
        if not failures_with_flips.empty:
            summary = failures_with_flips.groupby('requirement_name').agg({
                'msg_type': 'count',
                'flip_count': 'sum',
                'test_result': lambda x: 'MIXED' if 'MIXED' in x.values else 'FAILED'
            }).reset_index()
            
            summary.columns = ['requirement_name', 'failed_msg_types_with_flips', 
                            'total_flips', 'overall_result']
            summary = summary.sort_values('total_flips', ascending=False)
            
            self.df_requirements_with_bus_issues = summary
            
            print(f"\nFailure Analysis Complete:")
            print(f"  Total failed test cases analyzed: {len(failure_analysis)}")
            print(f"  Failed tests with bus flips: {self.df_failed_requirements_analysis['has_bus_flips'].sum()}")
            print(f"  Requirements with bus issues: {len(summary)}")
        else:
            print("\nNo bus flips found for failed requirements")
            self.df_requirements_with_bus_issues = pd.DataFrame()
    
    def analyze_requirements_at_risk(self):
        """
        Cross-reference bus flips with requirements and test cases to identify truly affected requirements.
        Only includes requirements where bus flips occurred during their actual test case executions.
        """
        if self.df_flips is None or self.df_flips.empty:
            print("No bus flips to analyze for requirements")
            return
        
        # Load requirements data
        requirements_data = self.load_requirements_files()
        
        if not requirements_data:
            print("No requirements data loaded")
            return
        
        # Check if we have test case data
        has_test_cases = self.df_test_cases is not None and not self.df_test_cases.empty
        
        # Create a lookup of bus flip issues by (unit_id, station, save, msg_type, timestamp)
        flip_lookup = defaultdict(list)
        for _, flip in self.df_flips.iterrows():
            if flip.get('msg_type'):
                key = (str(flip['unit_id']), str(flip['station']), str(flip['save']), str(flip['msg_type']))
                flip_lookup[key].append({
                    'bus_transition': flip.get('bus_transition', ''),
                    'timestamp_busA': flip.get('timestamp_busA', 0),
                    'timestamp_busB': flip.get('timestamp_busB', 0),
                    'timestamp_diff_ms': flip.get('timestamp_diff_ms', 0)
                })
        
        # Check each requirement against bus flips
        seen_combinations = set()
        affected_requirements = []
        
        for req in requirements_data:
            # Get unique message types for this requirement
            unique_msg_types = list(set(req['msg_types_tested']))
            
            # Check each unique message type this requirement tests
            for msg_type in unique_msg_types:
                key = (req['unit_id'], req['station'], req['save'], msg_type)
                
                if key in flip_lookup:
                    flips_info = flip_lookup[key]
                    
                    # If we have test case data, filter flips to only those during test cases
                    if has_test_cases:
                        # Get test cases for this unit/station/save combo
                        matching_test_cases = self.df_test_cases[
                            (self.df_test_cases['unit_id'] == req['unit_id']) &
                            (self.df_test_cases['station'] == req['station']) &
                            (self.df_test_cases['save'] == req['save'])
                        ]
                        
                        if matching_test_cases.empty:
                            # No test cases for this combo, skip
                            continue
                        
                        # Filter flips to only those within test case time ranges
                        filtered_flips = []
                        for flip_info in flips_info:
                            flip_time = flip_info['timestamp_busA']
                            
                            # Check if this flip occurred during any test case
                            in_test_case = False
                            test_case_ids = []
                            
                            for _, test_case in matching_test_cases.iterrows():
                                if (flip_time >= test_case['timestamp_start'] and 
                                    flip_time <= test_case['timestamp_end']):
                                    in_test_case = True
                                    test_case_ids.append(str(test_case['test_case_id']))
                            
                            if in_test_case:
                                flip_info['test_case_ids'] = test_case_ids
                                filtered_flips.append(flip_info)
                        
                        # If no flips during test cases, this requirement is not at risk
                        if not filtered_flips:
                            continue
                        
                        flips_info = filtered_flips
                    
                    # Create a unique identifier for this combination
                    combo_key = (req['requirement_name'], req['unit_id'], req['station'], 
                                req['save'], msg_type)
                    
                    # Only add if we haven't seen this combination before
                    if combo_key not in seen_combinations:
                        seen_combinations.add(combo_key)
                        
                        # Aggregate test case information if available
                        test_case_info = ''
                        test_case_count = 0
                        if has_test_cases and 'test_case_ids' in flips_info[0]:
                            all_test_case_ids = []
                            for flip in flips_info:
                                if 'test_case_ids' in flip:
                                    all_test_case_ids.extend(flip['test_case_ids'])
                            unique_test_cases = list(set(all_test_case_ids))
                            test_case_count = len(unique_test_cases)
                            test_case_info = ', '.join(unique_test_cases[:10])  # Limit to first 10
                            if len(unique_test_cases) > 10:
                                test_case_info += f'... (+{len(unique_test_cases)-10} more)'
                        
                        affected_requirements.append({
                            'requirement_name': req['requirement_name'],
                            'unit_id': req['unit_id'],
                            'station': req['station'],
                            'save': req['save'],
                            'ofp': req['ofp'],
                            'msg_type_affected': msg_type,
                            'flip_count': len(flips_info),
                            'bus_transitions': ', '.join(sorted(set([f['bus_transition'] for f in flips_info]))),
                            'test_cases_affected': test_case_count if has_test_cases else 'N/A',
                            'test_case_ids': test_case_info if test_case_info else 'N/A',
                            'in_test_case': 'Yes' if has_test_cases else 'Unknown'
                        })
        
        # Create DataFrames
        if affected_requirements:
            self.df_requirements_at_risk = pd.DataFrame(affected_requirements)
            self.df_requirements_at_risk = self.df_requirements_at_risk.sort_values(
                ['requirement_name', 'flip_count'], 
                ascending=[True, False]
            )
            
            # Create summary by requirement
            req_summary = self.df_requirements_at_risk.groupby('requirement_name').agg({
                'msg_type_affected': lambda x: ', '.join(sorted(set(x))),
                'flip_count': 'sum',
                'unit_id': 'nunique',
                'station': 'nunique',
                'save': 'nunique',
                'test_cases_affected': lambda x: sum([int(v) for v in x if v != 'N/A'])
            }).reset_index()
            
            req_summary.columns = ['requirement_name', 'affected_message_types', 'total_flips', 
                                'unique_units', 'unique_stations', 'unique_saves', 'total_test_cases_affected']
            req_summary = req_summary.sort_values('total_flips', ascending=False)
            self.df_requirements_summary = req_summary
            
            print(f"\nRequirements Analysis Complete:")
            print(f"  Total affected requirement entries: {len(affected_requirements)}")
            print(f"  Unique requirements with issues: {len(self.df_requirements_summary)}")
            if has_test_cases:
                print(f"  Requirements filtered to test case timeframes only")
        else:
            print("  No requirements at risk found (no bus flips during test cases)")
            self.df_requirements_at_risk = pd.DataFrame()
            self.df_requirements_summary = pd.DataFrame()
    
    def analyze_data_word_patterns(self):
        """Analyze data word error patterns and create comprehensive summaries"""
        if not self.data_word_issues:
            return
        
        # Create DataFrame from data word issues
        df_issues = pd.DataFrame(self.data_word_issues)
        
        # Analysis 1: Basic data word analysis by message type and data word
        grouped = df_issues.groupby(['msg_type', 'data_word'])
        
        analysis_results = []
        for (msg_type, data_word), group in grouped:
            # Count unique error patterns (value_before -> value_after)
            error_patterns = group.apply(lambda x: f"{x['value_before']} -> {x['value_after']}", axis=1)
            pattern_counts = error_patterns.value_counts()
            
            # Get top 3 most common error patterns
            top_patterns = []
            for pattern, count in pattern_counts.head(3).items():
                top_patterns.append(f"{pattern} ({count}x)")
            
            # Calculate percentage of total issues for this combination
            issue_percentage = (len(group) / len(df_issues)) * 100 if len(df_issues) > 0 else 0
            
            # Analyze if this data word is part of multi-word changes
            multi_word_count = group[group['num_data_changes'] > 1].shape[0]
            single_word_count = group[group['num_data_changes'] == 1].shape[0]
            
            # Get average flip speed for this data word
            avg_flip_speed = group['timestamp_diff_ms'].mean() if 'timestamp_diff_ms' in group.columns else 0
            
            analysis_results.append({
                'msg_type': msg_type,
                'data_word': data_word,
                'total_issues': len(group),
                'issue_percentage': round(issue_percentage, 2),
                'unique_patterns': len(pattern_counts),
                'top_error_patterns': ' | '.join(top_patterns),
                'most_common_error': pattern_counts.index[0] if len(pattern_counts) > 0 else 'N/A',
                'most_common_count': pattern_counts.iloc[0] if len(pattern_counts) > 0 else 0,
                'affected_units': group['unit_id'].nunique(),
                'affected_stations': group['station'].nunique(),
                'affected_saves': group['save'].nunique(),
                'multi_word_changes': multi_word_count,
                'single_word_changes': single_word_count,
                'avg_flip_speed_ms': round(avg_flip_speed, 3)
            })
        
        self.df_data_word_analysis = pd.DataFrame(analysis_results)
        if not self.df_data_word_analysis.empty:
            self.df_data_word_analysis = self.df_data_word_analysis.sort_values('total_issues', ascending=False)
        
        # Analysis 2: Multi-word change analysis
        if 'num_data_changes' in df_issues.columns:
            # Group by message type to see which have multi-word issues
            multi_word_analysis = []
            for msg_type in df_issues['msg_type'].unique():
                msg_data = df_issues[df_issues['msg_type'] == msg_type]
                
                # Count single vs multi-word changes
                single_changes = msg_data[msg_data['num_data_changes'] == 1].shape[0]
                multi_changes = msg_data[msg_data['num_data_changes'] > 1].shape[0]
                
                # Get the most common number of words that change together
                if multi_changes > 0:
                    multi_only = msg_data[msg_data['num_data_changes'] > 1]
                    most_common_multi = multi_only['num_data_changes'].mode().iloc[0] if len(multi_only) > 0 else 0
                else:
                    most_common_multi = 0
                
                multi_word_analysis.append({
                    'msg_type': msg_type,
                    'single_word_changes': single_changes,
                    'multi_word_changes': multi_changes,
                    'most_common_multi_count': most_common_multi,
                    'multi_word_percentage': round((multi_changes / (single_changes + multi_changes)) * 100, 2) if (single_changes + multi_changes) > 0 else 0
                })
            
            self.df_multi_word_analysis = pd.DataFrame(multi_word_analysis)
            if not self.df_multi_word_analysis.empty:
                self.df_multi_word_analysis = self.df_multi_word_analysis.sort_values('multi_word_changes', ascending=False)
        
        # Analysis 3: Station/Save/Unit combinations with most issues
        location_analysis = df_issues.groupby(['unit_id', 'station', 'save']).agg({
            'data_word': 'count',
            'msg_type': 'nunique',
            'num_data_changes': 'mean' if 'num_data_changes' in df_issues.columns else 'count'
        }).reset_index()
        
        location_analysis.columns = ['unit_id', 'station', 'save', 'total_data_issues', 'unique_msg_types', 'avg_words_per_flip']
        self.df_location_data_issues = location_analysis.sort_values('total_data_issues', ascending=False)
        
        # Create pattern frequency table
        pattern_results = []
        for (msg_type, data_word), group in grouped:
            for _, row in group.iterrows():
                pattern = f"{row['value_before']} -> {row['value_after']}"
                pattern_results.append({
                    'msg_type': msg_type,
                    'data_word': data_word,
                    'error_pattern': pattern,
                    'unit_id': row['unit_id'],
                    'station': row['station'],
                    'save': row['save'],
                    'num_data_changes': row.get('num_data_changes', 1)
                })
        
        if pattern_results:
            df_patterns = pd.DataFrame(pattern_results)
            pattern_freq = df_patterns.groupby(['msg_type', 'data_word', 'error_pattern']).agg({
                'unit_id': 'count',
                'num_data_changes': 'mean'
            }).reset_index()
            pattern_freq.columns = ['msg_type', 'data_word', 'error_pattern', 'frequency', 'avg_words_in_flip']
            self.df_data_word_patterns = pattern_freq.sort_values(['msg_type', 'data_word', 'frequency'], 
                                                                 ascending=[True, True, False])
    
    def detect_bus_flips(self, df: pd.DataFrame, file_info: dict):
        """Detect rapid bus flips with matching decoded_description"""
        flips = []
        flips_no_changes = []
        
        if 'timestamp' not in df.columns or 'decoded_description' not in df.columns:
            return flips
        
        # Check for DC state columns
        has_dc_columns = 'dc1_state' in df.columns or 'dc2_state' in df.columns
        
        df = df.copy()
        df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        if len(df) < 2:
            return flips
        
        # Vectorized operations
        df['prev_bus'] = df['bus'].shift(1)
        df['prev_timestamp'] = df['timestamp'].shift(1)
        df['prev_decoded'] = df['decoded_description'].shift(1)
        df['time_diff_ms'] = (df['timestamp'] - df['prev_timestamp']) * 1000
        
        # Find flips with matching decoded_description
        mask = (
            (df['bus'] != df['prev_bus']) & 
            (df['time_diff_ms'] < 100) & 
            (df['time_diff_ms'].notna()) &
            (df['decoded_description'] == df['prev_decoded'])
        )
        
        flip_indices = df[mask].index.tolist()
        
        for idx in flip_indices:
            curr_row = df.iloc[idx]
            prev_row = df.iloc[idx - 1]
            
            # Check DC states if columns exist
            if has_dc_columns:
                prev_dc_valid = self.check_dc_states(prev_row)
                curr_dc_valid = self.check_dc_states(curr_row)
                
                # Skip if both DCs are off
                if not prev_dc_valid and not curr_dc_valid:
                    self.invalid_dc_messages += 2
                    continue
            
            msg_type, _ = self.extract_message_type(curr_row.get('decoded_description'))
            
            # Determine bus transition direction
            bus_transition = f"{prev_row['bus']} to {curr_row['bus']}"
            
            # Determine which timestamp belongs to which bus
            timestamp_busA = prev_row['timestamp'] if prev_row['bus'] == 'A' else curr_row['timestamp']
            timestamp_busB = prev_row['timestamp'] if prev_row['bus'] == 'B' else curr_row['timestamp']
            timestamp_diff = abs(curr_row['timestamp'] - prev_row['timestamp'])
            
            # Check for data changes and count them
            data_changes = self.compare_data_words(prev_row, curr_row)
            has_data_changes = len(data_changes) > 0
            num_data_changes = len(data_changes)
            
            # Get list of changed data words
            changed_data_words = list(data_changes.keys()) if data_changes else []
            
            # Check header validation
            if 'data01' in df.columns and msg_type:
                is_valid_prev, expected = self.validate_header(msg_type, prev_row.get('data01'))
                is_valid_curr, _ = self.validate_header(msg_type, curr_row.get('data01'))
                
                if not is_valid_prev or not is_valid_curr:
                    self.header_issues.append({
                        'unit_id': file_info['unit_id'],
                        'station': file_info['station'],
                        'save': file_info['save'],
                        'bus_transition': bus_transition,
                        'timestamp_busA': timestamp_busA,
                        'timestamp_busB': timestamp_busB,
                        'timestamp_diff': round(timestamp_diff, 6),
                        'msg_type': msg_type,
                        'actual_header_busA': prev_row.get('data01') if prev_row['bus'] == 'A' else curr_row.get('data01'),
                        'actual_header_busB': prev_row.get('data01') if prev_row['bus'] == 'B' else curr_row.get('data01'),
                        'expected_headers': ', '.join(map(str, expected))
                    })
            
            # Create flip record with enhanced tracking
            flip_info = {
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'bus_transition': bus_transition,
                'timestamp_busA': timestamp_busA,
                'timestamp_busB': timestamp_busB,
                'timestamp_diff': round(timestamp_diff, 6),
                'timestamp_diff_ms': round(timestamp_diff * 1000, 3),  # Add milliseconds for easier reading
                'msg_type': msg_type,
                'decoded_description': curr_row['decoded_description'],
                'has_data_changes': has_data_changes,
                'num_data_changes': num_data_changes,
                'changed_data_words': ', '.join(changed_data_words) if changed_data_words else 'none'
            }
            
            # Separate tracking based on whether data changes occurred
            if has_data_changes:
                flips.append(flip_info)
                
                # Track data changes with enhanced info
                for col, change in data_changes.items():
                    value_busA = change['before'] if prev_row['bus'] == 'A' else change['after']
                    value_busB = change['before'] if prev_row['bus'] == 'B' else change['after']
                    
                    self.data_changes.append({
                        'unit_id': file_info['unit_id'],
                        'station': file_info['station'],
                        'save': file_info['save'],
                        'bus_transition': bus_transition,
                        'timestamp_busA': timestamp_busA,
                        'timestamp_busB': timestamp_busB,
                        'timestamp_diff': round(timestamp_diff, 6),
                        'msg_type': msg_type,
                        'data_column': col,
                        'value_busA': value_busA,
                        'value_busB': value_busB,
                        'num_data_changes': num_data_changes  # Track how many words changed in this flip
                    })
                    
                    # Track for data word analysis
                    self.data_word_issues.append({
                        'unit_id': file_info['unit_id'],
                        'station': file_info['station'],
                        'save': file_info['save'],
                        'msg_type': msg_type,
                        'data_word': col,
                        'value_before': change['before'],
                        'value_after': change['after'],
                        'bus_before': change['bus_before'],
                        'bus_after': change['bus_after'],
                        'num_data_changes': num_data_changes,  # Track if this was part of multi-word change
                        'timestamp_diff_ms': round(timestamp_diff * 1000, 3)
                    })
            else:
                # Track flips with no data changes separately
                flips_no_changes.append(flip_info)
        
        # Add flips with no changes to the separate list
        if flips_no_changes:
            self.bus_flips_no_changes.extend(flips_no_changes)
        
        return flips
    
    def compare_data_words(self, row1: pd.Series, row2: pd.Series):
        """Compare data word columns between two rows"""
        changes = {}
        
        data_cols = [col for col in row1.index if col.startswith('data') and 
                    len(col) > 4 and col[4:].replace('0', '').isdigit()]
        
        for col in data_cols:
            if col in row2.index:
                val1, val2 = row1[col], row2[col]
                
                if pd.isna(val1) and pd.isna(val2):
                    continue
                    
                if str(val1) != str(val2):
                    changes[col] = {
                        'before': val1,
                        'after': val2,
                        'bus_before': row1['bus'],
                        'bus_after': row2['bus']
                    }
        
        return changes
    
    def process_csv(self, csv_path: Path):
        """Process a single CSV file"""
        file_info = self.parse_filename(csv_path.name)
        if not file_info:
            return None
        
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            
            required_cols = ['bus', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                return None
            
            # Track total messages
            self.total_messages_processed += len(df)
            
            # Track messages with DC on
            if 'dc1_state' in df.columns or 'dc2_state' in df.columns:
                dc_valid_count = df.apply(self.check_dc_states, axis=1).sum()
                self.messages_with_dc_on += dc_valid_count
            
            flips = self.detect_bus_flips(df, file_info)
            if flips:
                self.bus_flips.extend(flips)
            
            bus_counts = df['bus'].value_counts()
            
            # Count total flips (including those without data changes)
            flips_no_changes_count = len([f for f in self.bus_flips_no_changes 
                                         if f['unit_id'] == file_info['unit_id'] 
                                         and f['station'] == file_info['station'] 
                                         and f['save'] == file_info['save']])
            
            return {
                'filename': csv_path.name,
                'unit_id': file_info['unit_id'],
                'station': file_info['station'],
                'save': file_info['save'],
                'total_rows': len(df),
                'bus_flips': len(flips),
                'bus_flips_no_changes': flips_no_changes_count,
                'bus_a_count': bus_counts.get('A', 0),
                'bus_b_count': bus_counts.get('B', 0),
                'unique_messages': df['decoded_description'].nunique() if 'decoded_description' in df.columns else 0
            }
            
        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}")
            return None
    
    def create_summaries(self):
        """Create summary dataframes at different levels"""
        if not self.df_file_summary.empty:
            # Unit ID Summary
            unit_agg = self.df_file_summary.groupby('unit_id').agg({
                'total_rows': 'sum',
                'bus_flips': 'sum',
                'bus_flips_no_changes': 'sum',
                'bus_a_count': 'sum',
                'bus_b_count': 'sum',
                'unique_messages': 'sum',
                'filename': 'count',
                'station': 'nunique',
                'save': 'nunique'
            }).reset_index()
            
            unit_agg.columns = ['unit_id', 'total_rows', 'total_flips', 'flips_no_changes', 
                               'bus_a_total', 'bus_b_total', 'total_unique_messages', 
                               'file_count', 'station_count', 'save_count']
            
            self.df_unit_summary = unit_agg.sort_values('total_flips', ascending=False)
            
            # Station Summary (includes unit_id)
            station_agg = self.df_file_summary.groupby(['unit_id', 'station']).agg({
                'total_rows': 'sum',
                'bus_flips': 'sum',
                'bus_flips_no_changes': 'sum',
                'bus_a_count': 'sum',
                'bus_b_count': 'sum',
                'unique_messages': 'sum',
                'filename': 'count',
                'save': 'nunique'
            }).reset_index()
            
            station_agg.columns = ['unit_id', 'station', 'total_rows', 'total_flips', 
                                  'flips_no_changes', 'bus_a_total', 'bus_b_total', 
                                  'total_unique_messages', 'file_count', 'save_count']
            
            self.df_station_summary = station_agg.sort_values('total_flips', ascending=False)
            
            # Save Summary (includes unit_id and station)
            save_agg = self.df_file_summary.groupby(['unit_id', 'station', 'save']).agg({
                'total_rows': 'sum',
                'bus_flips': 'sum',
                'bus_flips_no_changes': 'sum',
                'bus_a_count': 'sum',
                'bus_b_count': 'sum',
                'unique_messages': 'sum',
                'filename': 'count'
            }).reset_index()
            
            save_agg.columns = ['unit_id', 'station', 'save', 'total_rows', 'total_flips', 
                               'flips_no_changes', 'bus_a_total', 'bus_b_total', 
                               'total_unique_messages', 'file_count']
            
            self.df_save_summary = save_agg.sort_values('total_flips', ascending=False)
    
    def create_station_save_matrix(self):
        """Create a matrix view of stations vs saves with flip counts per unit_id"""
        if self.df_flips is not None and not self.df_flips.empty:
            # Group by unit_id first, then create pivot for each
            matrices = {}
            for unit_id in self.df_flips['unit_id'].unique():
                unit_flips = self.df_flips[self.df_flips['unit_id'] == unit_id]
                matrix = unit_flips.groupby(['station', 'save']).size().reset_index(name='flip_count')
                pivot = matrix.pivot(index='station', columns='save', values='flip_count').fillna(0).astype(int)
                
                # Add totals
                pivot['TOTAL'] = pivot.sum(axis=1)
                pivot.loc['TOTAL'] = pivot.sum(axis=0)
                
                matrices[unit_id] = pivot
            
            # Combine all unit matrices (for the main matrix sheet)
            combined = self.df_flips.groupby(['station', 'save']).size().reset_index(name='flip_count')
            self.df_station_save_matrix = combined.pivot(index='station', columns='save', values='flip_count').fillna(0).astype(int)
            self.df_station_save_matrix['TOTAL'] = self.df_station_save_matrix.sum(axis=1)
            self.df_station_save_matrix.loc['TOTAL'] = self.df_station_save_matrix.sum(axis=0)

    def load_tca_files(self):
        """Load all TCA CSV files and extract test case information"""
        if not self.tca_folder.exists():
            print(f"Note: TCA folder '{self.tca_folder}' does not exist")
            return []
        
        tca_files = list(self.tca_folder.glob("*_tca-data.csv"))
        
        if not tca_files:
            print(f"No TCA files found in {self.tca_folder}")
            return []
        
        print(f"\nLoading {len(tca_files)} TCA files...")
        
        test_case_data = []
        
        for tca_file in tca_files:
            try:
                # Parse filename: RSFC_L1_4_tca-data.csv
                parts = tca_file.stem.replace('_tca-data', '').split('_')
                if len(parts) >= 3:
                    unit_id = parts[0]
                    station = parts[1]
                    save = parts[2]
                else:
                    print(f"  Warning: Could not parse filename {tca_file.name}")
                    continue
                
                # Read TCA file
                df_tca = pd.read_csv(tca_file)
                
                # Check for required columns
                required_cols = ['platform', 'test_event_id', 'test_case_id', 
                               'timestamp_start', 'test_case_analysis_timestamp_end']
                
                if not all(col in df_tca.columns for col in required_cols):
                    print(f"  Warning: {tca_file.name} missing required columns")
                    continue
                
                # Process each test case
                for _, row in df_tca.iterrows():
                    test_case_data.append({
                        'unit_id': unit_id,
                        'station': station,
                        'save': save,
                        'platform': row.get('platform', ''),
                        'ofp': row.get('ofp', ''),
                        'test_event_id': row.get('test_event_id'),
                        'test_case_id': row.get('test_case_id'),
                        'timestamp_start': row.get('timestamp_start'),
                        'timestamp_end': row.get('test_case_analysis_timestamp_end'),
                        'duration': row.get('test_case_analysis_timestamp_end', 0) - row.get('timestamp_start', 0) 
                                   if pd.notna(row.get('test_case_analysis_timestamp_end')) and pd.notna(row.get('timestamp_start')) else 0
                    })
                
                print(f"  Loaded {tca_file.name}: {len(df_tca)} test cases")
                
            except Exception as e:
                print(f"  Error loading {tca_file.name}: {e}")
        
        self.test_cases = test_case_data
        if test_case_data:
            self.df_test_cases = pd.DataFrame(test_case_data)
        
        print(f"  Total test cases loaded: {len(test_case_data)}")
        return test_case_data
    
    def analyze_test_case_coverage(self):
        """Analyze what percentage of each file's messages were covered by test cases"""
        if self.df_test_cases is None or self.df_test_cases.empty:
            print("No test cases loaded")
            return
        
        coverage_data = []
        
        for csv_file in self.csv_folder.glob("*.csv"):
            file_info = self.parse_filename(csv_file.name)
            if not file_info:
                continue
            
            try:
                # Read the main data file
                df = pd.read_csv(csv_file, low_memory=False)
                
                if 'timestamp' not in df.columns:
                    continue
                
                # Get test cases for this file
                matching_test_cases = self.df_test_cases[
                    (self.df_test_cases['unit_id'] == file_info['unit_id']) &
                    (self.df_test_cases['station'] == file_info['station']) &
                    (self.df_test_cases['save'] == file_info['save'])
                ]
                
                if matching_test_cases.empty:
                    coverage_data.append({
                        'unit_id': file_info['unit_id'],
                        'station': file_info['station'],
                        'save': file_info['save'],
                        'filename': csv_file.name,
                        'total_messages': len(df),
                        'test_cases_run': 0,
                        'messages_in_test_cases': 0,
                        'coverage_percentage': 0,
                        'test_case_ids': ''
                    })
                    continue
                
                # Calculate coverage for each test case
                messages_in_test_cases = 0
                test_case_details = []
                
                for _, test_case in matching_test_cases.iterrows():
                    # Count messages within test case time range
                    mask = (
                        (df['timestamp'] >= test_case['timestamp_start']) &
                        (df['timestamp'] <= test_case['timestamp_end'])
                    )
                    messages_in_range = df[mask].shape[0]
                    messages_in_test_cases += messages_in_range
                    
                    test_case_details.append({
                        'test_case_id': test_case['test_case_id'],
                        'messages_count': messages_in_range
                    })
                
                # Remove duplicate message counts if test cases overlap
                # Get unique timestamps within all test case ranges
                all_masks = pd.Series([False] * len(df))
                for _, test_case in matching_test_cases.iterrows():
                    mask = (
                        (df['timestamp'] >= test_case['timestamp_start']) &
                        (df['timestamp'] <= test_case['timestamp_end'])
                    )
                    all_masks = all_masks | mask
                
                unique_messages_in_test_cases = df[all_masks].shape[0]
                
                coverage_data.append({
                    'unit_id': file_info['unit_id'],
                    'station': file_info['station'],
                    'save': file_info['save'],
                    'filename': csv_file.name,
                    'total_messages': len(df),
                    'test_cases_run': len(matching_test_cases),
                    'messages_in_test_cases': unique_messages_in_test_cases,
                    'coverage_percentage': round((unique_messages_in_test_cases / len(df)) * 100, 2) if len(df) > 0 else 0,
                    'test_case_ids': ', '.join([str(tc['test_case_id']) for tc in test_case_details])
                })
                
            except Exception as e:
                print(f"Error analyzing coverage for {csv_file.name}: {e}")
        
        if coverage_data:
            self.df_test_case_coverage = pd.DataFrame(coverage_data)
            self.df_test_case_coverage = self.df_test_case_coverage.sort_values('coverage_percentage', ascending=False)
    
    def analyze_test_case_bus_flips(self):
        """Analyze bus flips that occurred during each test case"""
        if self.df_test_cases is None or self.df_test_cases.empty:
            print("No test cases loaded")
            return
        
        if self.df_flips is None or self.df_flips.empty:
            print("No bus flips to analyze")
            return
        
        test_case_flip_data = []
        
        # For each test case, find bus flips within its time range
        for _, test_case in self.df_test_cases.iterrows():
            # Find matching bus flips
            matching_flips = self.df_flips[
                (self.df_flips['unit_id'] == test_case['unit_id']) &
                (self.df_flips['station'] == test_case['station']) &
                (self.df_flips['save'] == test_case['save'])
            ]
            
            if matching_flips.empty:
                continue
            
            # Filter by timestamp range
            flips_in_range = matching_flips[
                (matching_flips['timestamp_busA'] >= test_case['timestamp_start']) &
                (matching_flips['timestamp_busA'] <= test_case['timestamp_end'])
            ]
            
            if flips_in_range.empty:
                continue
            
            # Analyze message types affected
            msg_types_affected = flips_in_range['msg_type'].value_counts()
            
            # Analyze data words affected
            data_words_affected = []
            if 'changed_data_words' in flips_in_range.columns:
                for words in flips_in_range['changed_data_words']:
                    if words and words != 'none':
                        data_words_affected.extend(words.split(', '))
            
            data_word_counts = Counter(data_words_affected)
            
            test_case_flip_data.append({
                'test_case_id': test_case['test_case_id'],
                'test_event_id': test_case['test_event_id'],
                'unit_id': test_case['unit_id'],
                'station': test_case['station'],
                'save': test_case['save'],
                'platform': test_case['platform'],
                'test_case_duration': test_case['duration'],
                'total_bus_flips': len(flips_in_range),
                'unique_msg_types_affected': len(msg_types_affected),
                'msg_types_list': ', '.join(msg_types_affected.index.astype(str).tolist()),
                'most_common_msg_type': msg_types_affected.index[0] if len(msg_types_affected) > 0 else '',
                'most_common_msg_type_flips': msg_types_affected.iloc[0] if len(msg_types_affected) > 0 else 0,
                'unique_data_words_affected': len(data_word_counts),
                'most_common_data_word': data_word_counts.most_common(1)[0][0] if data_word_counts else '',
                'most_common_data_word_count': data_word_counts.most_common(1)[0][1] if data_word_counts else 0,
                'top_data_words': ', '.join([f"{word}({count})" for word, count in data_word_counts.most_common(5)])
            })
            
            # Also track individual bus flips with test case association
            for _, flip in flips_in_range.iterrows():
                self.test_case_bus_flips.append({
                    'test_case_id': test_case['test_case_id'],
                    'test_event_id': test_case['test_event_id'],
                    'unit_id': flip['unit_id'],
                    'station': flip['station'],
                    'save': flip['save'],
                    'msg_type': flip['msg_type'],
                    'bus_transition': flip['bus_transition'],
                    'timestamp_busA': flip['timestamp_busA'],
                    'timestamp_busB': flip['timestamp_busB'],
                    'timestamp_diff_ms': flip.get('timestamp_diff_ms', 0),
                    'changed_data_words': flip.get('changed_data_words', ''),
                    'num_data_changes': flip.get('num_data_changes', 0)
                })
        
        if test_case_flip_data:
            self.df_test_case_bus_flips = pd.DataFrame(test_case_flip_data)
            self.df_test_case_bus_flips = self.df_test_case_bus_flips.sort_values('total_bus_flips', ascending=False)
    
    def calculate_message_rates(self):
        """Calculate message rates at various granularities"""
        message_rate_data = []
        
        for csv_file in self.csv_folder.glob("*.csv"):
            file_info = self.parse_filename(csv_file.name)
            if not file_info:
                continue
            
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                
                if 'timestamp' not in df.columns or 'decoded_description' not in df.columns:
                    continue
                
                # Extract message types
                df['msg_type'] = df['decoded_description'].apply(lambda x: self.extract_message_type(x)[0])
                df = df[df['msg_type'].notna()]
                
                if df.empty:
                    continue
                
                # Calculate overall rates for each message type
                for msg_type in df['msg_type'].unique():
                    msg_df = df[df['msg_type'] == msg_type].copy()
                    msg_df = msg_df.sort_values('timestamp')
                    
                    if len(msg_df) < 2:
                        continue
                    
                    # Calculate time differences between consecutive messages
                    msg_df['time_diff'] = msg_df['timestamp'].diff()
                    time_diffs = msg_df['time_diff'].dropna()
                    
                    if time_diffs.empty:
                        continue
                    
                    # Calculate rates (messages per second)
                    rates = 1 / time_diffs[time_diffs > 0]
                    
                    if rates.empty:
                        continue
                    
                    # Get bus flip info for this message type
                    flip_count = 0
                    flip_avg_rate = 0
                    if self.df_flips is not None and not self.df_flips.empty:
                        msg_flips = self.df_flips[
                            (self.df_flips['unit_id'] == file_info['unit_id']) &
                            (self.df_flips['station'] == file_info['station']) &
                            (self.df_flips['save'] == file_info['save']) &
                            (self.df_flips['msg_type'] == msg_type)
                        ]
                        flip_count = len(msg_flips)
                        if flip_count > 0:
                            flip_rates = 1000 / msg_flips['timestamp_diff_ms'][msg_flips['timestamp_diff_ms'] > 0]
                            flip_avg_rate = flip_rates.mean() if not flip_rates.empty else 0
                    
                    message_rate_data.append({
                        'unit_id': file_info['unit_id'],
                        'station': file_info['station'],
                        'save': file_info['save'],
                        'msg_type': msg_type,
                        'total_messages': len(msg_df),
                        'min_rate': round(rates.min(), 3),
                        'max_rate': round(rates.max(), 3),
                        'mean_rate': round(rates.mean(), 3),
                        'std_rate': round(rates.std(), 3),
                        'median_rate': round(rates.median(), 3),
                        'bus_flip_count': flip_count,
                        'bus_flip_avg_rate': round(flip_avg_rate, 3)
                    })
                
            except Exception as e:
                print(f"Error calculating message rates for {csv_file.name}: {e}")
        
        if message_rate_data:
            self.df_message_rates = pd.DataFrame(message_rate_data)
            
            # Create aggregated views
            self.create_message_rate_aggregations()
    
    def create_message_rate_aggregations(self):
        """Create aggregated message rate views at different levels"""
        if self.df_message_rates is None or self.df_message_rates.empty:
            return
        
        # By Unit ID
        unit_agg = self.df_message_rates.groupby(['unit_id', 'msg_type']).agg({
            'total_messages': 'sum',
            'min_rate': 'min',
            'max_rate': 'max',
            'mean_rate': 'mean',
            'std_rate': 'mean',
            'median_rate': 'median',
            'bus_flip_count': 'sum',
            'bus_flip_avg_rate': 'mean'
        }).reset_index()
        self.df_message_rates_by_unit = unit_agg.sort_values(['unit_id', 'msg_type'])
        
        # By Station
        station_agg = self.df_message_rates.groupby(['unit_id', 'station', 'msg_type']).agg({
            'total_messages': 'sum',
            'min_rate': 'min',
            'max_rate': 'max',
            'mean_rate': 'mean',
            'std_rate': 'mean',
            'median_rate': 'median',
            'bus_flip_count': 'sum',
            'bus_flip_avg_rate': 'mean'
        }).reset_index()
        self.df_message_rates_by_station = station_agg.sort_values(['unit_id', 'station', 'msg_type'])
        
        # By Save
        self.df_message_rates_by_save = self.df_message_rates.copy()
        self.df_message_rates_by_save = self.df_message_rates_by_save.sort_values(['unit_id', 'station', 'save', 'msg_type'])
    
    def calculate_test_case_message_rates(self):
        """Calculate message rates specifically during test cases"""
        if self.df_test_cases is None or self.df_test_cases.empty:
            print("No test cases loaded")
            return
        
        test_case_rate_data = []
        
        for csv_file in self.csv_folder.glob("*.csv"):
            file_info = self.parse_filename(csv_file.name)
            if not file_info:
                continue
            
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                
                if 'timestamp' not in df.columns or 'decoded_description' not in df.columns:
                    continue
                
                # Extract message types
                df['msg_type'] = df['decoded_description'].apply(lambda x: self.extract_message_type(x)[0])
                df = df[df['msg_type'].notna()]
                
                if df.empty:
                    continue
                
                # Get test cases for this file
                matching_test_cases = self.df_test_cases[
                    (self.df_test_cases['unit_id'] == file_info['unit_id']) &
                    (self.df_test_cases['station'] == file_info['station']) &
                    (self.df_test_cases['save'] == file_info['save'])
                ]
                
                # Calculate rates for each test case
                for _, test_case in matching_test_cases.iterrows():
                    # Filter messages within test case time range
                    test_df = df[
                        (df['timestamp'] >= test_case['timestamp_start']) &
                        (df['timestamp'] <= test_case['timestamp_end'])
                    ].copy()
                    
                    if test_df.empty:
                        continue
                    
                    # Calculate rates for each message type in this test case
                    for msg_type in test_df['msg_type'].unique():
                        msg_df = test_df[test_df['msg_type'] == msg_type].copy()
                        msg_df = msg_df.sort_values('timestamp')
                        
                        if len(msg_df) < 2:
                            continue
                        
                        # Calculate time differences
                        msg_df['time_diff'] = msg_df['timestamp'].diff()
                        time_diffs = msg_df['time_diff'].dropna()
                        
                        if time_diffs.empty:
                            continue
                        
                        # Calculate rates
                        rates = 1 / time_diffs[time_diffs > 0]
                        
                        if rates.empty:
                            continue
                        
                        test_case_rate_data.append({
                            'test_case_id': test_case['test_case_id'],
                            'test_event_id': test_case['test_event_id'],
                            'unit_id': file_info['unit_id'],
                            'station': file_info['station'],
                            'save': file_info['save'],
                            'platform': test_case['platform'],
                            'msg_type': msg_type,
                            'total_messages': len(msg_df),
                            'min_rate': round(rates.min(), 3),
                            'max_rate': round(rates.max(), 3),
                            'mean_rate': round(rates.mean(), 3),
                            'std_rate': round(rates.std(), 3),
                            'median_rate': round(rates.median(), 3),
                            'test_case_duration': test_case['duration']
                        })
                
            except Exception as e:
                print(f"Error calculating test case message rates for {csv_file.name}: {e}")
        
        if test_case_rate_data:
            self.df_message_rates_by_test_case = pd.DataFrame(test_case_rate_data)
            self.df_message_rates_by_test_case = self.df_message_rates_by_test_case.sort_values(
                ['test_case_id', 'msg_type']
            )
    
    def create_test_case_summary(self):
        """Create a comprehensive summary of test cases"""
        if self.df_test_cases is None or self.df_test_cases.empty:
            return
        
        summary_data = []
        
        # Group by unit, station, save combo
        grouped = self.df_test_cases.groupby(['unit_id', 'station', 'save'])
        
        for (unit_id, station, save), group in grouped:
            # Get coverage info
            coverage_info = None
            if self.df_test_case_coverage is not None:
                coverage_match = self.df_test_case_coverage[
                    (self.df_test_case_coverage['unit_id'] == unit_id) &
                    (self.df_test_case_coverage['station'] == station) &
                    (self.df_test_case_coverage['save'] == save)
                ]
                if not coverage_match.empty:
                    coverage_info = coverage_match.iloc[0]
            
            # Get bus flip info
            flip_count = 0
            if self.df_test_case_bus_flips is not None:
                flip_match = self.df_test_case_bus_flips[
                    (self.df_test_case_bus_flips['unit_id'] == unit_id) &
                    (self.df_test_case_bus_flips['station'] == station) &
                    (self.df_test_case_bus_flips['save'] == save)
                ]
                flip_count = flip_match['total_bus_flips'].sum()
            
            summary_data.append({
                'unit_id': unit_id,
                'station': station,
                'save': save,
                'platform': group['platform'].iloc[0] if 'platform' in group.columns else '',
                'test_cases_count': len(group),
                'unique_test_events': group['test_event_id'].nunique(),
                'total_duration': group['duration'].sum(),
                'avg_duration': round(group['duration'].mean(), 2),
                'min_timestamp': group['timestamp_start'].min(),
                'max_timestamp': group['timestamp_end'].max(),
                'coverage_percentage': coverage_info['coverage_percentage'] if coverage_info is not None else 0,
                'messages_in_test_cases': coverage_info['messages_in_test_cases'] if coverage_info is not None else 0,
                'total_bus_flips_in_tests': flip_count
            })
        
        if summary_data:
            self.df_test_case_summary = pd.DataFrame(summary_data)
            self.df_test_case_summary = self.df_test_case_summary.sort_values(
                ['unit_id', 'station', 'save']
            )

    def run_analysis(self):
        """Run the complete analysis on all CSV files"""
        if not self.csv_folder.exists():
            print(f"ERROR: CSV folder '{self.csv_folder}' does not exist!")
            return []
            
        csv_files = list(self.csv_folder.glob("*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in {self.csv_folder}")
            return []
            
        print(f"Found {len(csv_files)} CSV files to process")
        print("-" * 50)
        
        file_results = []
        for i, csv_file in enumerate(csv_files, 1):
            print(f"[{i}/{len(csv_files)}] Processing {csv_file.name}...")
            result = self.process_csv(csv_file)
            if result:
                file_results.append(result)
                self.file_summary.append(result)
        
        # Create DataFrames
        if self.file_summary:
            self.df_file_summary = pd.DataFrame(self.file_summary)
        
        if self.bus_flips:
            self.df_flips = pd.DataFrame(self.bus_flips)
        
        if self.bus_flips_no_changes:
            self.df_flips_no_changes = pd.DataFrame(self.bus_flips_no_changes)
        
        if self.data_changes:
            self.df_data_changes = pd.DataFrame(self.data_changes)
        
        if self.header_issues:
            self.df_header_issues = pd.DataFrame(self.header_issues)
        
        # Create summaries
        print("\nCreating summaries...")
        self.create_summaries()
        self.create_station_save_matrix()
        
        # Analyze data word patterns
        print("\nAnalyzing data word patterns...")
        self.analyze_data_word_patterns()
        
        # Analyze requirements at risk
        print("\nAnalyzing requirements at risk...")
        self.analyze_requirements_at_risk()
        
        print("\nLoading test case data...")
        self.load_tca_files()
        
        if self.df_test_cases is not None and not self.df_test_cases.empty:
            print("\nAnalyzing test case coverage...")
            self.analyze_test_case_coverage()
            
            print("\nAnalyzing test case bus flips...")
            self.analyze_test_case_bus_flips()
            
            print("\nCalculating test case message rates...")
            self.calculate_test_case_message_rates()
            
            print("\nCreating test case summary...")
            self.create_test_case_summary()
        
        # Calculate message rates
        print("\nCalculating message rates...")
        self.calculate_message_rates()

        print("\nAnalyzing failed requirements vs bus flips...")
        self.analyze_failed_requirements_vs_bus_flips()        
        return file_results
        
    def export_to_excel(self):
        """Export only essential data to Excel"""
        excel_path = self.output_folder / "bus_monitor_analysis.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#4CAF50',
                'font_color': 'white',
                'border': 1
            })
            
            # 1. Bus Flips Sheet (Main data - with data changes)
            if self.df_flips is not None and len(self.df_flips) > 0:
                self.df_flips.to_excel(writer, sheet_name='Bus_Flips', index=False)
                print(f"  Exported Bus Flips: {len(self.df_flips)} flips (with data changes)")
            
            # 2. Bus Flips No Changes Sheet
            if self.df_flips_no_changes is not None and len(self.df_flips_no_changes) > 0:
                self.df_flips_no_changes.to_excel(writer, sheet_name='Flips_No_Changes', index=False)
                print(f"  Exported Flips No Changes: {len(self.df_flips_no_changes)} flips")
            
            # 3. Data Word Analysis
            if self.df_data_word_analysis is not None and len(self.df_data_word_analysis) > 0:
                self.df_data_word_analysis.to_excel(writer, sheet_name='Data_Word_Analysis', index=False)
                print(f"  Exported Data Word Analysis: {len(self.df_data_word_analysis)} msg_type/data_word combinations")
            
            # 4. Multi-Word Change Analysis
            if hasattr(self, 'df_multi_word_analysis') and self.df_multi_word_analysis is not None and len(self.df_multi_word_analysis) > 0:
                self.df_multi_word_analysis.to_excel(writer, sheet_name='Multi_Word_Analysis', index=False)
                print(f"  Exported Multi-Word Analysis: {len(self.df_multi_word_analysis)} message types")
            
            # 5. Location Data Issues
            if hasattr(self, 'df_location_data_issues') and self.df_location_data_issues is not None and len(self.df_location_data_issues) > 0:
                self.df_location_data_issues.to_excel(writer, sheet_name='Location_Data_Issues', index=False)
                print(f"  Exported Location Data Issues: {len(self.df_location_data_issues)} locations")
            
            # 6. Data Word Patterns
            if self.df_data_word_patterns is not None and len(self.df_data_word_patterns) > 0:
                # Limit to top 1000 patterns if too many
                if len(self.df_data_word_patterns) > 1000:
                    self.df_data_word_patterns.head(1000).to_excel(writer, sheet_name='Data_Word_Patterns', index=False)
                    print(f"  Exported Data Word Patterns: 1000 patterns (truncated from {len(self.df_data_word_patterns)})")
                else:
                    self.df_data_word_patterns.to_excel(writer, sheet_name='Data_Word_Patterns', index=False)
                    print(f"  Exported Data Word Patterns: {len(self.df_data_word_patterns)} patterns")
            
            # 7. Requirements Affected
            if self.df_requirements_at_risk is not None and len(self.df_requirements_at_risk) > 0:
                self.df_requirements_at_risk.to_excel(writer, sheet_name='Requirements_Affected', index=False)
                print(f"  Exported Requirements Affected: {len(self.df_requirements_at_risk)} requirement instances")
            
            # 8. Requirements Summary
            if self.df_requirements_summary is not None and len(self.df_requirements_summary) > 0:
                self.df_requirements_summary.to_excel(writer, sheet_name='Requirements_Summary', index=False)
                print(f"  Exported Requirements Summary: {len(self.df_requirements_summary)} unique requirements")
            
            # 9. Header Issues
            if self.df_header_issues is not None and len(self.df_header_issues) > 0:
                self.df_header_issues.to_excel(writer, sheet_name='Header_Issues', index=False)
                print(f"  Exported Header Issues: {len(self.df_header_issues)} issues")
            
            # 10. Data Changes
            if self.df_data_changes is not None and len(self.df_data_changes) > 0:
                # Limit to first 10000 rows if too many
                if len(self.df_data_changes) > 10000:
                    self.df_data_changes.iloc[:10000].to_excel(writer, sheet_name='Data_Changes', index=False)
                    print(f"  Exported Data Changes: 10000 rows (truncated from {len(self.df_data_changes)})")
                else:
                    self.df_data_changes.to_excel(writer, sheet_name='Data_Changes', index=False)
                    print(f"  Exported Data Changes: {len(self.df_data_changes)} rows")
            
            # 11. File Summary
            if self.df_file_summary is not None and len(self.df_file_summary) > 0:
                self.df_file_summary.to_excel(writer, sheet_name='File_Summary', index=False)
                print(f"  Exported File Summary: {len(self.df_file_summary)} files")
            
            # 12. Station-Save Matrix
            if self.df_station_save_matrix is not None and len(self.df_station_save_matrix) > 0:
                self.df_station_save_matrix.to_excel(writer, sheet_name='Station_Save_Matrix')
                print(f"  Exported Station-Save Matrix")
            
            # 13. Unit ID Summary
            if self.df_unit_summary is not None and len(self.df_unit_summary) > 0:
                self.df_unit_summary.to_excel(writer, sheet_name='Unit_Summary', index=False)
                print(f"  Exported Unit Summary: {len(self.df_unit_summary)} units")
            
            # 14. Station Summary
            if self.df_station_summary is not None and len(self.df_station_summary) > 0:
                self.df_station_summary.to_excel(writer, sheet_name='Station_Summary', index=False)
                print(f"  Exported Station Summary: {len(self.df_station_summary)} stations")
            
            # 15. Save Summary
            if self.df_save_summary is not None and len(self.df_save_summary) > 0:
                self.df_save_summary.to_excel(writer, sheet_name='Save_Summary', index=False)
                print(f"  Exported Save Summary: {len(self.df_save_summary)} saves")
        
            # 16. Test Case Summary
            if self.df_test_case_summary is not None and len(self.df_test_case_summary) > 0:
                self.df_test_case_summary.to_excel(writer, sheet_name='Test_Case_Summary', index=False)
                print(f"  Exported Test Case Summary: {len(self.df_test_case_summary)} combinations")
            
            # 17. Test Case Coverage
            if self.df_test_case_coverage is not None and len(self.df_test_case_coverage) > 0:
                self.df_test_case_coverage.to_excel(writer, sheet_name='Test_Case_Coverage', index=False)
                print(f"  Exported Test Case Coverage: {len(self.df_test_case_coverage)} files analyzed")
            
            # 18. Test Case Bus Flips
            if self.df_test_case_bus_flips is not None and len(self.df_test_case_bus_flips) > 0:
                self.df_test_case_bus_flips.to_excel(writer, sheet_name='Test_Case_Bus_Flips', index=False)
                print(f"  Exported Test Case Bus Flips: {len(self.df_test_case_bus_flips)} test cases with flips")
            
            # 19. Test Cases Detail
            if self.df_test_cases is not None and len(self.df_test_cases) > 0:
                # Limit if too many
                if len(self.df_test_cases) > 10000:
                    self.df_test_cases.head(10000).to_excel(writer, sheet_name='Test_Cases_Detail', index=False)
                    print(f"  Exported Test Cases Detail: 10000 test cases (truncated from {len(self.df_test_cases)})")
                else:
                    self.df_test_cases.to_excel(writer, sheet_name='Test_Cases_Detail', index=False)
                    print(f"  Exported Test Cases Detail: {len(self.df_test_cases)} test cases")
            
            # 20. Message Rates Overall
            if self.df_message_rates is not None and len(self.df_message_rates) > 0:
                self.df_message_rates.to_excel(writer, sheet_name='Message_Rates', index=False)
                print(f"  Exported Message Rates: {len(self.df_message_rates)} message type/location combinations")
            
            # 21. Message Rates by Unit
            if self.df_message_rates_by_unit is not None and len(self.df_message_rates_by_unit) > 0:
                self.df_message_rates_by_unit.to_excel(writer, sheet_name='Message_Rates_Unit', index=False)
                print(f"  Exported Message Rates by Unit: {len(self.df_message_rates_by_unit)} entries")
            
            # 22. Message Rates by Station
            if self.df_message_rates_by_station is not None and len(self.df_message_rates_by_station) > 0:
                self.df_message_rates_by_station.to_excel(writer, sheet_name='Message_Rates_Station', index=False)
                print(f"  Exported Message Rates by Station: {len(self.df_message_rates_by_station)} entries")
            
            # 23. Message Rates by Save
            if self.df_message_rates_by_save is not None and len(self.df_message_rates_by_save) > 0:
                # Limit if too many
                if len(self.df_message_rates_by_save) > 5000:
                    self.df_message_rates_by_save.head(5000).to_excel(writer, sheet_name='Message_Rates_Save', index=False)
                    print(f"  Exported Message Rates by Save: 5000 entries (truncated from {len(self.df_message_rates_by_save)})")
                else:
                    self.df_message_rates_by_save.to_excel(writer, sheet_name='Message_Rates_Save', index=False)
                    print(f"  Exported Message Rates by Save: {len(self.df_message_rates_by_save)} entries")
            
            # 24. Message Rates by Test Case
            if self.df_message_rates_by_test_case is not None and len(self.df_message_rates_by_test_case) > 0:
                # Limit if too many
                if len(self.df_message_rates_by_test_case) > 5000:
                    self.df_message_rates_by_test_case.head(5000).to_excel(writer, sheet_name='Message_Rates_TestCase', index=False)
                    print(f"  Exported Message Rates by Test Case: 5000 entries (truncated from {len(self.df_message_rates_by_test_case)})")
                else:
                    self.df_message_rates_by_test_case.to_excel(writer, sheet_name='Message_Rates_TestCase', index=False)
                    print(f"  Exported Message Rates by Test Case: {len(self.df_message_rates_by_test_case)} entries")

            # 25. Failed Requirements Analysis
            if hasattr(self, 'df_failed_requirements_analysis') and self.df_failed_requirements_analysis is not None and len(self.df_failed_requirements_analysis) > 0:
                self.df_failed_requirements_analysis.to_excel(writer, sheet_name='Failed_Requirements_Analysis', index=False)
                print(f"  Exported Failed Requirements Analysis: {len(self.df_failed_requirements_analysis)} failed tests analyzed")
            
            # 26. Requirements with Bus Issues
            if hasattr(self, 'df_requirements_with_bus_issues') and self.df_requirements_with_bus_issues is not None and len(self.df_requirements_with_bus_issues) > 0:
                self.df_requirements_with_bus_issues.to_excel(writer, sheet_name='Requirements_With_Bus_Issues', index=False)
                print(f"  Exported Requirements with Bus Issues: {len(self.df_requirements_with_bus_issues)} requirements")

        print(f"\nExcel file saved to: {excel_path.absolute()}")
        return excel_path

    def create_interactive_dashboard(self):
        """Create an enhanced interactive HTML dashboard with comprehensive filters and analytics"""
        import json
        
        dashboard_path = self.output_folder / "dashboard.html"
        
        # Prepare data for JavaScript
        flips_data = []
        if self.df_flips is not None and not self.df_flips.empty:
            df_temp = self.df_flips.copy()
            for col in ['timestamp_busA', 'timestamp_busB']:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col].astype(str)
            flips_data = df_temp.to_dict('records')
        
        # Prepare test case data
        test_case_data = []
        if self.df_test_cases is not None and not self.df_test_cases.empty:
            df_tc_temp = self.df_test_cases.copy()
            for col in ['timestamp_start', 'timestamp_end']:
                if col in df_tc_temp.columns:
                    df_tc_temp[col] = df_tc_temp[col].astype(str)
            test_case_data = df_tc_temp.to_dict('records')
        
        # Prepare test case bus flips data
        test_case_flip_data = []
        if self.df_test_case_bus_flips is not None and not self.df_test_case_bus_flips.empty:
            test_case_flip_data = self.df_test_case_bus_flips.to_dict('records')
        
        # Prepare message rates data
        message_rates_data = []
        if self.df_message_rates is not None and not self.df_message_rates.empty:
            message_rates_data = self.df_message_rates.to_dict('records')
        
        # Prepare failed requirements data
        failed_requirements_data = []
        if hasattr(self, 'df_failed_requirements_analysis') and self.df_failed_requirements_analysis is not None:
            failed_requirements_data = self.df_failed_requirements_analysis.to_dict('records')
        
        # Get unique values for filters
        unit_ids = sorted(self.df_flips['unit_id'].unique().tolist()) if self.df_flips is not None else []
        stations = sorted(self.df_flips['station'].unique().tolist()) if self.df_flips is not None else []
        saves = sorted(self.df_flips['save'].unique().tolist()) if self.df_flips is not None else []
        msg_types = sorted(self.df_flips['msg_type'].dropna().unique().tolist()) if self.df_flips is not None else []
        
        # Get test case IDs for filter
        test_case_ids = []
        if self.df_test_cases is not None and not self.df_test_cases.empty:
            test_case_ids = sorted(self.df_test_cases['test_case_id'].unique().tolist())
        
        # Calculate summary stats
        total_flips = len(self.df_flips) if self.df_flips is not None else 0
        total_flips_no_changes = len(self.df_flips_no_changes) if self.df_flips_no_changes is not None else 0
        total_units = len(unit_ids)
        total_stations = len(stations)
        total_saves = len(saves)
        
        # Calculate flip percentage
        flip_percentage = 0
        if self.total_messages_processed > 0:
            flip_percentage = (total_flips / self.total_messages_processed) * 100
        
        # Calculate unique data word issues (avoiding overlap)
        unique_data_issues = 0
        if self.df_data_word_analysis is not None and not self.df_data_word_analysis.empty:
            # Count unique flip events, not total data word changes
            if self.df_flips is not None and 'num_data_changes' in self.df_flips.columns:
                unique_data_issues = len(self.df_flips[self.df_flips['num_data_changes'] > 0])
        
        # Get data word analysis for dashboard
        data_word_data = []
        if self.df_data_word_analysis is not None and not self.df_data_word_analysis.empty:
            data_word_data = self.df_data_word_analysis.head(50).to_dict('records')
        
        # Convert to JSON for JavaScript
        flips_data_json = json.dumps(flips_data)
        test_case_data_json = json.dumps(test_case_data)
        test_case_flip_data_json = json.dumps(test_case_flip_data)
        message_rates_data_json = json.dumps(message_rates_data)
        failed_requirements_data_json = json.dumps(failed_requirements_data)
        data_word_data_json = json.dumps(data_word_data)
        unit_ids_json = json.dumps(unit_ids)
        stations_json = json.dumps(stations)
        saves_json = json.dumps(saves)
        msg_types_json = json.dumps(msg_types)
        test_case_ids_json = json.dumps(test_case_ids)
        
        html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Bus Monitor Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }}
            .container {{
                max-width: 1800px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 4px solid #3498db;
                padding-bottom: 15px;
                font-size: 32px;
            }}
            .filters {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px;
                margin: 30px 0;
                padding: 25px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px;
                box-shadow: 0 3px 15px rgba(0,0,0,0.2);
            }}
            .filter-group {{
                display: flex;
                flex-direction: column;
            }}
            .filter-group label {{
                font-weight: 600;
                margin-bottom: 6px;
                color: white;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .filter-group select {{
                padding: 10px;
                border: 2px solid rgba(255,255,255,0.3);
                border-radius: 6px;
                background: rgba(255,255,255,0.95);
                font-size: 14px;
                transition: all 0.3s ease;
            }}
            .filter-group select:hover {{
                border-color: rgba(255,255,255,0.6);
                transform: translateY(-1px);
            }}
            .stats-row {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                gap: 15px;
                margin: 30px 0;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                transition: transform 0.3s ease;
                cursor: pointer;
            }}
            .stat-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 5px 20px rgba(0,0,0,0.3);
            }}
            .stat-card.warning {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            }}
            .stat-card.info {{
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            }}
            .stat-card.success {{
                background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
            }}
            .stat-value {{
                font-size: 28px;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .stat-label {{
                opacity: 0.95;
                font-size: 13px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .chart-container {{
                margin: 30px 0;
                padding: 25px;
                background: white;
                border-radius: 12px;
                box-shadow: 0 3px 15px rgba(0,0,0,0.1);
                border: 1px solid #e1e8ed;
            }}
            .chart-title {{
                font-size: 20px;
                font-weight: 600;
                margin-bottom: 20px;
                color: #2c3e50;
                border-left: 4px solid #3498db;
                padding-left: 12px;
            }}
            #filteredCount {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px 24px;
                border-radius: 8px;
                display: inline-block;
                margin: 20px 0;
                font-weight: 600;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            }}
            .reset-btn {{
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                margin-left: 10px;
                transition: all 0.3s ease;
            }}
            .reset-btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            }}
            .info-box {{
                margin-bottom: 15px;
                padding: 15px;
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                border-left: 4px solid #2196F3;
                border-radius: 6px;
            }}
            .data-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            .data-table th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 12px;
                text-align: left;
                cursor: pointer;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .data-table td {{
                padding: 10px;
                border-bottom: 1px solid #e1e8ed;
            }}
            .data-table tr:hover {{
                background: #f8f9fa;
            }}
            .tab-container {{
                margin-top: 30px;
            }}
            .tab-buttons {{
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                border-bottom: 2px solid #e1e8ed;
            }}
            .tab-button {{
                padding: 12px 24px;
                background: none;
                border: none;
                cursor: pointer;
                font-size: 16px;
                font-weight: 600;
                color: #7f8c8d;
                transition: all 0.3s ease;
                border-bottom: 3px solid transparent;
                margin-bottom: -2px;
            }}
            .tab-button.active {{
                color: #3498db;
                border-bottom-color: #3498db;
            }}
            .tab-content {{
                display: none;
            }}
            .tab-content.active {{
                display: block;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-card {{
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }}
            .metric-title {{
                font-size: 14px;
                color: #7f8c8d;
                margin-bottom: 5px;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚌 Enhanced Bus Monitor Dashboard</h1>
            
            <!-- Dynamic Stats Row -->
            <div id="statsRow" class="stats-row">
                <!-- Will be populated dynamically -->
            </div>
            
            <!-- Enhanced Filters -->
            <div class="filters">
                <div class="filter-group">
                    <label for="globalUnitFilter">Unit ID:</label>
                    <select id="globalUnitFilter" onchange="updateAllCharts()">
                        <option value="">All Units</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="globalStationFilter">Station:</label>
                    <select id="globalStationFilter" onchange="updateAllCharts()">
                        <option value="">All Stations</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="globalSaveFilter">Save:</label>
                    <select id="globalSaveFilter" onchange="updateAllCharts()">
                        <option value="">All Saves</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="globalMsgTypeFilter">Message Type:</label>
                    <select id="globalMsgTypeFilter" onchange="updateAllCharts()">
                        <option value="">All Message Types</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label for="globalTestCaseFilter">Test Case:</label>
                    <select id="globalTestCaseFilter" onchange="updateAllCharts()">
                        <option value="">All Test Cases</option>
                    </select>
                </div>
            </div>
            
            <div>
                <span id="filteredCount">Loading...</span>
                <button class="reset-btn" onclick="resetAllFilters()">🔄 Reset All Filters</button>
            </div>
            
            <!-- Tab Navigation -->
            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="switchTab('flips')">Bus Flips Analysis</button>
                    <button class="tab-button" onclick="switchTab('rates')">Message Rates</button>
                    <button class="tab-button" onclick="switchTab('testcases')">Test Cases</button>
                    <button class="tab-button" onclick="switchTab('requirements')">Requirements</button>
                    <button class="tab-button" onclick="switchTab('datawords')">Data Words</button>
                </div>
                
                <!-- Bus Flips Tab -->
                <div id="flips-tab" class="tab-content active">
                    <div class="chart-container">
                        <div class="chart-title">Bus Flips by Unit ID</div>
                        <div id="unitChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Bus Flips by Station</div>
                        <div id="stationChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Bus Flips by Save</div>
                        <div id="saveChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Bus Flips by Message Type</div>
                        <div id="msgTypeChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">R vs T Message Distribution</div>
                        <div id="rtChart"></div>
                    </div>
                </div>
                
                <!-- Message Rates Tab -->
                <div id="rates-tab" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">Message Rate Overview</div>
                        <div class="info-box">
                            <strong>Message Rate Analysis:</strong> Shows message transmission rates for all message types, including those with and without bus flips.
                            Higher rates may correlate with increased flip probability.
                        </div>
                        <div id="messageRateOverview"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Message Rate vs Bus Flip Correlation</div>
                        <div id="rateFlipCorrelation"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Message Rate Distribution by Type</div>
                        <div id="rateDistribution"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Message Rate Statistics</div>
                        <div class="metric-grid" id="rateMetrics"></div>
                    </div>
                </div>
                
                <!-- Test Cases Tab -->
                <div id="testcases-tab" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">Test Case Coverage</div>
                        <div id="testCaseCoverage"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Bus Flips by Test Case</div>
                        <div id="testCaseFlips"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Test Case Summary Table</div>
                        <div style="overflow-x: auto;">
                            <table id="testCaseTable" class="data-table">
                                <thead>
                                    <tr>
                                        <th onclick="sortTable('testCaseTable', 0)">Test Case ID</th>
                                        <th onclick="sortTable('testCaseTable', 1)">Unit ID</th>
                                        <th onclick="sortTable('testCaseTable', 2)">Station</th>
                                        <th onclick="sortTable('testCaseTable', 3)">Save</th>
                                        <th onclick="sortTable('testCaseTable', 4)">Bus Flips</th>
                                        <th onclick="sortTable('testCaseTable', 5)">Msg Types Affected</th>
                                        <th onclick="sortTable('testCaseTable', 6)">Duration</th>
                                    </tr>
                                </thead>
                                <tbody id="testCaseTableBody"></tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <!-- Requirements Tab -->
                <div id="requirements-tab" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">Failed Requirements with Bus Flips</div>
                        <div id="failedRequirements"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Requirements Impact Analysis</div>
                        <div id="requirementsImpact"></div>
                    </div>
                </div>
                
                <!-- Data Words Tab -->
                <div id="datawords-tab" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">Data Word Error Analysis</div>
                        <div class="info-box">
                            <strong>Note:</strong> Each message with data word changes counts as 1 event, regardless of how many data words changed.
                        </div>
                        <div id="dataWordChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Multi-Word Change Analysis</div>
                        <div id="multiWordChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Data Word Patterns</div>
                        <div style="overflow-x: auto;">
                            <table id="dataWordTable" class="data-table">
                                <thead>
                                    <tr>
                                        <th>Message Type</th>
                                        <th>Data Word</th>
                                        <th>Total Issues</th>
                                        <th>Single Word Changes</th>
                                        <th>Multi Word Changes</th>
                                        <th>Avg Speed (ms)</th>
                                        <th>Top Patterns</th>
                                    </tr>
                                </thead>
                                <tbody id="dataWordTableBody"></tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Data from Python
            const allFlipsData = {flips_data_json};
            const testCaseData = {test_case_data_json};
            const testCaseFlipData = {test_case_flip_data_json};
            const messageRatesData = {message_rates_data_json};
            const failedRequirementsData = {failed_requirements_data_json};
            const dataWordData = {data_word_data_json};
            
            // Filtered data
            let filteredFlipsData = [...allFlipsData];
            let filteredTestCaseData = [...testCaseData];
            let filteredMessageRatesData = [...messageRatesData];
            let filteredDataWordData = [...dataWordData];
            
            // Unique values for filters
            const uniqueUnits = {unit_ids_json};
            const uniqueStations = {stations_json};
            const uniqueSaves = {saves_json};
            const uniqueMsgTypes = {msg_types_json};
            const uniqueTestCases = {test_case_ids_json};
            
            // Initialize filters
            function initializeFilters() {{
                const unitFilter = document.getElementById('globalUnitFilter');
                uniqueUnits.forEach(unit => {{
                    const option = document.createElement('option');
                    option.value = unit;
                    option.textContent = unit;
                    unitFilter.appendChild(option);
                }});
                
                const stationFilter = document.getElementById('globalStationFilter');
                uniqueStations.forEach(station => {{
                    const option = document.createElement('option');
                    option.value = station;
                    option.textContent = station;
                    stationFilter.appendChild(option);
                }});
                
                const saveFilter = document.getElementById('globalSaveFilter');
                uniqueSaves.forEach(save => {{
                    const option = document.createElement('option');
                    option.value = save;
                    option.textContent = save;
                    saveFilter.appendChild(option);
                }});
                
                const msgTypeFilter = document.getElementById('globalMsgTypeFilter');
                uniqueMsgTypes.forEach(msgType => {{
                    const option = document.createElement('option');
                    option.value = msgType;
                    option.textContent = msgType;
                    msgTypeFilter.appendChild(option);
                }});
                
                const testCaseFilter = document.getElementById('globalTestCaseFilter');
                uniqueTestCases.forEach(testCase => {{
                    const option = document.createElement('option');
                    option.value = testCase;
                    option.textContent = testCase;
                    testCaseFilter.appendChild(option);
                }});
            }}
            
            function updateAllCharts() {{
                const unitFilter = document.getElementById('globalUnitFilter').value;
                const stationFilter = document.getElementById('globalStationFilter').value;
                const saveFilter = document.getElementById('globalSaveFilter').value;
                const msgTypeFilter = document.getElementById('globalMsgTypeFilter').value;
                const testCaseFilter = document.getElementById('globalTestCaseFilter').value;
                
                // Filter flips data
                filteredFlipsData = allFlipsData.filter(row => {{
                    return (!unitFilter || row.unit_id === unitFilter) &&
                        (!stationFilter || row.station === stationFilter) &&
                        (!saveFilter || row.save === saveFilter) &&
                        (!msgTypeFilter || row.msg_type === msgTypeFilter);
                }});
                
                // Filter by test case if selected
                if (testCaseFilter && testCaseData.length > 0) {{
                    const testCase = testCaseData.find(tc => tc.test_case_id === testCaseFilter);
                    if (testCase) {{
                        filteredFlipsData = filteredFlipsData.filter(flip => {{
                            const flipTime = parseFloat(flip.timestamp_busA);
                            const testStart = parseFloat(testCase.timestamp_start);
                            const testEnd = parseFloat(testCase.timestamp_end);
                            return flipTime >= testStart && flipTime <= testEnd;
                        }});
                    }}
                }}
                
                // Filter test case data
                filteredTestCaseData = testCaseData.filter(row => {{
                    return (!unitFilter || row.unit_id === unitFilter) &&
                        (!stationFilter || row.station === stationFilter) &&
                        (!saveFilter || row.save === saveFilter) &&
                        (!testCaseFilter || row.test_case_id === testCaseFilter);
                }});
                
                // Filter message rates data
                filteredMessageRatesData = messageRatesData.filter(row => {{
                    return (!unitFilter || row.unit_id === unitFilter) &&
                        (!stationFilter || row.station === stationFilter) &&
                        (!saveFilter || row.save === saveFilter) &&
                        (!msgTypeFilter || row.msg_type === msgTypeFilter);
                }});
                
                // Filter data word data
                filteredDataWordData = dataWordData.filter(row => {{
                    return (!msgTypeFilter || row.msg_type === msgTypeFilter);
                }});
                
                // Update stats and charts
                updateStats();
                updateFilteredCount();
                drawAllCharts();
            }}
            
            function updateStats() {{
                const statsRow = document.getElementById('statsRow');
                
                // Calculate filtered stats
                const totalFlips = filteredFlipsData.length;
                const uniqueUnits = [...new Set(filteredFlipsData.map(f => f.unit_id))].length;
                const uniqueStations = [...new Set(filteredFlipsData.map(f => f.station))].length;
                const uniqueSaves = [...new Set(filteredFlipsData.map(f => f.save))].length;
                const uniqueMsgTypes = [...new Set(filteredFlipsData.filter(f => f.msg_type).map(f => f.msg_type))].length;
                
                // Count unique data issues (messages with data changes, not total data words)
                const uniqueDataIssues = filteredFlipsData.filter(f => f.has_data_changes).length;
                
                // Test case stats
                const testCasesWithFlips = testCaseFlipData.filter(tc => {{
                    const unitFilter = document.getElementById('globalUnitFilter').value;
                    const stationFilter = document.getElementById('globalStationFilter').value;
                    const saveFilter = document.getElementById('globalSaveFilter').value;
                    return (!unitFilter || tc.unit_id === unitFilter) &&
                        (!stationFilter || tc.station === stationFilter) &&
                        (!saveFilter || tc.save === saveFilter);
                }}).length;
                
                statsRow.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-value">${{totalFlips.toLocaleString()}}</div>
                        <div class="stat-label">Bus Flips</div>
                    </div>
                    <div class="stat-card warning">
                        <div class="stat-value">${{uniqueDataIssues.toLocaleString()}}</div>
                        <div class="stat-label">Messages w/ Data Changes</div>
                    </div>
                    <div class="stat-card info">
                        <div class="stat-value">${{uniqueUnits}}</div>
                        <div class="stat-label">Units</div>
                    </div>
                    <div class="stat-card info">
                        <div class="stat-value">${{uniqueStations}}</div>
                        <div class="stat-label">Stations</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-value">${{uniqueSaves}}</div>
                        <div class="stat-label">Saves</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${{uniqueMsgTypes}}</div>
                        <div class="stat-label">Message Types</div>
                    </div>
                    <div class="stat-card warning">
                        <div class="stat-value">${{testCasesWithFlips}}</div>
                        <div class="stat-label">Test Cases w/ Flips</div>
                    </div>
                `;
            }}
            
            function updateFilteredCount() {{
                const filterStatus = [];
                const unitFilter = document.getElementById('globalUnitFilter').value;
                const stationFilter = document.getElementById('globalStationFilter').value;
                const saveFilter = document.getElementById('globalSaveFilter').value;
                const msgTypeFilter = document.getElementById('globalMsgTypeFilter').value;
                const testCaseFilter = document.getElementById('globalTestCaseFilter').value;
                
                if (unitFilter) filterStatus.push(`Unit: ${{unitFilter}}`);
                if (stationFilter) filterStatus.push(`Station: ${{stationFilter}}`);
                if (saveFilter) filterStatus.push(`Save: ${{saveFilter}}`);
                if (msgTypeFilter) filterStatus.push(`Msg Type: ${{msgTypeFilter}}`);
                if (testCaseFilter) filterStatus.push(`Test Case: ${{testCaseFilter}}`);
                
                const filterText = filterStatus.length > 0 
                    ? `Filtered by: ${{filterStatus.join(' | ')}} - Showing ${{filteredFlipsData.length}} of ${{allFlipsData.length}} flips`
                    : `Showing all ${{allFlipsData.length}} flips`;
                
                document.getElementById('filteredCount').textContent = filterText;
            }}
            
            function resetAllFilters() {{
                document.getElementById('globalUnitFilter').value = '';
                document.getElementById('globalStationFilter').value = '';
                document.getElementById('globalSaveFilter').value = '';
                document.getElementById('globalMsgTypeFilter').value = '';
                document.getElementById('globalTestCaseFilter').value = '';
                updateAllCharts();
            }}
            
            function switchTab(tabName) {{
                // Update button states
                document.querySelectorAll('.tab-button').forEach(btn => {{
                    btn.classList.remove('active');
                }});
                event.target.classList.add('active');
                
                // Update content visibility
                document.querySelectorAll('.tab-content').forEach(content => {{
                    content.classList.remove('active');
                }});
                document.getElementById(`${{tabName}}-tab`).classList.add('active');
                
                // Redraw charts for the active tab
                if (tabName === 'rates') {{
                    drawMessageRateCharts();
                }} else if (tabName === 'testcases') {{
                    drawTestCaseCharts();
                }} else if (tabName === 'requirements') {{
                    drawRequirementsCharts();
                }} else if (tabName === 'datawords') {{
                    drawDataWordCharts();
                }}
            }}
            
            function drawAllCharts() {{
                drawBusFlipCharts();
                drawMessageRateCharts();
                drawTestCaseCharts();
                drawRequirementsCharts();
                drawDataWordCharts();
            }}
            
            function drawBusFlipCharts() {{
                // Unit ID Chart
                const unitCounts = {{}};
                filteredFlipsData.forEach(d => {{
                    unitCounts[d.unit_id] = (unitCounts[d.unit_id] || 0) + 1;
                }});
                
                Plotly.newPlot('unitChart', [{{
                    x: Object.keys(unitCounts),
                    y: Object.values(unitCounts),
                    type: 'bar',
                    marker: {{ color: '#667eea' }}
                }}], {{
                    margin: {{ t: 10, b: 40, l: 60, r: 20 }},
                    xaxis: {{ title: 'Unit ID' }},
                    yaxis: {{ title: 'Number of Flips' }}
                }});
                
                // Station Chart
                const stationCounts = {{}};
                filteredFlipsData.forEach(d => {{
                    stationCounts[d.station] = (stationCounts[d.station] || 0) + 1;
                }});
                
                Plotly.newPlot('stationChart', [{{
                    x: Object.keys(stationCounts),
                    y: Object.values(stationCounts),
                    type: 'bar',
                    marker: {{ color: '#764ba2' }}
                }}], {{
                    margin: {{ t: 10, b: 40, l: 60, r: 20 }},
                    xaxis: {{ title: 'Station' }},
                    yaxis: {{ title: 'Number of Flips' }}
                }});
                
                // Save Chart
                const saveCounts = {{}};
                filteredFlipsData.forEach(d => {{
                    saveCounts[d.save] = (saveCounts[d.save] || 0) + 1;
                }});
                
                Plotly.newPlot('saveChart', [{{
                    x: Object.keys(saveCounts),
                    y: Object.values(saveCounts),
                    type: 'bar',
                    marker: {{ color: '#4CAF50' }}
                }}], {{
                    margin: {{ t: 10, b: 40, l: 60, r: 20 }},
                    xaxis: {{ title: 'Save' }},
                    yaxis: {{ title: 'Number of Flips' }}
                }});
                
                // Message Type Chart
                const msgTypeCounts = {{}};
                filteredFlipsData.forEach(d => {{
                    if (d.msg_type) {{
                        msgTypeCounts[d.msg_type] = (msgTypeCounts[d.msg_type] || 0) + 1;
                    }}
                }});
                
                const sortedMsgTypes = Object.entries(msgTypeCounts)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 20);
                
                Plotly.newPlot('msgTypeChart', [{{
                    x: sortedMsgTypes.map(x => x[0]),
                    y: sortedMsgTypes.map(x => x[1]),
                    type: 'bar',
                    marker: {{ color: '#ff9800' }}
                }}], {{
                    margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                    xaxis: {{ title: 'Message Type', tickangle: -45 }},
                    yaxis: {{ title: 'Number of Flips' }}
                }});
                
                // R vs T Chart
                const rCount = filteredFlipsData.filter(d => {{
                    if (!d.msg_type) return false;
                    const msgStr = d.msg_type.toString().trim();
                    return /^\\d+R/.test(msgStr);
                }}).length;
                
                const tCount = filteredFlipsData.filter(d => {{
                    if (!d.msg_type) return false;
                    const msgStr = d.msg_type.toString().trim();
                    return /^\\d+T/.test(msgStr);
                }}).length;
                
                Plotly.newPlot('rtChart', [{{
                    x: ['R Messages', 'T Messages'],
                    y: [rCount, tCount],
                    type: 'bar',
                    marker: {{ color: ['#4CAF50', '#2196F3'] }},
                    text: [rCount, tCount],
                    textposition: 'auto'
                }}], {{
                    margin: {{ t: 10, b: 40, l: 60, r: 20 }},
                    yaxis: {{ title: 'Count' }}
                }});
            }}
            
            function drawMessageRateCharts() {{
                if (filteredMessageRatesData.length === 0) {{
                    document.getElementById('messageRateOverview').innerHTML = '<p>No message rate data available</p>';
                    return;
                }}
                
                // Overview chart - rates by message type
                const ratesByType = {{}};
                filteredMessageRatesData.forEach(d => {{
                    if (!ratesByType[d.msg_type]) {{
                        ratesByType[d.msg_type] = {{
                            rates: [],
                            flip_count: 0
                        }};
                    }}
                    ratesByType[d.msg_type].rates.push(d.mean_rate);
                    ratesByType[d.msg_type].flip_count += d.bus_flip_count;
                }});
                
                const msgTypes = Object.keys(ratesByType);
                const avgRates = msgTypes.map(type => {{
                    const rates = ratesByType[type].rates;
                    return rates.reduce((a, b) => a + b, 0) / rates.length;
                }});
                const flipCounts = msgTypes.map(type => ratesByType[type].flip_count);
                
                Plotly.newPlot('messageRateOverview', [{{
                    x: msgTypes,
                    y: avgRates,
                    name: 'Avg Message Rate (msg/s)',
                    type: 'bar',
                    marker: {{ color: '#3498db' }}
                }}, {{
                    x: msgTypes,
                    y: flipCounts,
                    name: 'Bus Flip Count',
                    type: 'scatter',
                    mode: 'lines+markers',
                    yaxis: 'y2',
                    line: {{ color: '#e74c3c', width: 3 }},
                    marker: {{ size: 8 }}
                }}], {{
                    margin: {{ t: 10, b: 100, l: 60, r: 60 }},
                    xaxis: {{ title: 'Message Type', tickangle: -45 }},
                    yaxis: {{ title: 'Message Rate (msg/s)' }},
                    yaxis2: {{
                        title: 'Bus Flip Count',
                        overlaying: 'y',
                        side: 'right'
                    }}
                }});
                
                // Correlation scatter plot
                const scatterData = filteredMessageRatesData.map(d => ({{
                    x: d.mean_rate,
                    y: d.bus_flip_count,
                    text: `${{d.msg_type}} (${{d.unit_id}}-${{d.station}}-${{d.save}})`
                }}));
                
                Plotly.newPlot('rateFlipCorrelation', [{{
                    x: scatterData.map(d => d.x),
                    y: scatterData.map(d => d.y),
                    text: scatterData.map(d => d.text),
                    mode: 'markers',
                    type: 'scatter',
                    marker: {{
                        size: 10,
                        color: scatterData.map(d => d.y),
                        colorscale: 'Viridis',
                        showscale: true,
                        colorbar: {{ title: 'Flip Count' }}
                    }}
                }}], {{
                    margin: {{ t: 10, b: 60, l: 60, r: 60 }},
                    xaxis: {{ title: 'Message Rate (msg/s)' }},
                    yaxis: {{ title: 'Bus Flip Count' }}
                }});
                
                // Rate distribution histogram
                const allRates = filteredMessageRatesData.map(d => d.mean_rate);
                
                Plotly.newPlot('rateDistribution', [{{
                    x: allRates,
                    type: 'histogram',
                    nbinsx: 30,
                    marker: {{ color: '#9b59b6' }}
                }}], {{
                    margin: {{ t: 10, b: 60, l: 60, r: 20 }},
                    xaxis: {{ title: 'Message Rate (msg/s)' }},
                    yaxis: {{ title: 'Frequency' }}
                }});
                
                // Rate metrics
                const metricsHtml = `
                    <div class="metric-card">
                        <div class="metric-title">Total Message Types</div>
                        <div class="metric-value">${{[...new Set(filteredMessageRatesData.map(d => d.msg_type))].length}}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg Rate (All Types)</div>
                        <div class="metric-value">${{(allRates.reduce((a, b) => a + b, 0) / allRates.length).toFixed(2)}} msg/s</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Max Rate Observed</div>
                        <div class="metric-value">${{Math.max(...filteredMessageRatesData.map(d => d.max_rate)).toFixed(2)}} msg/s</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Min Rate Observed</div>
                        <div class="metric-value">${{Math.min(...filteredMessageRatesData.map(d => d.min_rate)).toFixed(2)}} msg/s</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Types with Bus Flips</div>
                        <div class="metric-value">${{filteredMessageRatesData.filter(d => d.bus_flip_count > 0).length}}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Avg Flip Rate (when flipped)</div>
                        <div class="metric-value">${{
                            (() => {{
                                const withFlips = filteredMessageRatesData.filter(d => d.bus_flip_avg_rate > 0);
                                if (withFlips.length === 0) return 'N/A';
                                return (withFlips.map(d => d.bus_flip_avg_rate).reduce((a, b) => a + b, 0) / withFlips.length).toFixed(2);
                            }})()
                        }} flips/s</div>
                    </div>
                `;
                document.getElementById('rateMetrics').innerHTML = metricsHtml;
            }}
            
            function drawTestCaseCharts() {{
                if (testCaseFlipData.length === 0) {{
                    document.getElementById('testCaseCoverage').innerHTML = '<p>No test case data available</p>';
                    return;
                }}
                
                // Filter test case flip data based on current filters
                const filteredTestFlips = testCaseFlipData.filter(tc => {{
                    const unitFilter = document.getElementById('globalUnitFilter').value;
                    const stationFilter = document.getElementById('globalStationFilter').value;
                    const saveFilter = document.getElementById('globalSaveFilter').value;
                    return (!unitFilter || tc.unit_id === unitFilter) &&
                        (!stationFilter || tc.station === stationFilter) &&
                        (!saveFilter || tc.save === saveFilter);
                }});
                
                // Coverage chart
                const coverageData = filteredTestFlips.slice(0, 20).sort((a, b) => b.total_bus_flips - a.total_bus_flips);
                
                Plotly.newPlot('testCaseCoverage', [{{
                    x: coverageData.map(d => d.test_case_id),
                    y: coverageData.map(d => d.total_bus_flips),
                    type: 'bar',
                    marker: {{ color: '#3498db' }},
                    text: coverageData.map(d => `${{d.unit_id}}-${{d.station}}-${{d.save}}`),
                    hovertemplate: '%{{text}}<br>Flips: %{{y}}<extra></extra>'
                }}], {{
                    margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                    xaxis: {{ title: 'Test Case ID', tickangle: -45 }},
                    yaxis: {{ title: 'Bus Flips in Test Case' }}
                }});
                
                // Test case flips by message type
                const tcByMsgType = {{}};
                filteredTestFlips.forEach(tc => {{
                    if (tc.msg_types_list) {{
                        tc.msg_types_list.split(', ').forEach(msgType => {{
                            tcByMsgType[msgType] = (tcByMsgType[msgType] || 0) + tc.total_bus_flips;
                        }});
                    }}
                }});
                
                const sortedTcMsgTypes = Object.entries(tcByMsgType)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 15);
                
                Plotly.newPlot('testCaseFlips', [{{
                    x: sortedTcMsgTypes.map(x => x[0]),
                    y: sortedTcMsgTypes.map(x => x[1]),
                    type: 'bar',
                    marker: {{ color: '#e74c3c' }}
                }}], {{
                    margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                    xaxis: {{ title: 'Message Type', tickangle: -45 }},
                    yaxis: {{ title: 'Total Flips in Test Cases' }}
                }});
                
                // Populate test case table
                const tableBody = document.getElementById('testCaseTableBody');
                tableBody.innerHTML = '';
                
                filteredTestFlips.slice(0, 100).forEach(tc => {{
                    const row = tableBody.insertRow();
                    row.insertCell(0).textContent = tc.test_case_id || '';
                    row.insertCell(1).textContent = tc.unit_id || '';
                    row.insertCell(2).textContent = tc.station || '';
                    row.insertCell(3).textContent = tc.save || '';
                    row.insertCell(4).textContent = tc.total_bus_flips || 0;
                    row.insertCell(5).textContent = tc.unique_msg_types_affected || 0;
                    row.insertCell(6).textContent = `${{(tc.test_case_duration || 0).toFixed(2)}}s`;
                }});
            }}
            
            function drawRequirementsCharts() {{
                if (failedRequirementsData.length === 0) {{
                    document.getElementById('failedRequirements').innerHTML = '<p>No failed requirements data available</p>';
                    return;
                }}
                
                // Failed requirements with bus flips
                const reqWithFlips = failedRequirementsData.filter(r => r.has_bus_flips);
                const reqByName = {{}};
                
                reqWithFlips.forEach(r => {{
                    if (!reqByName[r.requirement_name]) {{
                        reqByName[r.requirement_name] = {{
                            flip_count: 0,
                            failed_types: new Set()
                        }};
                    }}
                    reqByName[r.requirement_name].flip_count += r.flip_count;
                    reqByName[r.requirement_name].failed_types.add(r.msg_type);
                }});
                
                const sortedReqs = Object.entries(reqByName)
                    .map(([name, data]) => ({{
                        name: name,
                        flip_count: data.flip_count,
                        failed_types: data.failed_types.size
                    }}))
                    .sort((a, b) => b.flip_count - a.flip_count)
                    .slice(0, 15);
                
                Plotly.newPlot('failedRequirements', [{{
                    x: sortedReqs.map(r => r.name),
                    y: sortedReqs.map(r => r.flip_count),
                    type: 'bar',
                    marker: {{ color: '#e74c3c' }},
                    text: sortedReqs.map(r => `${{r.failed_types}} failed types`),
                    hovertemplate: '%{{x}}<br>Flips: %{{y}}<br>%{{text}}<extra></extra>'
                }}], {{
                    margin: {{ t: 10, b: 120, l: 60, r: 20 }},
                    xaxis: {{ title: 'Requirement Name', tickangle: -45 }},
                    yaxis: {{ title: 'Bus Flips in Failed Tests' }}
                }});
                
                // Impact analysis
                const impactData = [
                    {{ label: 'Failed Tests', value: failedRequirementsData.length }},
                    {{ label: 'With Bus Flips', value: reqWithFlips.length }},
                    {{ label: 'Without Flips', value: failedRequirementsData.length - reqWithFlips.length }}
                ];
                
                Plotly.newPlot('requirementsImpact', [{{
                    labels: impactData.map(d => d.label),
                    values: impactData.map(d => d.value),
                    type: 'pie',
                    marker: {{
                        colors: ['#e74c3c', '#f39c12', '#27ae60']
                    }}
                }}], {{
                    margin: {{ t: 20, b: 20, l: 20, r: 20 }},
                    height: 400
                }});
            }}
            
            function drawDataWordCharts() {{
                if (filteredDataWordData.length === 0) {{
                    document.getElementById('dataWordChart').innerHTML = '<p>No data word analysis available</p>';
                    return;
                }}
                
                // Data word issues chart
                const sortedData = filteredDataWordData.slice(0, 20);
                
                Plotly.newPlot('dataWordChart', [{{
                    x: sortedData.map(d => `${{d.msg_type}}-${{d.data_word}}`),
                    y: sortedData.map(d => d.total_issues),
                    type: 'bar',
                    marker: {{
                        color: sortedData.map(d => d.multi_word_changes || 0),
                        colorscale: [
                            [0, '#4CAF50'],
                            [0.5, '#ff9800'],
                            [1, '#f44336']
                        ],
                        showscale: true,
                        colorbar: {{
                            title: 'Multi-Word<br>Changes',
                            thickness: 15
                        }}
                    }}
                }}], {{
                    margin: {{ t: 10, b: 120, l: 60, r: 80 }},
                    xaxis: {{ title: 'Message Type - Data Word', tickangle: -45 }},
                    yaxis: {{ title: 'Number of Issues' }}
                }});
                
                // Multi-word change analysis
                const multiWordStats = {{}};
                filteredDataWordData.forEach(d => {{
                    if (!multiWordStats[d.msg_type]) {{
                        multiWordStats[d.msg_type] = {{
                            single: 0,
                            multi: 0
                        }};
                    }}
                    multiWordStats[d.msg_type].single += d.single_word_changes || 0;
                    multiWordStats[d.msg_type].multi += d.multi_word_changes || 0;
                }});
                
                const msgTypesMulti = Object.keys(multiWordStats);
                const singleCounts = msgTypesMulti.map(t => multiWordStats[t].single);
                const multiCounts = msgTypesMulti.map(t => multiWordStats[t].multi);
                
                Plotly.newPlot('multiWordChart', [{{
                    x: msgTypesMulti,
                    y: singleCounts,
                    name: 'Single Word Changes',
                    type: 'bar',
                    marker: {{ color: '#4CAF50' }}
                }}, {{
                    x: msgTypesMulti,
                    y: multiCounts,
                    name: 'Multi Word Changes',
                    type: 'bar',
                    marker: {{ color: '#e74c3c' }}
                }}], {{
                    barmode: 'stack',
                    margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                    xaxis: {{ title: 'Message Type', tickangle: -45 }},
                    yaxis: {{ title: 'Number of Changes' }}
                }});
                
                // Populate data word table
                const tableBody = document.getElementById('dataWordTableBody');
                tableBody.innerHTML = '';
                
                sortedData.forEach(d => {{
                    const row = tableBody.insertRow();
                    row.insertCell(0).textContent = d.msg_type || '';
                    row.insertCell(1).textContent = d.data_word || '';
                    row.insertCell(2).textContent = d.total_issues || 0;
                    row.insertCell(3).textContent = d.single_word_changes || 0;
                    row.insertCell(4).textContent = d.multi_word_changes || 0;
                    row.insertCell(5).textContent = `${{(d.avg_flip_speed_ms || 0).toFixed(3)}}`;
                    
                    const patternsCell = row.insertCell(6);
                    const patterns = d.top_error_patterns || 'N/A';
                    patternsCell.textContent = patterns.length > 100 ? patterns.substring(0, 100) + '...' : patterns;
                    patternsCell.title = patterns;
                }});
            }}
            
            function sortTable(tableId, column) {{
                // Simple table sort functionality
                const table = document.getElementById(tableId);
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                
                rows.sort((a, b) => {{
                    const aVal = a.cells[column].textContent;
                    const bVal = b.cells[column].textContent;
                    
                    // Try to parse as number first
                    const aNum = parseFloat(aVal);
                    const bNum = parseFloat(bVal);
                    
                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return bNum - aNum; // Descending for numbers
                    }}
                    
                    return aVal.localeCompare(bVal);
                }});
                
                // Re-append sorted rows
                rows.forEach(row => tbody.appendChild(row));
            }}
            
            // Initialize on page load
            initializeFilters();
            updateStats();
            updateFilteredCount();
            drawAllCharts();
        </script>
    </body>
    </html>
    """
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nEnhanced interactive dashboard saved to: {dashboard_path.absolute()}")
        return dashboard_path
