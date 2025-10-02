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
        Load all requirements Excel files and extract relevant data
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
                # Extract requirement name from filename (e.g., ps-3000 from ps-3000_AllData.xlsx)
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
                    
                    # Extract message types tested
                    msg_types_tested = []
                    for col in msg_type_cols:
                        msg_type = self.extract_message_type_from_column(col)
                        if msg_type and pd.notna(row[col]):
                            # Check if there's actual data in this column for this row
                            if str(row[col]).strip():
                                msg_types_tested.append(msg_type)
                    
                    if unit_id and save and station:
                        requirements_data.append({
                            'requirement_name': requirement_name,
                            'unit_id': unit_id,
                            'save': save,
                            'station': station,
                            'ofp': ofp,
                            'msg_types_tested': msg_types_tested,
                            'msg_types_str': ', '.join(sorted(set(msg_types_tested)))
                        })
                
                print(f"  Loaded {excel_file.name}: {len(df)} rows")
                
            except Exception as e:
                print(f"  Error loading {excel_file.name}: {e}")
        
        print(f"  Total requirements records loaded: {len(requirements_data)}")
        return requirements_data
    
    def analyze_requirements_at_risk(self):
        """
        Cross-reference bus flips with requirements to identify affected requirements
        Only includes requirements where bus flips match their tested message types
        """
        if self.df_flips is None or self.df_flips.empty:
            print("No bus flips to analyze for requirements")
            return
        
        # Load requirements data
        requirements_data = self.load_requirements_files()
        
        if not requirements_data:
            print("No requirements data loaded")
            return
        
        # Create a lookup of bus flip issues by (unit_id, station, save, msg_type)
        flip_lookup = defaultdict(list)
        for _, flip in self.df_flips.iterrows():
            if flip.get('msg_type'):
                key = (str(flip['unit_id']), str(flip['station']), str(flip['save']), str(flip['msg_type']))
                flip_lookup[key].append({
                    'bus_transition': flip.get('bus_transition', ''),
                    'timestamp_busA': flip.get('timestamp_busA', 0),
                    'timestamp_busB': flip.get('timestamp_busB', 0)
                })
        
        # Check each requirement against bus flips
        # Use a set to track unique combinations and avoid duplicates
        seen_combinations = set()
        affected_requirements = []
        
        for req in requirements_data:
            # Get unique message types for this requirement
            unique_msg_types = list(set(req['msg_types_tested']))
            
            # Check each unique message type this requirement tests
            for msg_type in unique_msg_types:
                key = (req['unit_id'], req['station'], req['save'], msg_type)
                
                if key in flip_lookup:
                    # Create a unique identifier for this combination
                    combo_key = (req['requirement_name'], req['unit_id'], req['station'], 
                                 req['save'], msg_type)
                    
                    # Only add if we haven't seen this combination before
                    if combo_key not in seen_combinations:
                        seen_combinations.add(combo_key)
                        
                        # This requirement has bus flips for a message type it tests
                        flips_info = flip_lookup[key]
                        
                        affected_requirements.append({
                            'requirement_name': req['requirement_name'],
                            'unit_id': req['unit_id'],
                            'station': req['station'],
                            'save': req['save'],
                            'ofp': req['ofp'],
                            'msg_type_affected': msg_type,
                            'flip_count': len(flips_info),
                            'bus_transitions': ', '.join(sorted(set([f['bus_transition'] for f in flips_info])))
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
                'save': 'nunique'
            }).reset_index()
            
            req_summary.columns = ['requirement_name', 'affected_message_types', 'total_flips', 
                                  'unique_units', 'unique_stations', 'unique_saves']
            req_summary = req_summary.sort_values('total_flips', ascending=False)
            self.df_requirements_summary = req_summary
            
            print(f"\nRequirements Analysis Complete:")
            print(f"  Total affected requirement entries: {len(affected_requirements)}")
            print(f"  Unique requirements with issues: {len(self.df_requirements_summary)}")
    
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
        
        print(f"\nExcel file saved to: {excel_path.absolute()}")
        return excel_path
    
    def create_interactive_dashboard(self):
        """Create an interactive HTML dashboard with filters"""
        import json
        
        dashboard_path = self.output_folder / "dashboard.html"
        
        # Prepare data for JavaScript
        flips_data = []
        if self.df_flips is not None and not self.df_flips.empty:
            # Convert timestamps to strings to avoid JSON serialization issues
            df_temp = self.df_flips.copy()
            for col in ['timestamp_busA', 'timestamp_busB']:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col].astype(str)
            flips_data = df_temp.to_dict('records')
        
        # Get unique values for filters
        unit_ids = sorted(self.df_flips['unit_id'].unique().tolist()) if self.df_flips is not None else []
        stations = sorted(self.df_flips['station'].unique().tolist()) if self.df_flips is not None else []
        saves = sorted(self.df_flips['save'].unique().tolist()) if self.df_flips is not None else []
        msg_types = sorted(self.df_flips['msg_type'].dropna().unique().tolist()) if self.df_flips is not None else []
        
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
        
        # Calculate R vs T message counts
        r_messages = 0
        t_messages = 0
        other_messages = 0
        if self.df_flips is not None and not self.df_flips.empty:
            for msg_type in self.df_flips['msg_type'].dropna():
                msg_str = str(msg_type).strip()
                # Extract pattern like 19R, 27T, etc.
                if re.match(r'^\d+R', msg_str):
                    r_messages += 1
                elif re.match(r'^\d+T', msg_str):
                    t_messages += 1
                else:
                    other_messages += 1
        
        # Get data word analysis for dashboard
        data_word_data = []
        if self.df_data_word_analysis is not None and not self.df_data_word_analysis.empty:
            data_word_data = self.df_data_word_analysis.head(20).to_dict('records')
        
        # Convert to JSON for JavaScript
        flips_data_json = json.dumps(flips_data)
        data_word_data_json = json.dumps(data_word_data)
        unit_ids_json = json.dumps(unit_ids)
        stations_json = json.dumps(stations)
        saves_json = json.dumps(saves)
        msg_types_json = json.dumps(msg_types)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Bus Monitor Interactive Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 15px;
        }}
        .filters {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 30px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
        }}
        .filter-group {{
            display: flex;
            flex-direction: column;
        }}
        .filter-group label {{
            font-weight: bold;
            margin-bottom: 5px;
            color: #555;
        }}
        .filter-group select {{
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            font-size: 14px;
        }}
        .stats-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card.warning {{
            background: linear-gradient(135deg, #ff9800 0%, #ff5722 100%);
        }}
        .stat-card.info {{
            background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%);
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .stat-label {{
            margin-top: 5px;
            opacity: 0.9;
            font-size: 14px;
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        #filteredCount {{
            background: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            display: inline-block;
            margin: 20px 0;
            font-weight: bold;
        }}
        .reset-btn {{
            background: #ff9800;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin-left: 10px;
        }}
        .reset-btn:hover {{
            background: #e68900;
        }}
        .pivot-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }}
        .pivot-table th {{
            background: #2196F3;
            color: white;
            padding: 8px;
            text-align: center;
            border: 1px solid #ddd;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .pivot-table td {{
            padding: 6px;
            text-align: center;
            border: 1px solid #ddd;
            cursor: pointer;
        }}
        .pivot-table tr:hover {{
            background: rgba(33, 150, 243, 0.1);
        }}
        .pivot-table td:hover {{
            background: rgba(33, 150, 243, 0.2);
            font-weight: bold;
        }}
        .pivot-table .row-header {{
            background: #f5f5f5;
            font-weight: bold;
            text-align: left;
            position: sticky;
            left: 0;
            z-index: 5;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .data-table th {{
            background: #4CAF50;
            color: white;
            padding: 10px;
            text-align: left;
            cursor: pointer;
        }}
        .data-table td {{
            padding: 8px;
            border-bottom: 1px solid #ddd;
        }}
        .data-table tr:hover {{
            background: #f9f9f9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Bus Monitor Interactive Dashboard</h1>
        
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-value">{total_flips:,}</div>
                <div class="stat-label">Bus Flips (with changes)</div>
            </div>
            <div class="stat-card warning">
                <div class="stat-value">{total_flips_no_changes:,}</div>
                <div class="stat-label">Flips (no changes)</div>
            </div>
            <div class="stat-card info">
                <div class="stat-value">{flip_percentage:.3f}%</div>
                <div class="stat-label">Flip Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.total_messages_processed:,}</div>
                <div class="stat-label">Total Messages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_units}</div>
                <div class="stat-label">Unit IDs</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_stations}</div>
                <div class="stat-label">Stations</div>
            </div>
        </div>
        
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
        </div>
        
        <div>
            <span id="filteredCount">Showing all {total_flips} flips</span>
            <button class="reset-btn" onclick="resetAllFilters()">Reset All Filters</button>
        </div>
        
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
            <div class="chart-title">R vs T Message Distribution</div>
            <div id="rtChart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Bus Flips by Message Type</div>
            <div id="msgTypeChart"></div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Data Word Error Analysis - Pivot Table</div>
            <div style="margin-bottom: 15px; padding: 15px; background: #f0f7ff; border-left: 4px solid #2196F3; border-radius: 4px;">
                <strong>What This Shows:</strong> Matrix view of data word errors by message type<br>
                • Numbers = Total errors for that data word in that message type<br>
                • Color intensity = Severity (darker = more errors)<br>
                • Click cells for detailed error patterns
            </div>
            
            <div style="overflow-x: auto; max-height: 400px; overflow-y: auto;">
                <table id="dataWordPivotTable" class="pivot-table">
                    <thead id="pivotTableHeader"></thead>
                    <tbody id="pivotTableBody"></tbody>
                </table>
            </div>
            
            <div id="dataWordSummaryStats" style="margin-top: 20px; padding: 15px; background: #f9f9f9; border-radius: 8px;">
                <h4>Summary Statistics</h4>
                <div id="summaryStatsContent"></div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Bus Flip Timing Analysis - Summary Table</div>
            <div style="margin-bottom: 15px; padding: 15px; background: #f0f7ff; border-left: 4px solid #2196F3; border-radius: 4px;">
                <strong>Timing Statistics by Station/Message Type/Save</strong><br>
                • Shows average, min, max, and count of flip speeds<br>
                • All times in milliseconds (ms)
            </div>
            
            <div id="timingSummaryStats" style="padding: 15px; background: #f9f9f9; border-radius: 8px; margin-bottom: 20px;">
                <div id="timingStatsContent"></div>
            </div>
            
            <div style="overflow-x: auto;">
                <table id="timingTable" class="data-table">
                    <thead>
                        <tr>
                            <th onclick="sortTimingTable('station')">Station ↕</th>
                            <th onclick="sortTimingTable('msg_type')">Message Type ↕</th>
                            <th onclick="sortTimingTable('save')">Save ↕</th>
                            <th onclick="sortTimingTable('count')">Count ↕</th>
                            <th onclick="sortTimingTable('avg')">Avg (ms) ↕</th>
                            <th onclick="sortTimingTable('min')">Min (ms) ↕</th>
                            <th onclick="sortTimingTable('max')">Max (ms) ↕</th>
                            <th onclick="sortTimingTable('std')">Std Dev ↕</th>
                        </tr>
                    </thead>
                    <tbody id="timingTableBody"></tbody>
                </table>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Hierarchical View: Unit → Station → Save</div>
            <div id="hierarchicalChart"></div>
        </div>
    </div>
    
    <script>
        // Data from Python
        const allData = {flips_data_json};
        const dataWordData = {data_word_data_json};
        let filteredData = [...allData];
        let filteredDataWordData = [...dataWordData];
        
        // Unique values for filters
        const uniqueUnits = {unit_ids_json};
        const uniqueStations = {stations_json};
        const uniqueSaves = {saves_json};
        const uniqueMsgTypes = {msg_types_json};
        
        // Initialize global filters
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
        }}
        
        function updateAllCharts() {{
            const unitFilter = document.getElementById('globalUnitFilter').value;
            const stationFilter = document.getElementById('globalStationFilter').value;
            const saveFilter = document.getElementById('globalSaveFilter').value;
            const msgTypeFilter = document.getElementById('globalMsgTypeFilter').value;
            
            // Filter main data
            filteredData = allData.filter(row => {{
                return (!unitFilter || row.unit_id === unitFilter) &&
                       (!stationFilter || row.station === stationFilter) &&
                       (!saveFilter || row.save === saveFilter) &&
                       (!msgTypeFilter || row.msg_type === msgTypeFilter);
            }});
            
            // Filter data word data based on same filters
            filteredDataWordData = dataWordData.filter(row => {{
                return (!msgTypeFilter || row.msg_type === msgTypeFilter);
            }});
            
            // Update filtered count
            document.getElementById('filteredCount').textContent = 
                `Showing ${{filteredData.length}} of ${{allData.length}} flips`;
            
            // Redraw all charts
            drawCharts();
        }}
        
        function resetAllFilters() {{
            document.getElementById('globalUnitFilter').value = '';
            document.getElementById('globalStationFilter').value = '';
            document.getElementById('globalSaveFilter').value = '';
            document.getElementById('globalMsgTypeFilter').value = '';
            updateAllCharts();
        }}
        
        function drawCharts() {{
            // Unit ID Chart
            const unitCounts = {{}};
            filteredData.forEach(d => {{
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
            filteredData.forEach(d => {{
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
            filteredData.forEach(d => {{
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
            
            // R vs T Message Chart
            const rCount = filteredData.filter(d => {{
                if (!d.msg_type) return false;
                const msgStr = d.msg_type.toString().trim();
                return /^\\d+R/.test(msgStr);
            }}).length;
            
            const tCount = filteredData.filter(d => {{
                if (!d.msg_type) return false;
                const msgStr = d.msg_type.toString().trim();
                return /^\\d+T/.test(msgStr);
            }}).length;
            
            Plotly.newPlot('rtChart', [{{
                x: ['R Messages', 'T Messages'],
                y: [rCount, tCount],
                type: 'bar',
                marker: {{
                    color: ['#4CAF50', '#2196F3']
                }},
                text: [rCount, tCount],
                textposition: 'auto',
                hovertemplate: '%{{x}}: %{{y}}<extra></extra>'
            }}], {{
                margin: {{ t: 10, b: 40, l: 60, r: 20 }},
                height: 350,
                yaxis: {{ title: 'Count' }},
                showlegend: false
            }});
            
            // Message Type Chart
            const msgTypeCounts = {{}};
            filteredData.forEach(d => {{
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
                yaxis: {{ title: 'Number of Flips (with data changes)' }}
            }});
            
            // Data Word Pivot Table
            createDataWordPivotTable();
            
            // Timing Analysis Table
            createTimingAnalysisTable();
        }}
        
        function createDataWordPivotTable() {{
            if (filteredDataWordData.length === 0) {{
                document.getElementById('pivotTableBody').innerHTML = 
                    '<tr><td colspan="100%">No data matching filters</td></tr>';
                return;
            }}
            
            // Create pivot data structure
            const pivotData = {{}};
            const dataWords = new Set();
            const msgTypes = new Set();
            
            filteredDataWordData.forEach(d => {{
                if (!pivotData[d.msg_type]) {{
                    pivotData[d.msg_type] = {{}};
                }}
                pivotData[d.msg_type][d.data_word] = {{
                    count: d.total_issues,
                    single: d.single_word_changes || 0,
                    multi: d.multi_word_changes || 0,
                    patterns: d.top_error_patterns
                }};
                dataWords.add(d.data_word);
                msgTypes.add(d.msg_type);
            }});
            
            const sortedDataWords = Array.from(dataWords).sort();
            const sortedMsgTypes = Array.from(msgTypes).sort();
            
            // Create header
            const headerHtml = `
                <tr>
                    <th class="row-header">Msg Type \\ Data Word</th>
                    ${{sortedDataWords.map(dw => `<th>${{dw}}</th>`).join('')}}
                    <th style="background: #ff9800;">Total</th>
                </tr>
            `;
            document.getElementById('pivotTableHeader').innerHTML = headerHtml;
            
            // Create body
            let bodyHtml = '';
            const columnTotals = {{}};
            
            sortedMsgTypes.forEach(msgType => {{
                let rowTotal = 0;
                let rowHtml = `<td class="row-header">${{msgType}}</td>`;
                
                sortedDataWords.forEach(dataWord => {{
                    const cellData = pivotData[msgType] && pivotData[msgType][dataWord];
                    const count = cellData ? cellData.count : 0;
                    rowTotal += count;
                    columnTotals[dataWord] = (columnTotals[dataWord] || 0) + count;
                    
                    const bgColor = count === 0 ? '#fff' : 
                                  count < 10 ? '#e8f5e9' :
                                  count < 50 ? '#ffeb3b' :
                                  count < 100 ? '#ff9800' : '#f44336';
                    const textColor = count > 50 ? 'white' : 'black';
                    
                    rowHtml += `<td style="background: ${{bgColor}}; color: ${{textColor}};" 
                                   onclick="showCellDetails('${{msgType}}', '${{dataWord}}')"
                                   title="${{cellData ? `Single: ${{cellData.single}}, Multi: ${{cellData.multi}}` : ''}}">${{count || '-'}}</td>`;
                }});
                
                rowHtml += `<td style="background: #fff3e0; font-weight: bold;">${{rowTotal}}</td>`;
                bodyHtml += `<tr>${{rowHtml}}</tr>`;
            }});
            
            // Add totals row
            let totalRow = '<tr><td class="row-header" style="background: #ff9800; color: white;">TOTAL</td>';
            let grandTotal = 0;
            sortedDataWords.forEach(dw => {{
                const total = columnTotals[dw] || 0;
                grandTotal += total;
                totalRow += `<td style="background: #fff3e0; font-weight: bold;">${{total}}</td>`;
            }});
            totalRow += `<td style="background: #ff5722; color: white; font-weight: bold;">${{grandTotal}}</td></tr>`;
            
            document.getElementById('pivotTableBody').innerHTML = bodyHtml + totalRow;
            
            // Update summary stats
            updateDataWordSummaryStats(pivotData, sortedMsgTypes, sortedDataWords);
        }}
        
        function updateDataWordSummaryStats(pivotData, msgTypes, dataWords) {{
            const totalIssues = filteredDataWordData.reduce((sum, d) => sum + d.total_issues, 0);
            const avgIssuesPerCombo = totalIssues / filteredDataWordData.length || 0;
            
            // Find most problematic combinations
            const topIssues = filteredDataWordData
                .sort((a, b) => b.total_issues - a.total_issues)
                .slice(0, 5);
            
            // Calculate multi-word percentage
            const totalMulti = filteredDataWordData.reduce((sum, d) => sum + (d.multi_word_changes || 0), 0);
            const totalSingle = filteredDataWordData.reduce((sum, d) => sum + (d.single_word_changes || 0), 0);
            const multiPercent = totalMulti + totalSingle > 0 ? (totalMulti / (totalMulti + totalSingle) * 100).toFixed(1) : 0;
            
            const statsHtml = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                    <div><strong>Total Data Word Issues:</strong> ${{totalIssues}}</div>
                    <div><strong>Unique Message Types:</strong> ${{msgTypes.length}}</div>
                    <div><strong>Unique Data Words:</strong> ${{dataWords.length}}</div>
                    <div><strong>Multi-Word Change Rate:</strong> ${{multiPercent}}%</div>
                    <div><strong>Avg Issues per Combo:</strong> ${{avgIssuesPerCombo.toFixed(1)}}</div>
                </div>
                <div style="margin-top: 15px;">
                    <strong>Top 5 Problem Areas:</strong>
                    <ol>
                        ${{topIssues.map(d => `<li>${{d.msg_type}}-${{d.data_word}}: ${{d.total_issues}} issues</li>`).join('')}}
                    </ol>
                </div>
            `;
            
            document.getElementById('summaryStatsContent').innerHTML = statsHtml;
        }}
        
        function showCellDetails(msgType, dataWord) {{
            const item = filteredDataWordData.find(d => d.msg_type === msgType && d.data_word === dataWord);
            if (!item) return;
            
            alert(`${{msgType}} - ${{dataWord}}\\n\\nTotal Issues: ${{item.total_issues}}\\nSingle Word Changes: ${{item.single_word_changes || 0}}\\nMulti Word Changes: ${{item.multi_word_changes || 0}}\\n\\nTop Patterns:\\n${{item.top_error_patterns || 'N/A'}}`);
        }}
        
        function createTimingAnalysisTable() {{
            if (filteredData.length === 0) {{
                document.getElementById('timingTableBody').innerHTML = 
                    '<tr><td colspan="8">No data matching filters</td></tr>';
                return;
            }}
            
            // Group timing data by station, msg_type, save
            const timingGroups = {{}};
            filteredData.forEach(d => {{
                const key = `${{d.station}}|${{d.msg_type || 'unknown'}}|${{d.save}}`;
                if (!timingGroups[key]) {{
                    timingGroups[key] = {{
                        station: d.station,
                        msg_type: d.msg_type || 'unknown',
                        save: d.save,
                        timings: []
                    }};
                }}
                const ms = d.timestamp_diff_ms || (d.timestamp_diff * 1000);
                if (ms && ms < 100) {{
                    timingGroups[key].timings.push(ms);
                }}
            }});
            
            // Calculate statistics for each group
            const timingStats = [];
            Object.values(timingGroups).forEach(group => {{
                if (group.timings.length > 0) {{
                    const sorted = group.timings.sort((a, b) => a - b);
                    const sum = sorted.reduce((a, b) => a + b, 0);
                    const avg = sum / sorted.length;
                    const min = sorted[0];
                    const max = sorted[sorted.length - 1];
                    
                    // Calculate standard deviation
                    const squaredDiffs = sorted.map(v => Math.pow(v - avg, 2));
                    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / sorted.length;
                    const stdDev = Math.sqrt(avgSquaredDiff);
                    
                    timingStats.push({{
                        station: group.station,
                        msg_type: group.msg_type,
                        save: group.save,
                        count: group.timings.length,
                        avg: avg,
                        min: min,
                        max: max,
                        std: stdDev
                    }});
                }}
            }});
            
            // Sort by count descending
            timingStats.sort((a, b) => b.count - a.count);
            
            // Create table rows
            let tableHtml = '';
            timingStats.forEach(stat => {{
                tableHtml += `
                    <tr>
                        <td>${{stat.station}}</td>
                        <td>${{stat.msg_type}}</td>
                        <td>${{stat.save}}</td>
                        <td>${{stat.count}}</td>
                        <td>${{stat.avg.toFixed(3)}}</td>
                        <td>${{stat.min.toFixed(3)}}</td>
                        <td>${{stat.max.toFixed(3)}}</td>
                        <td>${{stat.std.toFixed(3)}}</td>
                    </tr>
                `;
            }});
            
            document.getElementById('timingTableBody').innerHTML = tableHtml;
            
            // Update overall timing stats
            updateTimingSummaryStats(filteredData);
        }}
        
        function updateTimingSummaryStats(data) {{
            const timings = data.map(d => d.timestamp_diff_ms || (d.timestamp_diff * 1000))
                                .filter(t => t && t < 100);
            
            if (timings.length === 0) {{
                document.getElementById('timingStatsContent').innerHTML = 'No timing data available';
                return;
            }}
            
            const sorted = timings.sort((a, b) => a - b);
            const sum = sorted.reduce((a, b) => a + b, 0);
            const avg = sum / sorted.length;
            const min = sorted[0];
            const max = sorted[sorted.length - 1];
            const median = sorted[Math.floor(sorted.length / 2)];
            
            // Find percentiles
            const p25 = sorted[Math.floor(sorted.length * 0.25)];
            const p75 = sorted[Math.floor(sorted.length * 0.75)];
            const p95 = sorted[Math.floor(sorted.length * 0.95)];
            
            const statsHtml = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                    <div><strong>Total Flips:</strong> ${{timings.length}}</div>
                    <div><strong>Average:</strong> ${{avg.toFixed(3)}} ms</div>
                    <div><strong>Median:</strong> ${{median.toFixed(3)}} ms</div>
                    <div><strong>Min:</strong> ${{min.toFixed(3)}} ms</div>
                    <div><strong>Max:</strong> ${{max.toFixed(3)}} ms</div>
                    <div><strong>25th %ile:</strong> ${{p25.toFixed(3)}} ms</div>
                    <div><strong>75th %ile:</strong> ${{p75.toFixed(3)}} ms</div>
                    <div><strong>95th %ile:</strong> ${{p95.toFixed(3)}} ms</div>
                </div>
            `;
            
            document.getElementById('timingStatsContent').innerHTML = statsHtml;
        }}
        
        let timingSortColumn = 'count';
        let timingSortAsc = false;
        
        function sortTimingTable(column) {{
            if (column === timingSortColumn) {{
                timingSortAsc = !timingSortAsc;
            }} else {{
                timingSortColumn = column;
                timingSortAsc = true;
            }}
            
            createTimingAnalysisTable();
        }}
        
        // Initialize on page load
        initializeFilters();
        drawCharts();
    </script>
        
        // Initialize filters
        function initializeFilters() {{
            const unitFilter = document.getElementById('unitFilter');
            uniqueUnits.forEach(unit => {{
                const option = document.createElement('option');
                option.value = unit;
                option.textContent = unit;
                unitFilter.appendChild(option);
            }});
            
            const stationFilter = document.getElementById('stationFilter');
            uniqueStations.forEach(station => {{
                const option = document.createElement('option');
                option.value = station;
                option.textContent = station;
                stationFilter.appendChild(option);
            }});
            
            const saveFilter = document.getElementById('saveFilter');
            uniqueSaves.forEach(save => {{
                const option = document.createElement('option');
                option.value = save;
                option.textContent = save;
                saveFilter.appendChild(option);
            }});
            
            const msgTypeFilter = document.getElementById('msgTypeFilter');
            uniqueMsgTypes.forEach(msgType => {{
                const option = document.createElement('option');
                option.value = msgType;
                option.textContent = msgType;
                msgTypeFilter.appendChild(option);
            }});
        }}
        
        function updateFilters() {{
            const unitFilter = document.getElementById('unitFilter').value;
            const stationFilter = document.getElementById('stationFilter').value;
            const saveFilter = document.getElementById('saveFilter').value;
            const msgTypeFilter = document.getElementById('msgTypeFilter').value;
            
            // Filter data
            filteredData = allData.filter(row => {{
                return (!unitFilter || row.unit_id === unitFilter) &&
                       (!stationFilter || row.station === stationFilter) &&
                       (!saveFilter || row.save === saveFilter) &&
                       (!msgTypeFilter || row.msg_type === msgTypeFilter);
            }});
            
            // Update filtered count
            document.getElementById('filteredCount').textContent = 
                `Showing ${{filteredData.length}} of ${{allData.length}} flips`;
            
            // Update available options based on current selection
            updateAvailableOptions();
            
            // Redraw charts
            drawCharts();
        }}
        
        function updateAvailableOptions() {{
            // Get current selections
            const unitFilter = document.getElementById('unitFilter').value;
            const stationFilter = document.getElementById('stationFilter').value;
            const saveFilter = document.getElementById('saveFilter').value;
            
            // Update station options based on selected unit
            if (unitFilter) {{
                const availableStations = [...new Set(filteredData.map(d => d.station))];
                updateSelectOptions('stationFilter', availableStations, stationFilter);
            }}
            
            // Update save options based on selected unit and station
            if (unitFilter || stationFilter) {{
                const availableSaves = [...new Set(filteredData.map(d => d.save))];
                updateSelectOptions('saveFilter', availableSaves, saveFilter);
            }}
        }}
        
        function updateSelectOptions(selectId, options, currentValue) {{
            const select = document.getElementById(selectId);
            const previousValue = select.value;
            
            // Remove all options except the first (All)
            while (select.options.length > 1) {{
                select.remove(1);
            }}
            
            // Add new options
            options.sort().forEach(option => {{
                const optionElement = document.createElement('option');
                optionElement.value = option;
                optionElement.textContent = option;
                select.appendChild(optionElement);
            }});
            
            // Restore previous selection if it's still valid
            if (options.includes(previousValue)) {{
                select.value = previousValue;
            }}
        }}
        
        function resetFilters() {{
            document.getElementById('unitFilter').value = '';
            document.getElementById('stationFilter').value = '';
            document.getElementById('saveFilter').value = '';
            document.getElementById('msgTypeFilter').value = '';
            updateFilters();
        }}
        
        function drawCharts() {{
            // Unit ID Chart
            const unitCounts = {{}};
            filteredData.forEach(d => {{
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
            filteredData.forEach(d => {{
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
            filteredData.forEach(d => {{
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
            
            // R vs T Message Chart
            const rCount = filteredData.filter(d => {{
                if (!d.msg_type) return false;
                const msgStr = d.msg_type.toString().trim();
                return /^\\d+R/.test(msgStr);  // Match patterns like 19R, 27R, etc.
            }}).length;
            
            const tCount = filteredData.filter(d => {{
                if (!d.msg_type) return false;
                const msgStr = d.msg_type.toString().trim();
                return /^\\d+T/.test(msgStr);  // Match patterns like 19T, 27T, etc.
            }}).length;
            
            // Bar chart for R vs T distribution
            Plotly.newPlot('rtChart', [{{
                x: ['R Messages', 'T Messages'],
                y: [rCount, tCount],
                type: 'bar',
                marker: {{
                    color: ['#4CAF50', '#2196F3']
                }},
                text: [rCount, tCount],
                textposition: 'auto',
                hovertemplate: '%{{x}}: %{{y}}<extra></extra>'
            }}], {{
                margin: {{ t: 10, b: 40, l: 60, r: 20 }},
                height: 350,
                yaxis: {{ title: 'Count' }},
                showlegend: false
            }});
            
            // Message Type Chart
            const msgTypeCounts = {{}};
            filteredData.forEach(d => {{
                if (d.msg_type) {{
                    msgTypeCounts[d.msg_type] = (msgTypeCounts[d.msg_type] || 0) + 1;
                }}
            }});
            
            // Sort by count and take top 20
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
                yaxis: {{ title: 'Number of Flips (with data changes)' }}
            }});
            
        // Initialize on page load
        initializeFilters();
        drawCharts();
    </script>
</body>
</html>
"""
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nInteractive dashboard saved to: {dashboard_path.absolute()}")
        return dashboard_path
    </script>
</body>
</html>
"""
        
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nInteractive dashboard saved to: {dashboard_path.absolute()}")
        return dashboard_path
