def create_test_case_risk_analysis(self):
        """
        Create comprehensive test case risk analysis.
        Each bus flip is counted ONCE per test case group, even with overlapping instances.
        Maps requirements based on lookup, NOT message type.
        """
        if self.df_test_cases is None or self.df_test_cases.empty:
            print("No test cases loaded for risk analysis")
            return
        
        if self.df_flips is None or self.df_flips.empty:
            print("No bus flips for risk analysis")
            return
        
        risk_analysis = []
        
        # Group test cases by base name (already split, no ampersands)
        grouped_test_cases = {}
        for _, tc in self.df_test_cases.iterrows():
            base_name = tc['test_case_id']
            
            if base_name not in grouped_test_cases:
                grouped_test_cases[base_name] = []
            grouped_test_cases[base_name].append(tc)
        
        # Analyze each grouped test case
        for base_name, test_case_instances in grouped_test_cases.items():
            # Track UNIQUE flips for THIS test case group
            # Key: (unit_id, station, save, msg_type, timestamp_busA)
            unique_flips_for_this_group = set()
            
            unit_station_save_flips = {}
            msg_type_flips = {}
            time_ranges = []
            requirements_at_risk = set()
            units_tested = set()
            
            # Analyze each instance of this test case group
            for tc_instance in test_case_instances:
                units_tested.add(tc_instance['unit_id'])
                
                # Find flips during this test case instance
                matching_flips = self.df_flips[
                    (self.df_flips['unit_id'] == tc_instance['unit_id']) &
                    (self.df_flips['station'] == tc_instance['station']) &
                    (self.df_flips['save'] == tc_instance['save']) &
                    (self.df_flips['timestamp_busA'] >= tc_instance['timestamp_start']) &
                    (self.df_flips['timestamp_busA'] <= tc_instance['timestamp_end'])
                ]
                
                if matching_flips.empty:
                    continue
                
                unit_station_save_key = f"{tc_instance['unit_id']}/{tc_instance['station']}/{tc_instance['save']}"
                
                # Process each flip
                for _, flip in matching_flips.iterrows():
                    # Create unique flip identifier
                    flip_id = (
                        str(flip['unit_id']),
                        str(flip['station']),
                        str(flip['save']),
                        str(flip['msg_type']),
                        float(flip['timestamp_busA'])
                    )
                    
                    # Only count this flip ONCE for THIS test case group
                    if flip_id not in unique_flips_for_this_group:
                        unique_flips_for_this_group.add(flip_id)
                        
                        # Track location
                        if unit_station_save_key not in unit_station_save_flips:
                            unit_station_save_flips[unit_station_save_key] = 0
                        unit_station_save_flips[unit_station_save_key] += 1
                        
                        # Track message type
                        msg_type = str(flip['msg_type']) if pd.notna(flip['msg_type']) else 'Unknown'
                        if msg_type not in msg_type_flips:
                            msg_type_flips[msg_type] = 0
                        msg_type_flips[msg_type] += 1
                
                # Track time ranges
                if len(matching_flips) > 0:
                    flip_times = matching_flips['timestamp_busA'].tolist()
                    time_ranges.append({
                        'start': min(flip_times),
                        'end': max(flip_times),
                        'unit_station_save': unit_station_save_key
                    })
            
            # Find requirements at risk
            total_flips = len(unique_flips_for_this_group)
            
            # *** MODIFICATION START ***
            # Only map requirements if bus flips were found
            if total_flips > 0 and self.requirement_testcase_mapping:
                for requirement, test_cases in self.requirement_testcase_mapping.items():
                    if base_name in test_cases:
                        # User wants to map ANY requirement linked to this test case,
                        # if the test case had ANY flips.
                        requirements_at_risk.add(requirement)
            # *** MODIFICATION END ***
            
            if total_flips > 0:
                sorted_combos = sorted(unit_station_save_flips.items(), key=lambda x: x[1], reverse=True)
                sorted_msg_types = sorted(msg_type_flips.items(), key=lambda x: x[1], reverse=True)
                
                # Format time ranges
                time_range_strs = []
                for tr in time_ranges[:10]:
                    start_str = pd.to_datetime(tr['start'], unit='s').strftime('%Y-%m-%d %H:%M:%S')
                    end_str = pd.to_datetime(tr['end'], unit='s').strftime('%Y-%m-%d %H:%M:%S')
                    time_range_strs.append(f"{tr['unit_station_save']}: {start_str} to {end_str}")
                
                risk_analysis.append({
                    'test_case': base_name,
                    'total_bus_flips': total_flips,
                    'units_tested': ', '.join(sorted(units_tested)),
                    'num_units': len(units_tested),
                    'instances_run': len(test_case_instances),
                    'avg_flips_per_run': round(total_flips / len(test_case_instances), 1),
                    'top_station_save_combo': sorted_combos[0][0] if sorted_combos else '',
                    'top_combo_flips': sorted_combos[0][1] if sorted_combos else 0,
                    'station_save_combos': ', '.join([f"{k}({v})" for k, v in sorted_combos[:5]]),
                    'top_msg_types': ', '.join([f"{k}({v})" for k, v in sorted_msg_types[:5]]),
                    'time_ranges': '\n'.join(time_range_strs) if time_range_strs else 'N/A',
                    'requirements_at_risk': len(requirements_at_risk),
                    'requirement_list': ', '.join(sorted(requirements_at_risk)[:10]) if requirements_at_risk else 'None'
                })
        
        if risk_analysis:
            self.df_test_case_risk = pd.DataFrame(risk_analysis)
            self.df_test_case_risk = self.df_test_case_risk.sort_values('total_bus_flips', ascending=False)
            
            print(f"\nTest Case Risk Analysis Complete:")
            print(f"  Test cases analyzed: {len(self.df_test_case_risk)}")
            print(f"  Test cases with bus flips: {len(self.df_test_case_risk[self.df_test_case_risk['total_bus_flips'] > 0])}")
        else:
            self.df_test_case_risk = pd.DataFrame()

    def create_interactive_dashboard(self):
        """Create an enhanced interactive HTML dashboard with comprehensive filters and analytics"""
        import json
        from datetime import datetime
        
        dashboard_path = self.output_folder / "dashboard.html"
        
        # ... (rest of the data preparation remains the same) ...
        
        # Prepare data for JavaScript
        flips_data = []
        if self.df_flips is not None and not self.df_flips.empty:
            df_temp = self.df_flips.copy()
            for col in ['timestamp_busA', 'timestamp_busB']:
                if col in df_temp.columns:
                    df_temp[col] = df_temp[col].astype(str)
            flips_data = df_temp.to_dict('records')

        # Prepare test case risk data (ADD THIS)
        test_case_risk_data = []
        if hasattr(self, 'df_test_case_risk') and self.df_test_case_risk is not None:
            # *** MODIFICATION START ***
            # Ensure time_ranges is properly formatted for JS
            df_risk_temp = self.df_test_case_risk.copy()
            if 'time_ranges' in df_risk_temp.columns:
                df_risk_temp['time_ranges'] = df_risk_temp['time_ranges'].replace(r'\n', '\n', regex=False)
            test_case_risk_data = df_risk_temp.to_dict('records')
            # *** MODIFICATION END ***

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
            df_rates_temp = self.df_message_rates.copy()
            # Convert milliseconds to messages per second
            df_rates_temp['avg_rate'] = 1000 / df_rates_temp['avg_time_diff_ms']
            df_rates_temp['min_rate'] = 1000 / df_rates_temp['max_time_diff_ms']
            df_rates_temp['max_rate'] = 1000 / df_rates_temp['min_time_diff_ms']
            message_rates_data = df_rates_temp.to_dict('records')
        
        # Prepare message rates by location
        message_rates_by_location = []
        if self.df_message_rates is not None and not self.df_message_rates.empty:
            df_loc_temp = self.df_message_rates.copy()
            # Add rate calculations
            df_loc_temp['msg_per_sec'] = 1000 / df_loc_temp['avg_time_diff_ms']
            df_loc_temp['min_rate'] = 1000 / df_loc_temp['max_time_diff_ms']
            df_loc_temp['max_rate'] = 1000 / df_loc_temp['min_time_diff_ms']
            message_rates_by_location = df_loc_temp.to_dict('records')
        
        # Prepare failed requirements data
        failed_requirements_data = []
        if hasattr(self, 'df_failed_requirements_analysis') and self.df_failed_requirements_analysis is not None:
            failed_requirements_data = self.df_failed_requirements_analysis.to_dict('records')
        
        # Prepare requirements at risk data
        requirements_at_risk_data = []
        if self.df_requirements_at_risk is not None and not self.df_requirements_at_risk.empty:
            df_req_temp = self.df_requirements_at_risk.copy()
            if 'test_cases_affected' in df_req_temp.columns:
                df_req_temp['test_cases_affected'] = df_req_temp['test_cases_affected'].astype(str)
            requirements_at_risk_data = df_req_temp.to_dict('records')

        # Prepare test case requirements data (NEW)
        test_case_requirements_data = []
        if hasattr(self, 'df_test_case_req_with_flips') and self.df_test_case_req_with_flips is not None:
            test_case_requirements_data = self.df_test_case_req_with_flips.to_dict('records')

        # Prepare requirement-test case mapping (NEW)
        requirement_mapping_data = []
        if hasattr(self, 'df_requirement_test_case_mapping') and self.df_requirement_test_case_mapping is not None:
            requirement_mapping_data = self.df_requirement_test_case_mapping.to_dict('records')

        # Get unique requirement names for filter (NEW)
        requirement_names = []
        if hasattr(self, 'df_test_case_requirements') and self.df_test_case_requirements is not None:
            requirement_names = sorted(self.df_test_case_requirements['requirement_name'].unique().tolist())

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
        
        # Prepare data word analysis
        data_word_data = []
        if self.df_data_word_analysis is not None and not self.df_data_word_analysis.empty:
            data_word_data = self.df_data_word_analysis.head(50).to_dict('records')
        
        # Convert to JSON
        flips_data_json = json.dumps(flips_data)
        test_case_data_json = json.dumps(test_case_data)
        test_case_flip_data_json = json.dumps(test_case_flip_data)
        message_rates_data_json = json.dumps(message_rates_data)
        message_rates_by_location_json = json.dumps(message_rates_by_location)
        failed_requirements_data_json = json.dumps(failed_requirements_data)
        requirements_at_risk_data_json = json.dumps(requirements_at_risk_data)
        data_word_data_json = json.dumps(data_word_data)
        unit_ids_json = json.dumps(unit_ids)
        stations_json = json.dumps(stations)
        saves_json = json.dumps(saves)
        msg_types_json = json.dumps(msg_types)
        test_case_ids_json = json.dumps(test_case_ids)
        test_case_requirements_data_json = json.dumps(test_case_requirements_data)
        requirement_mapping_data_json = json.dumps(requirement_mapping_data)
        requirement_names_json = json.dumps(requirement_names)    
        test_case_risk_data_json = json.dumps(test_case_risk_data)   
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
            .data-table td {{
                padding: 10px;
                border-bottom: 1px solid #e1e8ed;
                white-space: pre-wrap; /* ADD THIS LINE */
                word-wrap: break-word; /* ADD THIS LINE */
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
                flex-wrap: wrap;
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
            
            <div id="statsRow" class="stats-row">
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
            
            <div class="tab-container">
                <div class="tab-buttons">
                    <button class="tab-button active" onclick="switchTab('overview')">Overview</button>
                    <button class="tab-button" onclick="switchTab('risk')">Test Case Risk</button>
                    <button class="tab-button" onclick="switchTab('timeline')">Timeline</button>
                    <button class="tab-button" onclick="switchTab('rates')">Message Rates</button>
                    <button class="tab-button" onclick="switchTab('testcases')">Test Cases</button>
                    <button class="tab-button" onclick="switchTab('requirements')">Requirements</button>
                    <button class="tab-button" onclick="switchTab('datawords')">Data Words</button>
                </div>
                
                <div id="overview-tab" class="tab-content active">
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
                        <div class="chart-title">Top 20 Message Types with Bus Flips</div>
                        <div id="msgTypeChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">R vs T Message Distribution</div>
                        <div id="rtChart"></div>
                    </div>
                </div>

                <div id="risk-tab" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">⚠️ Test Cases at Highest Risk of Bus Flip Errors</div>
                        <div class="info-box" style="background: linear-gradient(135deg, #fee 0%, #fcc 100%); border-left-color: #e74c3c;">
                            <strong>CRITICAL RISK ANALYSIS:</strong> These test cases have significant bus flip activity that could affect test results.
                            Focus on the top items - they represent the highest risk to requirement validation.
                        </div>
                        <div id="riskOverviewChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Station/Save Impact per Test Case</div>
                        <div id="riskStationSaveChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Mapped Requirements for Risky Test Cases</div>
                        <div id="riskRequirementsChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Detailed Test Case Risk Analysis</div>
                        <div style="overflow-x: auto;">
                            <table id="riskDetailTable" class="data-table">
                                <thead>
                                    <tr>
                                        <th onclick="sortTable('riskDetailTable', 0)">Test Case</th>
                                        <th onclick="sortTable('riskDetailTable', 1)">Total Flips</th>
                                        <th onclick="sortTable('riskDetailTable', 2)">Avg Flips/Run</th>
                                        <th onclick="sortTable('riskDetailTable', 3)">Top Station/Save</th>
                                        <th onclick="sortTable('riskDetailTable', 4)">All Station/Save Combos</th>
                                        <th onclick="sortTable('riskDetailTable', 5)">Top Message Types</th>
                                        <th onclick="sortTable('riskDetailTable', 6)">Mapped Requirements</th>
                                        <th onclick="sortTable('riskDetailTable', 7)">Time Ranges</th>
                                    </tr>
                                </thead>
                                <tbody id="riskDetailTableBody"></tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <div id="timeline-tab" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">Bus Flips Over Time</div>
                        <div class="info-box">
                            <strong>Timeline Analysis:</strong> Shows the distribution of bus flips over time. 
                            Look for unusual spikes or patterns that might indicate specific issues.
                        </div>
                        <div id="timelineChart"></div>
                    </div>

                    <div class="chart-container">
                        <div class="chart-title">Hourly Bus Flip Distribution</div>
                        <div id="hourlyChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Bus Flip Spike Analysis</div>
                        <div class="info-box">
                            <strong>Spike Details:</strong> Shows peak periods with test cases, message types, and station/save combinations during those times.
                        </div>
                        <div id="spikeAnalysis"></div>
                    </div>
                </div>
                
                <div id="rates-tab" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">Message Rate Statistics by Type</div>
                        <div class="info-box">
                            <strong>Message Rate Analysis:</strong> Shows sampling rates per station-save-msgtype combination.
                            Each bar represents the maximum rate observed for that message type across ALL locations.
                            Critical: >1000 Hz, Warning: >100 Hz.
                        </div>
                        <div id="messageRateStats"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Station Message Rate Summary</div>
                        <div id="stationRateSummary"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">High-Frequency Station-Save Combinations</div>
                        <div class="info-box">
                            <strong>Note:</strong> Rates are calculated per save, not aggregated across saves in a station.
                        </div>
                        <div id="stationSaveHighFreq"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Message Rate Metrics</div>
                        <div class="metric-grid" id="rateMetrics"></div>
                    </div>
                </div>
                
                <div id="testcases-tab" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">Test Cases with Most Bus Flips (Deduplicated)</div>
                        <div class="info-box">
                            <strong>Test Case Analysis:</strong> Shows grouped test cases with aggregated, deduplicated bus flip counts.
                            Data is sourced from the risk analysis to prevent double-counting from overlapping runs.
                        </div>
                        <div id="testCaseFlipCounts"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Test Case Detail Table (Deduplicated)</div>
                        <div style="overflow-x: auto;">
                            <table id="testCaseTable" class="data-table">
                                <thead>
                                    <tr>
                                        <th onclick="sortTable('testCaseTable', 0)">Test Case Group</th>
                                        <th onclick="sortTable('testCaseTable', 1)">Total Bus Flips</th>
                                        <th onclick="sortTable('testCaseTable', 2)">Runs</th>
                                        <th onclick="sortTable('testCaseTable', 3)">Avg Flips/Run</th>
                                        <th onclick="sortTable('testCaseTable', 4)">Unique Msg Types</th>
                                        <th onclick="sortTable('testCaseTable', 5)">Locations</th>
                                        <th onclick="sortTable('testCaseTable', 6)">Time Range (Sample)</th>
                                    </tr>
                                </thead>
                                <tbody id="testCaseTableBody"></tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <div id="requirements-tab" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">📋 Universal Requirements with Bus Flips</div>
                        <div class="info-box">
                            <strong>Universal Requirements:</strong> These requirements exist across multiple locations and have experienced bus flips.
                            Hover over bars to see station/save combinations and affected message types.
                        </div>
                        <div id="universalRequirementsChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">🎯 Test Case Requirements with Bus Flips</div>
                        <div class="info-box">
                            <strong>Test Case Requirements:</strong> Shows requirements tested in specific test cases that experienced bus flips.
                            Bus flips are counted across all test runs, stations, and saves where the requirement was tested.
                        </div>
                        <div id="testCaseRequirementsChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Requirement-Test Case Mapping</div>
                        <div class="info-box">
                            <strong>Mapping Info:</strong> Shows which requirements are tested by which test cases. 
                            Some requirements may appear in multiple test cases. (Uses grouped test case IDs).
                        </div>
                        <div style="overflow-x: auto;">
                            <table id="requirementMappingTable" class="data-table">
                                <thead>
                                    <tr>
                                        <th onclick="sortTable('requirementMappingTable', 0)">Requirement</th>
                                        <th onclick="sortTable('requirementMappingTable', 1)">Test Cases</th>
                                        <th onclick="sortTable('requirementMappingTable', 2)">Failures</th>
                                        <th onclick="sortTable('requirementMappingTable', 3)">Pass Rate %</th>
                                        <th onclick="sortTable('requirementMappingTable', 4)">Locations</th>
                                    </tr>
                                </thead>
                                <tbody id="requirementMappingTableBody"></tbody>
                            </table>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Failed Requirements with Bus Flips</div>
                        <div style="overflow-x: auto;">
                        <table id="testCaseReqFailuresTable" class="data-table">
                            <thead>
                                <tr>
                                    <th onclick="sortTable('testCaseReqFailuresTable', 0)">Requirement</th>
                                    <th onclick="sortTable('testCaseReqFailuresTable', 1)">Test Case</th>
                                    <th onclick="sortTable('testCaseReqFailuresTable', 2)">Location</th>
                                    <th onclick="sortTable('testCaseReqFailuresTable', 3)">Msg Types Tested</th>
                                    <th onclick="sortTable('testCaseReqFailuresTable', 4)">Total Bus Flips</th>
                                    <th onclick="sortTable('testCaseReqFailuresTable', 5)">Stations/Saves Affected</th>
                                </tr>
                            </thead>
                            <tbody id="testCaseReqFailuresTableBody"></tbody>
                        </table>
                        </div>
                    </div>
                </div>
                
                <div id="datawords-tab" class="tab-content">
                    <div class="chart-container">
                        <div class="chart-title">Single vs Multi-Word Change Distribution</div>
                        <div class="info-box">
                            <strong>Change Types:</strong> Shows the distribution of single-word vs multi-word changes across message types.
                            Multi-word changes often indicate more severe issues.
                        </div>
                        <div id="singleVsMultiChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Multi-Word Change Patterns</div>
                        <div class="info-box">
                            <strong>Pattern Analysis:</strong> Shows which data words commonly change together in multi-word scenarios.
                        </div>
                        <div id="multiWordPatternChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Error Pattern Analysis (Enhanced)</div>
                        <div class="info-box">
                            <strong>Common Patterns:</strong> Shows the most frequent error patterns with single/multi-word change indicators.
                            Helps identify systematic issues and pattern types.
                        </div>
                        <div id="errorPatternChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Data Word Change Speed Analysis</div>
                        <div class="info-box">
                            <strong>Timing Analysis:</strong> Shows the relationship between flip speed and change type.
                        </div>
                        <div id="changeSpeedChart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Data Word Statistics</div>
                        <div class="metric-grid" id="dataWordMetrics"></div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Interactive Data Word Pivot Table</div>
                        <div style="overflow-x: auto;">
                            <table id="dataWordPivotTable" class="data-table">
                                <thead>
                                    <tr>
                                        <th onclick="sortPivotTable(0)">Message Type</th>
                                        <th onclick="sortPivotTable(1)">Total Issues</th>
                                        <th onclick="sortPivotTable(2)">Single Word</th>
                                        <th onclick="sortPivotTable(3)">Multi Word</th>
                                        <th onclick="sortPivotTable(4)">% Multi</th>
                                        <th onclick="sortPivotTable(5)">Avg Words/Change</th>
                                        <th onclick="sortPivotTable(6)">Unique Patterns</th>
                                        <th onclick="sortPivotTable(7)">Top Pattern</th>
                                    </tr>
                                </thead>
                                <tbody id="dataWordPivotBody"></tbody>
                            </table>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <div class="chart-title">Detailed Data Word Analysis</div>
                        <div style="overflow-x: auto;">
                            <table id="dataWordTable" class="data-table">
                                <thead>
                                    <tr>
                                        <th onclick="sortTable('dataWordTable', 0)">Message Type</th>
                                        <th onclick="sortTable('dataWordTable', 1)">Data Word</th>
                                        <th onclick="sortTable('dataWordTable', 2)">Total Issues</th>
                                        <th onclick="sortTable('dataWordTable', 3)">Single/Multi</th>
                                        <th onclick="sortTable('dataWordTable', 4)">Common Pattern</th>
                                        <th onclick="sortTable('dataWordTable', 5)">Avg Speed (ms)</th>
                                        <th onclick="sortTable('dataWordTable', 6)">Locations</th>
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
            const messageRatesByLocation = {message_rates_by_location_json};
            const failedRequirementsData = {failed_requirements_data_json};
            const requirementsAtRiskData = {requirements_at_risk_data_json};
            const dataWordData = {data_word_data_json};
            const testCaseRequirementsData = {test_case_requirements_data_json};
            const requirementMappingData = {requirement_mapping_data_json};
            const testCaseRiskData = {test_case_risk_data_json};

            // Filtered data
            let filteredFlipsData = [...allFlipsData];
            let filteredTestCaseData = [...testCaseData];
            let filteredMessageRatesData = [...messageRatesData];
            let filteredMessageRatesByLocation = [...messageRatesByLocation];
            let filteredRequirementsData = [...requirementsAtRiskData];
            let filteredDataWordData = [...dataWordData];
            let filteredTestCaseRequirements = [...testCaseRequirementsData];  // NEW
            
            // Unique values for filters
            const uniqueUnits = {unit_ids_json};
            const uniqueStations = {stations_json};
            const uniqueSaves = {saves_json};
            const uniqueMsgTypes = {msg_types_json};
            const uniqueTestCases = {test_case_ids_json};
            const uniqueRequirements = {requirement_names_json};  // NEW
            
            // Helper function to parse test case names and handle deduplication
            function parseTestCaseName(testCaseId) {{
                // Handle combined test cases like "QS_2000_02&TW104_01"
                if (testCaseId.includes('&')) {{
                    return testCaseId.split('&').map(tc => parseIndividualTestCase(tc));
                }}
                return [parseIndividualTestCase(testCaseId)];
            }}
            
            function parseIndividualTestCase(testCase) {{
                // Remove trailing numbers and underscores to get base name
                // e.g., "QS_2000_02" -> "QS_2000"
                const match = testCase.match(/^(.+?)(_\\d+)?$/);
                return match ? match[1] : testCase;
            }}
            
            function groupTestCases(testCases) {{
                const grouped = {{}};
                
                testCases.forEach(tc => {{
                    const baseName = tc.test_case_id; // Already grouped in Python
                    
                    if (!grouped[baseName]) {{
                        grouped[baseName] = {{
                            baseName: baseName,
                            instances: [],
                            totalFlips: 0,
                            uniqueMsgTypes: new Set(),
                            uniqueLocations: new Set(),
                            timeRanges: [],
                            uniqueRuns: 0
                        }};
                    }}
                    
                    grouped[baseName].instances.push(tc);
                    
                    // Aggregate stats
                    if (tc.total_bus_flips) {{
                        grouped[baseName].totalFlips += tc.total_bus_flips;
                    }}
                    
                    if (tc.msg_types_list) {{
                        tc.msg_types_list.split(', ').forEach(mt => grouped[baseName].uniqueMsgTypes.add(mt));
                    }}
                    
                    const location = `${{tc.unit_id}}/${{tc.station}}/${{tc.save}}`;
                    grouped[baseName].uniqueLocations.add(location);
                    
                    // Track time ranges
                    if (tc.timestamp_start && tc.timestamp_end) {{
                        grouped[baseName].timeRanges.push({{
                            start: parseFloat(tc.timestamp_start),
                            end: parseFloat(tc.timestamp_end),
                            location: location
                        }});
                    }}
                }});
                
                // Count unique runs (non-overlapping time ranges per location)
                Object.values(grouped).forEach(group => {{
                    group.uniqueRuns = group.instances.length; // Python already deduplicated
                    
                    // Calculate average
                    group.avgFlipsPerRun = group.uniqueRuns > 0 ? 
                        Math.round(group.totalFlips / group.uniqueRuns) : 0;
                }});
                
                return grouped;
            }}
            
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
                
                // Filter by test case if selected - BE CAREFUL WITH OVERLAPS
                if (testCaseFilter && testCaseData.length > 0) {{
                    // Get ALL instances of this test case (may be multiple due to splits)
                    const testCaseInstances = testCaseData.filter(tc => tc.test_case_id === testCaseFilter);
                    
                    if (testCaseInstances.length > 0) {{
                        // Use Set to avoid counting same flip multiple times if ranges overlap
                        const flipIndices = new Set();
                        
                        filteredFlipsData.forEach((flip, index) => {{
                            const flipTime = parseFloat(flip.timestamp_busA);
                            
                            // Check if this flip falls within ANY instance of the test case
                            for (let tc of testCaseInstances) {{
                                const testStart = parseFloat(tc.timestamp_start);
                                const testEnd = parseFloat(tc.timestamp_end);
                                
                                if (flipTime >= testStart && flipTime <= testEnd) {{
                                    flipIndices.add(index);
                                    break; // Found a match, no need to check other instances
                                }}
                            }}
                        }});
                        
                        // Filter to only the flips that matched
                        filteredFlipsData = Array.from(flipIndices).map(i => filteredFlipsData[i]);
                    }}
                }}
                
                // Filter test case data
                filteredTestCaseData = testCaseData.filter(row => {{
                    return (!unitFilter || row.unit_id === unitFilter) &&
                        (!stationFilter || row.station === stationFilter) &&
                        (!saveFilter || row.save === saveFilter) &&
                        (!testCaseFilter || row.test_case_id === testCaseFilter);
                }});
                
                // Filter message rates by location
                filteredMessageRatesByLocation = messageRatesByLocation.filter(row => {{
                    return (!unitFilter || row.unit_id === unitFilter) &&
                        (!stationFilter || row.station === stationFilter) &&
                        (!saveFilter || row.save === saveFilter) &&
                        (!msgTypeFilter || row.msg_type === msgTypeFilter);
                }});
                
                // Filter message rates data for summary
                if (msgTypeFilter) {{
                    filteredMessageRatesData = messageRatesData.filter(row => row.msg_type === msgTypeFilter);
                }} else {{
                    filteredMessageRatesData = [...messageRatesData];
                }}

                // Filter universal requirements data (requirements at risk)
                filteredRequirementsData = requirementsAtRiskData.filter(row => {{
                    return (!unitFilter || row.unit_id === unitFilter) &&
                        (!stationFilter || row.station === stationFilter) &&
                        (!saveFilter || row.save === saveFilter) &&
                        (!msgTypeFilter || row.msg_type_affected === msgTypeFilter);
                }});                

                filteredTestCaseRequirements = testCaseRequirementsData.filter(row => {{
                    return (!unitFilter || row.unit_id === unitFilter) &&
                        (!stationFilter || row.station === stationFilter) &&
                        (!saveFilter || row.save === saveFilter) &&
                        // NO requirement filter check here
                        (!testCaseFilter || row.test_case_id === testCaseFilter);
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
                
                // Count unique data issues
                const uniqueDataIssues = filteredFlipsData.filter(f => f.has_data_changes).length;
                
                // Test case stats
                const testCasesWithFlips = testCaseFlipData.filter(tc => {{
                    const unitFilter = document.getElementById('globalUnitFilter').value;
                    const stationFilter = document.getElementById('globalStationFilter').value;
                    const saveFilter = document.getElementById('globalSaveFilter').value;
                    return (!unitFilter || tc.unit_id === unitFilter) &&
                        (!stationFilter || tc.station === stationFilter) &&
                        (!saveFilter || tc.save === saveFilter) &&
                        tc.total_bus_flips > 0;
                }}).length;
                
                // Calculate flip percentage
                const flipPercentage = {flip_percentage:.4f};
                
                statsRow.innerHTML = `
                    <div class="stat-card">
                        <div class="stat-value">${{totalFlips.toLocaleString()}}</div>
                        <div class="stat-label">Bus Flips</div>
                    </div>
                    <div class="stat-card warning">
                        <div class="stat-value">${{flipPercentage.toFixed(4)}}%</div>
                        <div class="stat-label">Of Total Messages</div>
                    </div>
                    <div class="stat-card warning">
                        <div class="stat-value">${{uniqueDataIssues.toLocaleString()}}</div>
                        <div class="stat-label">With Data Changes</div>
                    </div>
                    <div class="stat-card info">
                        <div class="stat-value">${{uniqueUnits}}</div>
                        <div class="stat-label">Units Affected</div>
                    </div>
                    <div class="stat-card info">
                        <div class="stat-value">${{uniqueStations}}</div>
                        <div class="stat-label">Stations Affected</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-value">${{uniqueSaves}}</div>
                        <div class="stat-label">Saves Affected</div>
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
                switch(tabName) {{
                    case 'timeline':
                        drawTimelineCharts();
                        break;
                    case 'rates':
                        drawMessageRateCharts();
                        break;
                    case 'risk':  // ADD THIS CASE
                        drawRiskCharts();
                        break;
                    case 'testcases':
                        drawTestCaseCharts();
                        break;
                    case 'requirements':
                        drawRequirementsCharts();
                        break;
                    case 'datawords':
                        drawDataWordCharts();
                        break;
                    default:
                        drawOverviewCharts();
                }}
            }}
            
            function drawAllCharts() {{
                drawOverviewCharts();
                drawRiskCharts();
                drawTimelineCharts();
                drawMessageRateCharts();
                drawTestCaseCharts();
                drawRequirementsCharts();
                drawDataWordCharts();
            }}
            
            function drawOverviewCharts() {{
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
                
                // Message Type Chart - Top 20
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

            function drawRiskCharts() {{
                if (testCaseRiskData.length === 0) {{
                    document.getElementById('riskOverviewChart').innerHTML = '<p>No risk data available</p>';
                    return;
                }}
                
                // Filter risk data by current filters
                const testCaseFilter = document.getElementById('globalTestCaseFilter').value;
                const unitFilter = document.getElementById('globalUnitFilter').value;
                const stationFilter = document.getElementById('globalStationFilter').value;
                const saveFilter = document.getElementById('globalSaveFilter').value;
                
                const filteredRiskData = testCaseRiskData.filter(tc => {{
                    // Test case name filter
                    if (testCaseFilter && tc.test_case !== testCaseFilter) return false;
                    
                    // Location filters - check if ANY of the tested units/stations/saves match
                    if (unitFilter) {{
                        const units = tc.units_tested ? tc.units_tested.split(', ') : [];
                        if (!units.includes(unitFilter)) return false;
                    }}
                    
                    if (stationFilter) {{
                        // Check station_save_combos string
                        if (!tc.station_save_combos || !tc.station_save_combos.includes(stationFilter)) {{
                            return false;
                        }}
                    }}
                    
                    if (saveFilter) {{
                        // Check station_save_combos string
                        if (!tc.station_save_combos || !tc.station_save_combos.includes(saveFilter)) {{
                            return false;
                        }}
                    }}
                    
                    return true;
                }});
                
                // Python already did ALL the work - just display it
                const topRisk = filteredRiskData.slice(0, 20);
                
                // *** MODIFICATION START ***
                // 1. Remove hover from riskOverviewChart
                Plotly.newPlot('riskOverviewChart', [{{
                    x: topRisk.map(d => d.test_case),
                    y: topRisk.map(d => d.total_bus_flips),
                    type: 'bar',
                    marker: {{ 
                        color: topRisk.map(d => 
                            d.total_bus_flips > 100 ? '#e74c3c' :
                            d.total_bus_flips > 50 ? '#f39c12' : '#f0ad4e'
                        )
                    }},
                    hoverinfo: 'none' // <-- This disables hover
                }}], {{
                    margin: {{ t: 10, b: 120, l: 60, r: 20 }},
                    xaxis: {{ title: 'Test Case', tickangle: -45 }},
                    yaxis: {{ title: 'Total Bus Flips (Deduplicated)' }}
                }});
                // *** MODIFICATION END ***
                
                // Station/Save impact - Use pre-computed data
                const stationSaveImpact = {{}};
                
                topRisk.forEach(tc => {{
                    if (tc.station_save_combos) {{
                        // Parse format: "RSFC/L1/4(10), RSFC/L2/3(5)"
                        const combos = tc.station_save_combos.split(', ');
                        stationSaveImpact[tc.test_case] = {{}};
                        
                        combos.forEach(combo => {{
                            const match = combo.match(/(.+?)\((\d+)\)/);
                            if (match) {{
                                const location = match[1];
                                const flips = parseInt(match[2]);
                                stationSaveImpact[tc.test_case][location] = flips;
                            }}
                        }});
                    }}
                }});
                
                // Create stacked bar chart
                if (Object.keys(stationSaveImpact).length > 0) {{
                    const allLocations = new Set();
                    Object.values(stationSaveImpact).forEach(tcData => {{
                        Object.keys(tcData).forEach(loc => allLocations.add(loc));
                    }});
                    
                    const traces = [];
                    const locationArray = Array.from(allLocations).slice(0, 20);
                    
                    locationArray.forEach(location => {{
                        const yValues = Object.keys(stationSaveImpact).map(testCase => {{
                            return stationSaveImpact[testCase][location] || 0;
                        }});
                        
                        // *** MODIFICATION START ***
                        // 2. Clarify hover on station/save chart
                        traces.push({{
                            x: Object.keys(stationSaveImpact),
                            y: yValues,
                            name: location,
                            type: 'bar',
                            hovertemplate: `<b>Location:</b> ${{{location}}}<br><b>Test Case:</b> %{{x}}<br><b>Flips:</b> %{{y}}<extra></extra>`
                        }});
                        // *** MODIFICATION END ***
                    }});
                    
                    Plotly.newPlot('riskStationSaveChart', traces, {{
                        barmode: 'stack',
                        margin: {{ t: 10, b: 120, l: 60, r: 20 }},
                        xaxis: {{ title: 'Test Case', tickangle: -45 }},
                        yaxis: {{ title: 'Bus Flips by Location' }},
                        showlegend: true,
                        legend: {{ x: 1.05, y: 1 }}
                    }});
                }} else {{
                    document.getElementById('riskStationSaveChart').innerHTML = '<p>No station/save impact data available</p>';
                }}
                
                // *** MODIFICATION START ***
                // 3. Requirements chart (using updated Python logic)
                const reqRiskData = topRisk.filter(d => d.requirements_at_risk > 0);
                
                if (reqRiskData.length > 0) {{
                    Plotly.newPlot('riskRequirementsChart', [{{
                        x: reqRiskData.map(d => d.test_case),
                        y: reqRiskData.map(d => d.requirements_at_risk),
                        type: 'bar',
                        marker: {{ color: '#9b59b6' }},
                        text: reqRiskData.map(d => `${{d.total_bus_flips}} flips`),
                        hovertemplate: '<b>Test Case:</b> %{{x}}<br><b>Mapped Requirements:</b> %{{y}}<br><b>Total Flips:</b> %{{text}}<br><b>Reqs:</b> %{{customdata}}<extra></extra>',
                        customdata: reqRiskData.map(d => d.requirement_list)
                    }}], {{
                        margin: {{ t: 10, b: 120, l: 60, r: 20 }},
                        xaxis: {{ title: 'Test Case', tickangle: -45 }},
                        yaxis: {{ title: 'Number of Mapped Requirements' }}
                    }});
                }} else {{
                    document.getElementById('riskRequirementsChart').innerHTML = '<p>No mapped requirements found for risky test cases.</p>';
                }}
                // *** MODIFICATION END ***
                
                // *** MODIFICATION START ***
                // 3. Fix Detail table
                const tableBody = document.getElementById('riskDetailTableBody');
                tableBody.innerHTML = '';
                
                filteredRiskData.forEach(risk => {{
                    const row = tableBody.insertRow();
                    
                    row.insertCell(0).textContent = risk.test_case;
                    row.insertCell(1).textContent = risk.total_bus_flips;
                    row.insertCell(2).textContent = risk.avg_flips_per_run.toFixed(1);
                    row.insertCell(3).textContent = `${{risk.top_station_save_combo}} (${{risk.top_combo_flips}})`;
                    row.insertCell(4).textContent = risk.station_save_combos || 'N/A';
                    row.insertCell(5).textContent = risk.top_msg_types || 'N/A';
                    
                    // Use updated requirement data
                    const reqText = `${{risk.requirements_at_risk}} mapped: ${{risk.requirement_list || 'None'}}`;
                    row.insertCell(6).textContent = reqText;
                    
                    // Fix time_ranges display
                    row.cells[7].innerHTML = risk.time_ranges ? risk.time_ranges.replace(/\\n/g, '<br>') : 'N/A';
                }});
                // *** MODIFICATION END ***
            }}

            function drawTimelineCharts() {{
                if (filteredFlipsData.length === 0) {{
                    document.getElementById('timelineChart').innerHTML = '<p>No data available</p>';
                    return;
                }}
                
                // Timeline chart
                const timestamps = filteredFlipsData.map(d => parseFloat(d.timestamp_busA)).sort((a, b) => a - b);
                
                // Create bins for histogram
                const binSize = 3600; // 1 hour bins
                const minTime = Math.min(...timestamps);
                const maxTime = Math.max(...timestamps);
                const numBins = Math.ceil((maxTime - minTime) / binSize);
                
                const bins = new Array(numBins).fill(0);
                const binMsgTypes = new Array(numBins).fill(null).map(() => new Set());
                const binTestCases = new Array(numBins).fill(null).map(() => new Set());
                const binStationSaves = new Array(numBins).fill(null).map(() => ({{}}));
                
                timestamps.forEach(ts => {{
                    const binIndex = Math.floor((ts - minTime) / binSize);
                    if (binIndex >= 0 && binIndex < numBins) {{
                        bins[binIndex]++;
                    }}
                }});
                
                // Track message types, test cases, and station/saves for each bin
                filteredFlipsData.forEach(flip => {{
                    const ts = parseFloat(flip.timestamp_busA);
                    const binIndex = Math.floor((ts - minTime) / binSize);
                    if (binIndex >= 0 && binIndex < numBins) {{
                        if (flip.msg_type) binMsgTypes[binIndex].add(flip.msg_type);
                        
                        const stationSave = `${{flip.station}}/${{flip.save}}`;
                        if (!binStationSaves[binIndex][stationSave]) {{
                            binStationSaves[binIndex][stationSave] = 0;
                        }}
                        binStationSaves[binIndex][stationSave]++;
                        
                        // Find test cases running at this time
                        filteredTestCaseData.forEach(tc => {{
                            const tcStart = parseFloat(tc.timestamp_start);
                            const tcEnd = parseFloat(tc.timestamp_end);
                            if (ts >= tcStart && ts <= tcEnd) {{
                                binTestCases[binIndex].add(tc.test_case_id);
                            }}
                        }});
                    }}
                }});
                
                const binLabels = bins.map((_, i) => {{
                    const time = minTime + (i * binSize);
                    return new Date(time * 1000).toLocaleString();
                }});
                
                // Calculate statistics for spike detection
                const mean = bins.reduce((a, b) => a + b, 0) / bins.length;
                const stdDev = Math.sqrt(bins.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / bins.length);
                const threshold = mean + (2 * stdDev);
                
                // Color bars based on if they're spikes
                const colors = bins.map(count => count > threshold ? '#e74c3c' : '#3498db');
                
                Plotly.newPlot('timelineChart', [{{
                    x: binLabels,
                    y: bins,
                    type: 'bar',
                    marker: {{ color: colors }},
                    name: 'Bus Flips per Hour',
                    hovertemplate: 'Time: %{{x}}<br>Flips: %{{y}}<extra></extra>'
                }}], {{
                    margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                    xaxis: {{ title: 'Time', tickangle: -45 }},
                    yaxis: {{ title: 'Bus Flips per Hour' }}
                }});

                // Hourly distribution
                const hourlyData = filteredFlipsData.map(d => {{
                    const date = new Date(parseFloat(d.timestamp_busA) * 1000);
                    return date.getHours();
                }});
                
                const hourlyCounts = new Array(24).fill(0);
                hourlyData.forEach(hour => hourlyCounts[hour]++);
                
                Plotly.newPlot('hourlyChart', [{{
                    x: Array.from({{length: 24}}, (_, i) => `${{i.toString().padStart(2, '0')}}:00`),
                    y: hourlyCounts,
                    type: 'bar',
                    marker: {{ color: '#9b59b6' }}
                }}], {{
                    margin: {{ t: 10, b: 60, l: 60, r: 20 }},
                    xaxis: {{ title: 'Hour of Day' }},
                    yaxis: {{ title: 'Total Bus Flips' }}
                }});
                
                // Simplified spike analysis
                const spikes = [];
                bins.forEach((count, i) => {{
                    if (count > threshold) {{
                        // Get top 3 message types for this spike
                        const msgTypeCounts = {{}};
                        filteredFlipsData.forEach(flip => {{
                            const ts = parseFloat(flip.timestamp_busA);
                            const binIndex = Math.floor((ts - minTime) / binSize);
                            if (binIndex === i && flip.msg_type) {{
                                msgTypeCounts[flip.msg_type] = (msgTypeCounts[flip.msg_type] || 0) + 1;
                            }}
                        }});
                        const topMsgTypes = Object.entries(msgTypeCounts)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 3)
                            .map(([mt, count]) => `${{mt}}(${{count}})`)
                            .join(', ');
                        
                        // Get station/save combos with counts
                        const stationSaveList = Object.entries(binStationSaves[i])
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 3)
                            .map(([ss, count]) => `${{ss}}(${{count}})`)
                            .join(', ');
                        
                        const startTime = minTime + (i * binSize);
                        const endTime = startTime + binSize;
                        const startStr = new Date(startTime * 1000).toLocaleString();
                        const endStr = new Date(endTime * 1000).toLocaleString();
                        
                        spikes.push({{
                            timePeriod: `${{startStr}} to ${{endStr}}`,
                            count: count,
                            testCases: Array.from(binTestCases[i]).slice(0, 3).join(', ') || 'None',
                            msgTypes: topMsgTypes || 'None',
                            stationSaves: stationSaveList || 'None'
                        }});
                    }}
                }});
                
                if (spikes.length > 0) {{
                    const topSpikes = spikes.sort((a, b) => b.count - a.count).slice(0, 10);
                    
                    let spikeHtml = `
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Time Period</th>
                                    <th>Bus Flips</th>
                                    <th>Test Cases Running</th>
                                    <th>Top Message Types</th>
                                    <th>Station/Save Combos</th>
                                </tr>
                            </thead>
                            <tbody>`;
                    
                    topSpikes.forEach(spike => {{
                        spikeHtml += `
                            <tr>
                                <td>${{spike.timePeriod}}</td>
                                <td>${{spike.count}}</td>
                                <td>${{spike.testCases}}</td>
                                <td>${{spike.msgTypes}}</td>
                                <td>${{spike.stationSaves}}</td>
                            </tr>`;
                    }});
                    
                    spikeHtml += '</tbody></table>';
                    document.getElementById('spikeAnalysis').innerHTML = spikeHtml;
                }} else {{
                    document.getElementById('spikeAnalysis').innerHTML = '<p>No significant spikes detected</p>';
                }}
            }}
            
            function drawMessageRateCharts() {{
                // Message Rate Statistics - Box plot showing rate distribution per message type
                if (filteredMessageRatesByLocation.length > 0) {{
                    // Collect all rates per message type (keeping location-level granularity)
                    const msgTypeStats = {{}};
                    
                    filteredMessageRatesByLocation.forEach(d => {{
                        if (!msgTypeStats[d.msg_type]) {{
                            msgTypeStats[d.msg_type] = {{
                                rates_hz: [],
                                locations: [],
                                unit_ids: new Set()
                            }};
                        }}
                        
                        // Add rates from this specific location
                        if (d.avg_time_diff_ms > 0) {{
                            const avgRate = 1000 / d.avg_time_diff_ms;
                            msgTypeStats[d.msg_type].rates_hz.push(avgRate);
                            msgTypeStats[d.msg_type].locations.push(`${{d.unit_id}}/${{d.station}}/${{d.save}}`);
                            msgTypeStats[d.msg_type].unit_ids.add(d.unit_id);
                            
                            // Also include min/max rates for better distribution
                            if (d.min_time_diff_ms > 0) {{
                                msgTypeStats[d.msg_type].rates_hz.push(1000 / d.min_time_diff_ms);
                            }}
                            if (d.max_time_diff_ms > 0) {{
                                msgTypeStats[d.msg_type].rates_hz.push(1000 / d.max_time_diff_ms);
                            }}
                        }}
                    }});
                    
                    // Calculate statistics for sorting
                    const msgTypeData = [];
                    Object.entries(msgTypeStats).forEach(([msgType, stats]) => {{
                        if (stats.rates_hz.length > 0) {{
                            const sortedRates = [...stats.rates_hz].sort((a, b) => a - b);
                            const maxRate = Math.max(...stats.rates_hz);
                            const avgRate = stats.rates_hz.reduce((a, b) => a + b, 0) / stats.rates_hz.length;
                            
                            msgTypeData.push({{
                                msgType: msgType,
                                maxRate: maxRate,
                                avgRate: avgRate,
                                medianRate: sortedRates[Math.floor(sortedRates.length / 2)],
                                locationCount: new Set(stats.locations).size,
                                unitCount: stats.unit_ids.size,
                                allRates: stats.rates_hz
                            }});
                        }}
                    }});
                    
                    // Sort by max rate and get top 20
                    const topTypes = msgTypeData.sort((a, b) => b.maxRate - a.maxRate).slice(0, 20);
                    
                    // *** MODIFICATION START ***
                    // 5. Remove mean from box plot
                    // Create box plot traces for each message type
                    const boxData = topTypes.map(type => ({{
                        y: msgTypeStats[type.msgType].rates_hz,
                        type: 'box',
                        name: type.msgType,
                        // boxmean: 'sd', // <-- REMOVED THIS LINE
                        marker: {{
                            color: type.maxRate > 1000 ? '#e74c3c' :
                                type.maxRate > 100 ? '#f39c12' : '#3498db'
                        }},
                        text: `${{type.locationCount}} locations, ${{type.unitCount}} units`,
                        hovertemplate: 
                            '<b>%{{fullData.name}}</b><br>' +
                            'Rate: %{{y:.1f}} msg/s<br>' +
                            '%{{fullData.text}}<br>' +
                            '<extra></extra>'
                    }}));
                    // *** MODIFICATION END ***
                    
                    Plotly.newPlot('messageRateStats', boxData, {{
                        margin: {{ t: 40, b: 100, l: 60, r: 20 }},
                        xaxis: {{ 
                            title: 'Message Type', 
                            tickangle: -45 
                        }},
                        yaxis: {{ 
                            title: 'Sampling Rate (Hz)', 
                            type: 'log' 
                        }},
                        title: {{
                            text: 'Rate Distribution per Message Type (top 20 by max rate)',
                            font: {{ size: 14 }},
                            x: 0.5,
                            xanchor: 'center'
                        }},
                        showlegend: false,
                        hovermode: 'closest'
                    }});
                }} else {{
                    document.getElementById('messageRateStats').innerHTML = 
                        '<p>No message rate data available for current filter</p>';
                }}
                
                // Station Rate Summary - Aggregated view to handle many stations
                if (filteredMessageRatesByLocation.length > 0) {{
                    const stationStats = {{}};
                    
                    // Collect all rates per station
                    filteredMessageRatesByLocation.forEach(d => {{
                        if (!stationStats[d.station]) {{
                            stationStats[d.station] = {{
                                allRates: [],
                                saves: new Set(),
                                units: new Set()
                            }};
                        }}
                        if (d.msg_per_sec) {{
                            stationStats[d.station].allRates.push(d.msg_per_sec);
                            stationStats[d.station].saves.add(d.save);
                            stationStats[d.station].units.add(d.unit_id);
                        }}
                    }});
                    
                    const summaryData = Object.entries(stationStats).map(([station, data]) => {{
                        const rates = data.allRates;
                        return {{
                            station: station,
                            maxRate: Math.max(...rates),
                            avgRate: rates.reduce((a, b) => a + b, 0) / rates.length,
                            minRate: Math.min(...rates),
                            saveCount: data.saves.size,
                            unitCount: data.units.size
                        }};
                    }}).sort((a, b) => b.maxRate - a.maxRate);
                    
                    // Limit to top 30 stations to keep readable
                    const topStations = summaryData.slice(0, 30);
                    
                    Plotly.newPlot('stationRateSummary', [{{
                        x: topStations.map(d => d.station),
                        y: topStations.map(d => d.maxRate),
                        name: 'Max Rate',
                        type: 'bar',
                        marker: {{ 
                            color: topStations.map(d => 
                                d.maxRate > 1000 ? '#e74c3c' :
                                d.maxRate > 100 ? '#f39c12' : '#27ae60'
                            )
                        }},
                        text: topStations.map(d => `${{d.saveCount}} saves, ${{d.unitCount}} units`),
                        hovertemplate: 
                            'Station: %{{x}}<br>' +
                            'Max Rate: %{{y:.1f}} msg/s<br>' +
                            '%{{text}}<br>' +
                            '<extra></extra>'
                    }}], {{
                        margin: {{ t: 40, b: 80, l: 60, r: 20 }},
                        xaxis: {{ 
                            title: `Station (top 30 of ${{summaryData.length}})`,
                            tickangle: -45 
                        }},
                        yaxis: {{ 
                            title: 'Max Messages per Second', 
                            type: 'log' 
                        }},
                        title: {{
                            text: 'Station Max Rates (highest rates across all saves)',
                            font: {{ size: 14 }},
                            x: 0.5,
                            xanchor: 'center'
                        }}
                    }});
                    
                    // High-frequency combinations - FOCUS ON WORST OFFENDERS ONLY
                    const comboData = filteredMessageRatesByLocation.map(d => ({{
                        combo: `${{d.station}}/${{d.save}}`,  // Remove unit_id for clarity
                        msgType: d.msg_type,
                        rate: d.msg_per_sec || 0,
                        unit_id: d.unit_id,
                        station: d.station,
                        save: d.save
                    }})).filter(d => d.rate > 0);
                    
                    // Only show top 15 to keep it readable
                    const topHighFreq = comboData
                        .sort((a, b) => b.rate - a.rate)
                        .slice(0, 15);
                    
                    Plotly.newPlot('stationSaveHighFreq', [{{
                        x: topHighFreq.map((d, i) => `#${{i+1}}: ${{d.combo}}`),  // Add rank numbers
                        y: topHighFreq.map(d => d.rate),
                        type: 'bar',
                        marker: {{
                            color: topHighFreq.map(d => 
                                d.rate > 1000 ? '#e74c3c' : 
                                d.rate > 100 ? '#f39c12' : '#27ae60'
                            )
                        }},
                        text: topHighFreq.map(d => `${{d.msgType}}: ${{d.rate.toFixed(1)}} msg/s`),
                        textposition: 'outside',
                        textangle: -45,
                        hovertemplate: 
                            '<b>%{{x}}</b><br>' +
                            'Msg Type: ' + topHighFreq.map(d => d.msgType) + '<br>' +
                            'Unit: ' + topHighFreq.map(d => d.unit_id) + '<br>' +
                            'Rate: %{{y:.1f}} msg/s<br>' +
                            '<extra></extra>',
                        customdata: topHighFreq.map(d => ({{
                            msgType: d.msgType,
                            unit: d.unit_id
                        }}))
                    }}], {{
                        margin: {{ t: 60, b: 140, l: 60, r: 20 }},
                        xaxis: {{ 
                            title: 'Top 15 Station/Save Combinations', 
                            tickangle: -45 
                        }},
                        yaxis: {{ 
                            title: 'Messages per Second', 
                            type: 'log' 
                        }},
                        title: {{
                            text: 'Highest Rate Configurations (critical attention needed)',
                            font: {{ size: 14, color: '#e74c3c' }},
                            x: 0.5,
                            xanchor: 'center'
                        }}
                    }});
                }}
                
                // Rate metrics - Summary statistics
                if (filteredMessageRatesByLocation.length > 0) {{
                    const allRates = filteredMessageRatesByLocation
                        .filter(d => d.msg_per_sec)
                        .map(d => d.msg_per_sec);
                        
                    if (allRates.length > 0) {{
                        const sortedRates = [...allRates].sort((a, b) => a - b);
                        const criticalCount = allRates.filter(r => r > 1000).length;
                        const warningCount = allRates.filter(r => r > 100 && r <= 1000).length;
                        const normalCount = allRates.filter(r => r <= 100).length;
                        
                        // Count unique locations
                        const uniqueLocations = new Set(
                            filteredMessageRatesByLocation.map(d => 
                                `${{d.unit_id}}/${{d.station}}/${{d.save}}`
                            )
                        ).size;
                        
                        // Count unique stations and saves
                        const uniqueStations = new Set(
                            filteredMessageRatesByLocation.map(d => d.station)
                        ).size;
                        const uniqueSaves = new Set(
                            filteredMessageRatesByLocation.map(d => d.save)
                        ).size;
                        
                        const metricsHtml = `
                            <div class="metric-card">
                                <div class="metric-title">Unique Locations</div>
                                <div class="metric-value">${{uniqueLocations}}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Stations / Saves</div>
                                <div class="metric-value">${{uniqueStations}} / ${{uniqueSaves}}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Critical (>1000 msg/s)</div>
                                <div class="metric-value" style="color: #e74c3c;">${{criticalCount}}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Warning (100-1000 msg/s)</div>
                                <div class="metric-value" style="color: #f39c12;">${{warningCount}}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Normal (<100 msg/s)</div>
                                <div class="metric-value" style="color: #27ae60;">${{normalCount}}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Highest Rate</div>
                                <div class="metric-value">${{Math.max(...allRates).toFixed(1)}} msg/s</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Median Rate</div>
                                <div class="metric-value">${{sortedRates[Math.floor(sortedRates.length/2)].toFixed(1)}} msg/s</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">P95 Rate</div>
                                <div class="metric-value">${{sortedRates[Math.floor(sortedRates.length*0.95)].toFixed(1)}} msg/s</div>
                            </div>
                        `;
                        document.getElementById('rateMetrics').innerHTML = metricsHtml;
                    }}
                }}
            }}
            
            function drawTestCaseCharts() {{
                // *** MODIFICATION START ***
                // 4. Use testCaseRiskData to prevent double counting
                if (testCaseRiskData.length === 0) {{
                    document.getElementById('testCaseFlipCounts').innerHTML = '<p>No test case data available</p>';
                    return;
                }}

                // Filter test case risk data (which is already grouped and deduplicated)
                const filteredTestRisk = testCaseRiskData.filter(tc => {{
                    const unitFilter = document.getElementById('globalUnitFilter').value;
                    const stationFilter = document.getElementById('globalStationFilter').value;
                    const saveFilter = document.getElementById('globalSaveFilter').value;
                    const testCaseFilter = document.getElementById('globalTestCaseFilter').value;
                    
                    if (testCaseFilter && tc.test_case !== testCaseFilter) return false;
                    
                    if (unitFilter) {{
                        const units = tc.units_tested ? tc.units_tested.split(', ') : [];
                        if (!units.includes(unitFilter)) return false;
                    }}
                    
                    if (stationFilter) {{
                        if (!tc.station_save_combos || !tc.station_save_combos.includes(stationFilter)) {{
                            return false;
                        }}
                    }}
                    if (saveFilter) {{
                        if (!tc.station_save_combos || !tc.station_save_combos.includes(saveFilter)) {{
                            return false;
                        }}
                    }}
                    
                    return true;
                }});

                // Data is already sorted by flips in Python, just slice
                const topGroups = filteredTestRisk.slice(0, 20);
                
                // Test cases with most flips - grouped
                Plotly.newPlot('testCaseFlipCounts', [{{
                    x: topGroups.map(d => d.test_case),
                    y: topGroups.map(d => d.total_bus_flips),
                    type: 'bar',
                    marker: {{ color: '#3498db' }},
                    text: topGroups.map(d => `${{d.instances_run}} runs`),
                    hovertemplate: 'Test Case: %{{x}}<br>Total Flips: %{{y}}<br>%{{text}}<extra></extra>'
                }}], {{
                    margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                    xaxis: {{ title: 'Test Case Group', tickangle: -45 }},
                    yaxis: {{ title: 'Total Bus Flips (Deduplicated)' }}
                }});
                
                // Populate test case table with risk data
                const tableBody = document.getElementById('testCaseTableBody');
                tableBody.innerHTML = '';
                
                filteredTestRisk.slice(0, 100).forEach(risk => {{
                    const row = tableBody.insertRow();
                    
                    row.insertCell(0).textContent = risk.test_case;
                    row.insertCell(1).textContent = risk.total_bus_flips;
                    row.insertCell(2).textContent = risk.instances_run;
                    row.insertCell(3).textContent = risk.avg_flips_per_run.toFixed(1);
                    
                    // Parse msg types and locations from strings
                    const msgTypes = risk.top_msg_types ? risk.top_msg_types.split(', ').length : 0;
                    const locations = risk.station_save_combos ? risk.station_save_combos.split(', ').length : 0;
                    
                    row.insertCell(4).textContent = msgTypes;
                    row.insertCell(5).textContent = locations;
                    
                    // Format time range (basic)
                    let timeRange = 'N/A';
                    if (risk.time_ranges) {{
                        const firstLine = risk.time_ranges.split('\\n')[0] || risk.time_ranges.split('\n')[0];
                        if (firstLine) {{
                            timeRange = firstLine.split(': ')[1] || 'See Risk Tab';
                        }}
                    }}
                    row.insertCell(6).textContent = timeRange;
                }});
                // *** MODIFICATION END ***
            }}
            
            function drawRequirementsCharts() {{
                // UNIVERSAL REQUIREMENTS SECTION - Only show those with bus flips
                const universalReqsWithFlips = filteredRequirementsData.filter(r => r.flip_count > 0);
                
                if (universalReqsWithFlips.length > 0) {{
                    // Aggregate by requirement name
                    const reqSummary = {{}};
                    
                    universalReqsWithFlips.forEach(r => {{
                        if (!reqSummary[r.requirement_name]) {{
                            reqSummary[r.requirement_name] = {{
                                flipCount: 0,
                                msgTypes: new Set(),
                                stationSaves: [],
                                locations: []
                            }};
                        }}
                        reqSummary[r.requirement_name].flipCount += r.flip_count;
                        reqSummary[r.requirement_name].msgTypes.add(r.msg_type_affected);
                        
                        const location = `${{r.unit_id}}/${{r.station}}/${{r.save}}`;
                        reqSummary[r.requirement_name].stationSaves.push({{
                            location: `${{r.station}}/${{r.save}}`,
                            msgType: r.msg_type_affected,
                            flips: r.flip_count
                        }});
                    }});
                    
                    const reqData = Object.entries(reqSummary).map(([name, data]) => {{
                        // Create hover text with station/save details
                        const stationSaveInfo = data.stationSaves
                            .sort((a, b) => b.flips - a.flips)
                            .slice(0, 5)
                            .map(ss => `${{ss.location}}: ${{ss.msgType}}(${{ss.flips}})`)
                            .join('<br>');
                        
                        return {{
                            name: name,
                            flips: data.flipCount,
                            msgTypes: Array.from(data.msgTypes).join(', '),
                            hoverText: `${{name}}<br>Total Flips: ${{data.flipCount}}<br><br>Top Locations:<br>${{stationSaveInfo}}`
                        }};
                    }}).sort((a, b) => b.flips - a.flips).slice(0, 20);
                    
                    Plotly.newPlot('universalRequirementsChart', [{{
                        x: reqData.map(d => d.name),
                        y: reqData.map(d => d.flips),
                        type: 'bar',
                        marker: {{ color: '#e74c3c' }},
                        text: reqData.map(d => d.msgTypes),
                        hovertemplate: '%{{text}}<extra></extra>',
                        customdata: reqData.map(d => d.hoverText),
                        hovertext: reqData.map(d => d.hoverText)
                    }}], {{
                        margin: {{ t: 10, b: 120, l: 60, r: 20 }},
                        xaxis: {{ title: 'Universal Requirement Name', tickangle: -45 }},
                        yaxis: {{ title: 'Total Bus Flips' }},
                        hovermode: 'closest'
                    }});
                }} else {{
                    document.getElementById('universalRequirementsChart').innerHTML = '<p>No universal requirements with bus flips</p>';
                }}
                
                // TEST CASE SPECIFIC REQUIREMENTS - Simplified metrics
                // 6. Grouping is already done in Python, this code is correct
                if (filteredTestCaseRequirements.length > 0) {{
                    // Group by requirement and sum UNIQUE flips only
                    const tcReqSummary = {{}};
                    
                    filteredTestCaseRequirements.forEach(r => {{
                        if (!tcReqSummary[r.requirement_name]) {{
                            tcReqSummary[r.requirement_name] = {{
                                totalFlips: 0,  // Will use max unique_flip_count
                                testCases: new Set(),
                                stationSaves: new Set()
                            }};
                        }}
                        
                        // Use the unique flip count (max across all instances)
                        const uniqueFlips = r.unique_flip_count || r.flip_count || 0;
                        if (uniqueFlips > tcReqSummary[r.requirement_name].totalFlips) {{
                            tcReqSummary[r.requirement_name].totalFlips = uniqueFlips;
                        }}
                        
                        tcReqSummary[r.requirement_name].testCases.add(r.test_case_id);
                        tcReqSummary[r.requirement_name].stationSaves.add(`${{r.station}}/${{r.save}}`);
                    }});
                    
                    const tcReqData = Object.entries(tcReqSummary).map(([name, data]) => ({{
                        name: name,
                        totalFlips: data.totalFlips,
                        testCases: data.testCases.size,
                        stationSaves: data.stationSaves.size
                    }})).sort((a, b) => b.totalFlips - a.totalFlips).slice(0, 20);
                    
                    Plotly.newPlot('testCaseRequirementsChart', [{{
                        x: tcReqData.map(d => d.name),
                        y: tcReqData.map(d => d.totalFlips),
                        type: 'bar',
                        marker: {{ color: '#e74c3c' }},
                        text: tcReqData.map(d => `${{d.testCases}} test cases, ${{d.stationSaves}} locations`),
                        hovertemplate: '%{{x}}<br>Total Bus Flips: %{{y}}<br>%{{text}}<extra></extra>'
                    }}], {{
                        margin: {{ t: 10, b: 120, l: 60, r: 20 }},
                        xaxis: {{ title: 'Test Case Requirement', tickangle: -45 }},
                        yaxis: {{ title: 'Total Bus Flips Across All Test Runs' }}
                    }});
                    
                    // Update failures table with simplified columns
                    const failuresTableBody = document.getElementById('testCaseReqFailuresTableBody');
                    failuresTableBody.innerHTML = '';
                    
                    // Group by requirement to avoid duplicates
                    const groupedFailures = {{}};
                    filteredTestCaseRequirements.forEach(r => {{
                        const key = `${{r.requirement_name}}-${{r.test_case_id}}`;
                        if (!groupedFailures[key]) {{
                            groupedFailures[key] = {{
                                requirement_name: r.requirement_name,
                                test_case_id: r.test_case_id,
                                locations: new Set(),
                                msg_types: new Set(),
                                totalFlips: r.unique_flip_count || 0,
                                stationSaves: new Set()
                            }};
                        }}
                        groupedFailures[key].locations.add(`${{r.unit_id}}/${{r.station}}/${{r.save}}`);
                        if (r.msg_types_tested) {{
                            r.msg_types_tested.split(', ').forEach(mt => groupedFailures[key].msg_types.add(mt));
                        }}
                        groupedFailures[key].stationSaves.add(`${{r.station}}/${{r.save}}`);
                    }});
                    
                    Object.values(groupedFailures).slice(0, 100).forEach(r => {{
                        const row = failuresTableBody.insertRow();
                        row.insertCell(0).textContent = r.requirement_name;
                        row.insertCell(1).textContent = r.test_case_id; // This is the grouped ID
                        row.insertCell(2).textContent = Array.from(r.locations).join(', ');
                        row.insertCell(3).textContent = Array.from(r.msg_types).join(', ');
                        row.insertCell(4).textContent = r.totalFlips;
                        row.insertCell(5).textContent = Array.from(r.stationSaves).join(', ');
                    }});
                }} else {{
                    document.getElementById('testCaseRequirementsChart').innerHTML = '<p>No test case requirement failures with bus flips</p>';
                    document.getElementById('testCaseReqFailuresTableBody').innerHTML = '';
                }}
                
                // REQUIREMENT-TEST CASE MAPPING
                // 6. This also uses the grouped ID from Python, so it's correct.
                if (requirementMappingData.length > 0) {{
                    const mappingTableBody = document.getElementById('requirementMappingTableBody');
                    mappingTableBody.innerHTML = '';
                    
                    requirementMappingData.slice(0, 100).forEach(r => {{
                        const row = mappingTableBody.insertRow();
                        row.insertCell(0).textContent = r.requirement_name || '';
                        row.insertCell(1).textContent = r.test_cases || ''; // This is a list of grouped IDs
                        row.insertCell(2).textContent = r.total_failed || 0;
                        row.insertCell(3).textContent = r.pass_rate ? r.pass_rate.toFixed(1) + '%' : '0%';
                        row.insertCell(4).textContent = `${{r.unique_units}}U/${{r.unique_stations}}S/${{r.unique_saves}}Sv`;
                    }});
                }} else {{
                    document.getElementById('requirementMappingTableBody').innerHTML = '';
                }}
            }}
            
            function drawDataWordCharts() {{
                if (filteredDataWordData.length === 0) {{
                    document.getElementById('singleVsMultiChart').innerHTML = '<p>No data word analysis available</p>';
                    return;
                }}
                
                // Single vs Multi-Word Change Distribution
                const singleMultiData = {{}};
                filteredDataWordData.forEach(d => {{
                    if (!singleMultiData[d.msg_type]) {{
                        singleMultiData[d.msg_type] = {{
                            single: 0,
                            multi: 0,
                            total: 0
                        }};
                    }}
                    singleMultiData[d.msg_type].single += d.single_word_changes || 0;
                    singleMultiData[d.msg_type].multi += d.multi_word_changes || 0;
                    singleMultiData[d.msg_type].total += d.total_issues || 0;
                }});
                
                const msgTypes = Object.keys(singleMultiData).sort((a, b) => 
                    singleMultiData[b].total - singleMultiData[a].total
                ).slice(0, 30);
                
                Plotly.newPlot('singleVsMultiChart', [{{
                    x: msgTypes,
                    y: msgTypes.map(mt => singleMultiData[mt].single),
                    name: 'Single Word Changes',
                    type: 'bar',
                    marker: {{ color: '#3498db' }}
                }}, {{
                    x: msgTypes,
                    y: msgTypes.map(mt => singleMultiData[mt].multi),
                    name: 'Multi Word Changes',
                    type: 'bar',
                    marker: {{ color: '#e74c3c' }}
                }}], {{
                    barmode: 'stack',
                    margin: {{ t: 10, b: 100, l: 60, r: 20 }},
                    xaxis: {{ title: 'Message Type', tickangle: -45 }},
                    yaxis: {{ title: 'Number of Changes' }},
                    hovermode: 'x unified'
                }});
                
                // Multi-Word Change Patterns - Heatmap
                const multiWordPatterns = {{}};
                filteredDataWordData.forEach(d => {{
                    if (d.multi_word_changes > 0 && d.data_word) {{
                        if (!multiWordPatterns[d.msg_type]) {{
                            multiWordPatterns[d.msg_type] = {{}};
                        }}
                        if (!multiWordPatterns[d.msg_type][d.data_word]) {{
                            multiWordPatterns[d.msg_type][d.data_word] = 0;
                        }}
                        multiWordPatterns[d.msg_type][d.data_word] += d.multi_word_changes;
                    }}
                }});
                // Collect all unique data words
                const allDataWords = new Set();
                filteredDataWordData.forEach(d => {{
                    if (d.data_word) {{
                        allDataWords.add(d.data_word);
                    }}
                }});
                // Create heatmap data
                const heatmapMsgTypes = Object.keys(multiWordPatterns)
                    .sort((a, b) => {{
                        const sumA = Object.values(multiWordPatterns[a]).reduce((s, v) => s + v, 0);
                        const sumB = Object.values(multiWordPatterns[b]).reduce((s, v) => s + v, 0);
                        return sumB - sumA;
                    }})
                    .slice(0, 25);
                const heatmapDataWords = Array.from(allDataWords)
                    .sort((a, b) => {{
                        let sumA = 0, sumB = 0;
                        Object.values(multiWordPatterns).forEach(dw => {{
                            sumA += dw[a] || 0;
                            sumB += dw[b] || 0;
                        }});
                        return sumB - sumA;
                    }})
                    .slice(0, 30);
                
                const zValues = [];
                heatmapMsgTypes.forEach(mt => {{
                    const row = [];
                    heatmapDataWords.forEach(dw => {{
                        row.push(multiWordPatterns[mt]?.[dw] || 0);
                    }});
                    zValues.push(row);
                }});
                
                if (zValues.length > 0 && zValues[0].length > 0) {{
                    Plotly.newPlot('multiWordPatternChart', [{{
                        z: zValues,
                        x: heatmapDataWords,
                        y: heatmapMsgTypes,
                        type: 'heatmap',
                        colorscale: 'Viridis',
                        hovertemplate: 'Msg Type: %{{y}}<br>Data Word: %{{x}}<br>Multi Changes: %{{z}}<extra></extra>'
                    }}], {{
                        margin: {{ t: 10, b: 100, l: 100, r: 40 }},
                        xaxis: {{ title: 'Data Word', tickangle: -45 }},
                        yaxis: {{ title: 'Message Type' }}
                    }});
                }} else {{
                    document.getElementById('multiWordPatternChart').innerHTML = '<p>No multi-word patterns found</p>';
                }}
                
                // Enhanced Error Pattern Analysis
                const patternCounts = {{}};
                const patternTypes = {{}};
                
                filteredDataWordData.forEach(d => {{
                    if (d.most_common_error && d.most_common_error !== 'N/A') {{
                        const pattern = `${{d.msg_type}}: ${{d.most_common_error}}`;
                        if (!patternCounts[pattern]) {{
                            patternCounts[pattern] = 0;
                            patternTypes[pattern] = {{
                                single: 0,
                                multi: 0
                            }};
                        }}
                        patternCounts[pattern] += d.most_common_count || d.total_issues;
                        
                        // Track if this pattern occurs as single or multi-word change
                        if (d.single_word_changes > d.multi_word_changes) {{
                            patternTypes[pattern].single += d.single_word_changes;
                        }} else {{
                            patternTypes[pattern].multi += d.multi_word_changes;
                        }}
                    }}
                }});
                
                const topPatterns = Object.entries(patternCounts)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 30);
                
                // Create enhanced bar chart with single/multi indicators
                const patternData = topPatterns.map(([pattern, count]) => {{
                    const types = patternTypes[pattern];
                    const changeType = types.single > types.multi ? 'Single' : 'Multi';
                    const percentage = types.single > types.multi ? 
                        Math.round((types.single / (types.single + types.multi)) * 100) :
                        Math.round((types.multi / (types.single + types.multi)) * 100);
                    return {{
                        pattern: pattern,
                        count: count,
                        changeType: changeType,
                        percentage: percentage
                    }};
                }});
                
                Plotly.newPlot('errorPatternChart', [{{
                    x: patternData.map(p => p.count),
                    y: patternData.map(p => p.pattern),
                    type: 'bar',
                    orientation: 'h',
                    marker: {{
                        color: patternData.map(p => p.changeType === 'Single' ? '#3498db' : '#e74c3c')
                    }},
                    text: patternData.map(p => `${{p.changeType}} (${{p.percentage}}%)`),
                    hovertemplate: '%{{y}}<br>Frequency: %{{x}}<br>Type: %{{text}}<extra></extra>'
                }}], {{
                    margin: {{ t: 40, b: 60, l: 200, r: 60 }},
                    xaxis: {{ title: 'Frequency' }},
                    yaxis: {{ title: 'Error Pattern' }},
                    height: 800,
                    title: {{ 
                        text: 'Blue = Single-word, Red = Multi-word',
                        x: 0.5,
                        y: 0.99,
                        xanchor: 'center',
                        font: {{ size: 12, color: '#7f8c8d' }}
                    }}
                }});
                
                // Change Speed Analysis
                const speedData = {{
                    single: [],
                    multi: []
                }};
                
                filteredDataWordData.forEach(d => {{
                    if (d.avg_flip_speed_ms && d.avg_flip_speed_ms > 0) {{
                        if (d.single_word_changes > d.multi_word_changes) {{
                            speedData.single.push(d.avg_flip_speed_ms);
                        }} else if (d.multi_word_changes > 0) {{
                            speedData.multi.push(d.avg_flip_speed_ms);
                        }}
                    }}
                }});
                
                const traces = [];
                if (speedData.single.length > 0) {{
                    traces.push({{
                        y: speedData.single,
                        type: 'box',
                        name: 'Single Word Changes',
                        marker: {{ color: '#3498db' }},
                        boxmean: true
                    }});
                }}
                if (speedData.multi.length > 0) {{
                    traces.push({{
                        y: speedData.multi,
                        type: 'box',
                        name: 'Multi Word Changes',
                        marker: {{ color: '#e74c3c' }},
                        boxmean: true
                    }});
                }}
                
                if (traces.length > 0) {{
                    Plotly.newPlot('changeSpeedChart', traces, {{
                        margin: {{ t: 10, b: 60, l: 60, r: 20 }},
                        yaxis: {{ title: 'Flip Speed (ms)', type: 'log' }},
                        showlegend: true
                    }});
                }} else {{
                    document.getElementById('changeSpeedChart').innerHTML = '<p>No speed data available</p>';
                }}
                
                // Calculate metrics
                const totalIssues = filteredDataWordData.reduce((sum, d) => sum + d.total_issues, 0);
                const uniqueDataWords = new Set(filteredDataWordData.map(d => d.data_word)).size;
                const uniqueMsgTypes = new Set(filteredDataWordData.map(d => d.msg_type)).size;
                
                let singleWordIssues = 0;
                let multiWordIssues = 0;
                let fastestFlip = Infinity;
                let avgFlipSpeed = 0;
                let speedCount = 0;
                
                filteredDataWordData.forEach(d => {{
                    singleWordIssues += d.single_word_changes || 0;
                    multiWordIssues += d.multi_word_changes || 0;
                    if (d.avg_flip_speed_ms && d.avg_flip_speed_ms > 0) {{
                        fastestFlip = Math.min(fastestFlip, d.avg_flip_speed_ms);
                        avgFlipSpeed += d.avg_flip_speed_ms;
                        speedCount++;
                    }}
                }});
                
                avgFlipSpeed = speedCount > 0 ? (avgFlipSpeed / speedCount) : 0;
                const multiWordPercent = (singleWordIssues + multiWordIssues) > 0 ? 
                    ((multiWordIssues / (singleWordIssues + multiWordIssues)) * 100) : 0;
                
                const metricsHtml = `
                    <div class="metric-card">
                        <div class="metric-title">Total Data Word Issues</div>
                        <div class="metric-value">${{totalIssues.toLocaleString()}}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Single Word Changes</div>
                        <div class="metric-value" style="color: #3498db;">${{singleWordIssues.toLocaleString()}}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Multi Word Changes</div>
                        <div class="metric-value" style="color: #e74c3c;">${{multiWordIssues.toLocaleString()}}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Multi-Word Percentage</div>
                        <div class="metric-value">${{multiWordPercent.toFixed(1)}}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Unique Data Words</div>
                        <div class="metric-value">${{uniqueDataWords}}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-title">Affected Message Types</div>
                        <div class="metric-value">${{uniqueMsgTypes}}</div>
                    </div>
                `;
                document.getElementById('dataWordMetrics').innerHTML = metricsHtml;
                
                // Populate interactive pivot table
                const pivotData = {{}};
                filteredDataWordData.forEach(d => {{
                    if (!pivotData[d.msg_type]) {{
                        pivotData[d.msg_type] = {{
                            total: 0,
                            single: 0,
                            multi: 0,
                            patterns: new Set(),
                            topPattern: '',
                            topPatternCount: 0,
                            wordCounts: []
                        }};
                    }}
                    pivotData[d.msg_type].total += d.total_issues || 0;
                    pivotData[d.msg_type].single += d.single_word_changes || 0;
                    pivotData[d.msg_type].multi += d.multi_word_changes || 0;
                    
                    if (d.most_common_error && d.most_common_error !== 'N/A') {{
                        pivotData[d.msg_type].patterns.add(d.most_common_error);
                        if ((d.most_common_count || 0) > pivotData[d.msg_type].topPatternCount) {{
                            pivotData[d.msg_type].topPattern = d.most_common_error;
                            pivotData[d.msg_type].topPatternCount = d.most_common_count || 0;
                        }}
                    }}
                    
                    if (d.num_data_changes) {{
                        pivotData[d.msg_type].wordCounts.push(d.num_data_changes);
                    }}
                }});
                
                const pivotBody = document.getElementById('dataWordPivotBody');
                pivotBody.innerHTML = '';
                
                Object.entries(pivotData)
                    .sort((a, b) => b[1].total - a[1].total)
                    .forEach(([msgType, data]) => {{
                        const row = pivotBody.insertRow();
                        const multiPercent = data.total > 0 ? (data.multi / data.total * 100).toFixed(1) : '0.0';
                        const avgWords = data.wordCounts.length > 0 ? 
                            (data.wordCounts.reduce((a, b) => a + b, 0) / data.wordCounts.length).toFixed(2) : 'N/A';
                        
                        row.insertCell(0).textContent = msgType;
                        row.insertCell(1).textContent = data.total;
                        row.insertCell(2).textContent = data.single;
                        row.insertCell(3).textContent = data.multi;
                        row.insertCell(4).textContent = multiPercent + '%';
                        row.insertCell(5).textContent = avgWords;
                        row.insertCell(6).textContent = data.patterns.size;
                        row.insertCell(7).textContent = data.topPattern || 'N/A';
                    }});
                
                // Populate data word table
                const tableBody = document.getElementById('dataWordTableBody');
                tableBody.innerHTML = '';
                
                filteredDataWordData.slice(0, 50).forEach(d => {{
                    const row = tableBody.insertRow();
                    const singleMulti = (d.single_word_changes || 0) > (d.multi_word_changes || 0) ? 'Single' : 'Multi';
                    const locations = `${{d.affected_units || 0}}U/${{d.affected_stations || 0}}S/${{d.affected_saves || 0}}S`;
                    
                    row.insertCell(0).textContent = d.msg_type || '';
                    row.insertCell(1).textContent = d.data_word || '';
                    row.insertCell(2).textContent = d.total_issues || 0;
                    row.insertCell(3).textContent = singleMulti;
                    row.insertCell(4).textContent = d.most_common_error || 'N/A';
                    row.insertCell(5).textContent = d.avg_flip_speed_ms ? d.avg_flip_speed_ms.toFixed(3) : 'N/A';
                    row.insertCell(6).textContent = locations;
                }});
            }}
            
            function sortPivotTable(column) {{
                const table = document.getElementById('dataWordPivotTable');
                const tbody = table.querySelector('tbody');
                const rows = Array.from(tbody.querySelectorAll('tr'));
                
                rows.sort((a, b) => {{
                    const aVal = a.cells[column].textContent;
                    const bVal = b.cells[column].textContent;
                    
                    // Handle percentages
                    if (aVal.includes('%') && bVal.includes('%')) {{
                        const aNum = parseFloat(aVal.replace('%', ''));
                        const bNum = parseFloat(bVal.replace('%', ''));
                        return bNum - aNum;
                    }}
                    
                    // Try to parse as number
                    const aNum = parseFloat(aVal);
                    const bNum = parseFloat(bVal);
                    
                    if (!isNaN(aNum) && !isNaN(bNum)) {{
                        return bNum - aNum;
                    }}
                    
                    return aVal.localeCompare(bVal);
                }});
                
                rows.forEach(row => tbody.appendChild(row));
            }}
            
            function sortTable(tableId, column) {{
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
