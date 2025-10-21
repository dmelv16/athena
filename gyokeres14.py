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
