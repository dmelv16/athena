// Replace the updateAllCharts function with this updated version:
function updateAllCharts() {
    const unitFilter = document.getElementById('globalUnitFilter').value;
    const stationFilter = document.getElementById('globalStationFilter').value;
    const saveFilter = document.getElementById('globalSaveFilter').value;
    const msgTypeFilter = document.getElementById('globalMsgTypeFilter').value;
    
    // Filter main data
    filteredData = allData.filter(row => {
        return (!unitFilter || row.unit_id === unitFilter) &&
               (!stationFilter || row.station === stationFilter) &&
               (!saveFilter || row.save === saveFilter) &&
               (!msgTypeFilter || row.msg_type === msgTypeFilter);
    });
    
    // Filter data word data based on ALL filters including unit, station, save
    filteredDataWordData = dataWordData.filter(row => {
        return (!unitFilter || row.unit_id === unitFilter) &&
               (!stationFilter || row.station === stationFilter) &&
               (!saveFilter || row.save === saveFilter) &&
               (!msgTypeFilter || row.msg_type === msgTypeFilter);
    });
    
    // Update filtered count
    document.getElementById('filteredCount').textContent = 
        `Showing ${filteredData.length} of ${allData.length} flips`;
    
    // Redraw all charts
    drawCharts();
}

// Replace the createTimingAnalysisTable function with this updated version:
function createTimingAnalysisTable() {
    if (filteredData.length === 0) {
        document.getElementById('timingTableBody').innerHTML = 
            '<tr><td colspan="8">No data matching filters</td></tr>';
        return;
    }
    
    // Group timing data by station, msg_type, save
    const timingGroups = {};
    filteredData.forEach(d => {
        const key = `${d.station}|${d.msg_type || 'unknown'}|${d.save}`;
        if (!timingGroups[key]) {
            timingGroups[key] = {
                station: d.station,
                msg_type: d.msg_type || 'unknown',
                save: d.save,
                timings: []
            };
        }
        const ms = d.timestamp_diff_ms || (d.timestamp_diff * 1000);
        if (ms && ms < 100) {
            timingGroups[key].timings.push(ms);
        }
    });
    
    // Calculate statistics for each group
    const timingStats = [];
    Object.values(timingGroups).forEach(group => {
        if (group.timings.length > 0) {
            const sorted = group.timings.sort((a, b) => a - b);
            const sum = sorted.reduce((a, b) => a + b, 0);
            const avg = sum / sorted.length;
            const min = sorted[0];
            const max = sorted[sorted.length - 1];
            
            // Calculate standard deviation
            const squaredDiffs = sorted.map(v => Math.pow(v - avg, 2));
            const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / sorted.length;
            const stdDev = Math.sqrt(avgSquaredDiff);
            
            timingStats.push({
                station: group.station,
                msg_type: group.msg_type,
                save: group.save,
                count: group.timings.length,
                avg: avg,
                min: min,
                max: max,
                std: stdDev
            });
        }
    });
    
    // Apply sorting based on current sort column and direction
    timingStats.sort((a, b) => {
        let aVal = a[timingSortColumn];
        let bVal = b[timingSortColumn];
        
        // Handle string comparisons
        if (typeof aVal === 'string') {
            aVal = aVal.toLowerCase();
            bVal = bVal.toLowerCase();
        }
        
        if (timingSortAsc) {
            return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
        } else {
            return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
        }
    });
    
    // Create table rows
    let tableHtml = '';
    timingStats.forEach(stat => {
        tableHtml += `
            <tr>
                <td>${stat.station}</td>
                <td>${stat.msg_type}</td>
                <td>${stat.save}</td>
                <td>${stat.count}</td>
                <td>${stat.avg.toFixed(3)}</td>
                <td>${stat.min.toFixed(3)}</td>
                <td>${stat.max.toFixed(3)}</td>
                <td>${stat.std.toFixed(3)}</td>
            </tr>
        `;
    });
    
    document.getElementById('timingTableBody').innerHTML = tableHtml;
    
    // Update overall timing stats
    updateTimingSummaryStats(filteredData);
}

// Add/update the sortTimingTable function (make sure it's not nested inside another function):
let timingSortColumn = 'count';
let timingSortAsc = false;

function sortTimingTable(column) {
    if (column === timingSortColumn) {
        timingSortAsc = !timingSortAsc;
    } else {
        timingSortColumn = column;
        timingSortAsc = column === 'station' || column === 'msg_type' || column === 'save'; // String columns default to ascending
    }
    
    createTimingAnalysisTable();
}

// Update the showCellDetails function to show unit/station/save info if available:
function showCellDetails(msgType, dataWord) {
    const items = filteredDataWordData.filter(d => d.msg_type === msgType && d.data_word === dataWord);
    if (items.length === 0) return;
    
    let detailsText = `${msgType} - ${dataWord}\n\n`;
    
    if (items.length === 1) {
        const item = items[0];
        detailsText += `Unit: ${item.unit_id || 'All'}\n`;
        detailsText += `Station: ${item.station || 'All'}\n`;
        detailsText += `Save: ${item.save || 'All'}\n\n`;
        detailsText += `Total Issues: ${item.total_issues}\n`;
        detailsText += `Single Word Changes: ${item.single_word_changes || 0}\n`;
        detailsText += `Multi Word Changes: ${item.multi_word_changes || 0}\n\n`;
        detailsText += `Top Patterns:\n${item.top_error_patterns || 'N/A'}`;
    } else {
        // Multiple items, show summary
        const totalIssues = items.reduce((sum, item) => sum + item.total_issues, 0);
        detailsText += `Multiple locations found (${items.length} combinations)\n\n`;
        detailsText += `Total Issues Across All: ${totalIssues}\n\n`;
        detailsText += `Breakdown by Location:\n`;
        items.forEach(item => {
            detailsText += `\n${item.unit_id}/${item.station}/${item.save}: ${item.total_issues} issues`;
        });
    }
    
    alert(detailsText);
}

def analyze_data_word_patterns(self):
    """Analyze data word error patterns and create comprehensive summaries"""
    if not self.data_word_issues:
        return
    
    # Create DataFrame from data word issues
    df_issues = pd.DataFrame(self.data_word_issues)
    
    # Analysis 1: Detailed data word analysis including unit/station/save
    grouped = df_issues.groupby(['unit_id', 'station', 'save', 'msg_type', 'data_word'])
    
    analysis_results = []
    for (unit_id, station, save, msg_type, data_word), group in grouped:
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
            'unit_id': unit_id,
            'station': station,
            'save': save,
            'msg_type': msg_type,
            'data_word': data_word,
            'total_issues': len(group),
            'issue_percentage': round(issue_percentage, 2),
            'unique_patterns': len(pattern_counts),
            'top_error_patterns': ' | '.join(top_patterns),
            'most_common_error': pattern_counts.index[0] if len(pattern_counts) > 0 else 'N/A',
            'most_common_count': pattern_counts.iloc[0] if len(pattern_counts) > 0 else 0,
            'multi_word_changes': multi_word_count,
            'single_word_changes': single_word_count,
            'avg_flip_speed_ms': round(avg_flip_speed, 3)
        })
    
    self.df_data_word_analysis = pd.DataFrame(analysis_results)
    if not self.df_data_word_analysis.empty:
        self.df_data_word_analysis = self.df_data_word_analysis.sort_values('total_issues', ascending=False)
    
    # Analysis 2: Multi-word change analysis by message type
    if 'num_data_changes' in df_issues.columns:
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
    
    # Create pattern frequency table with location details
    pattern_results = []
    for (unit_id, station, save, msg_type, data_word), group in grouped:
        for _, row in group.iterrows():
            pattern = f"{row['value_before']} -> {row['value_after']}"
            pattern_results.append({
                'unit_id': unit_id,
                'station': station,
                'save': save,
                'msg_type': msg_type,
                'data_word': data_word,
                'error_pattern': pattern,
                'num_data_changes': row.get('num_data_changes', 1)
            })
    
    if pattern_results:
        df_patterns = pd.DataFrame(pattern_results)
        pattern_freq = df_patterns.groupby(['unit_id', 'station', 'save', 'msg_type', 'data_word', 'error_pattern']).agg({
            'num_data_changes': ['count', 'mean']
        }).reset_index()
        pattern_freq.columns = ['unit_id', 'station', 'save', 'msg_type', 'data_word', 'error_pattern', 'frequency', 'avg_words_in_flip']
        self.df_data_word_patterns = pattern_freq.sort_values(['msg_type', 'data_word', 'frequency'], 
                                                             ascending=[True, True, False])
