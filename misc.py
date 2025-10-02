// Replace createDataWordPivotTable with this corrected version that properly sums issues:

function createDataWordPivotTable() {{
    if (filteredDataWordData.length === 0) {{
        document.getElementById('pivotTableBody').innerHTML = 
            '<tr><td colspan="100%">No data matching filters</td></tr>';
        return;
    }}
    
    // Create pivot data structure - SUM the total_issues, don't count rows
    const pivotData = {{}};
    const dataWords = new Set();
    const msgTypes = new Set();
    let totalEvents = 0;  // Sum of all individual issues
    
    filteredDataWordData.forEach(d => {{
        if (!pivotData[d.msg_type]) {{
            pivotData[d.msg_type] = {{}};
        }}
        
        dataWords.add(d.data_word);
        msgTypes.add(d.msg_type);
        
        // Get the count of issues for this row
        const issueCount = d.total_issues || 0;
        
        if (pivotData[d.msg_type][d.data_word]) {{
            // ADD the issues to existing count - this is the key fix!
            pivotData[d.msg_type][d.data_word].count += issueCount;
            pivotData[d.msg_type][d.data_word].occurrences += 1;  // Track number of location combinations
            pivotData[d.msg_type][d.data_word].single += (d.single_word_changes || 0);
            pivotData[d.msg_type][d.data_word].multi += (d.multi_word_changes || 0);
            
            // Add location info
            pivotData[d.msg_type][d.data_word].locations.push({{
                unit: d.unit_id,
                station: d.station,
                save: d.save,
                count: issueCount
            }});
            
            // Append patterns if new ones exist
            if (d.top_error_patterns && !pivotData[d.msg_type][d.data_word].patterns.includes(d.top_error_patterns)) {{
                if (pivotData[d.msg_type][d.data_word].patterns) {{
                    pivotData[d.msg_type][d.data_word].patterns += ' | ' + d.top_error_patterns;
                }} else {{
                    pivotData[d.msg_type][d.data_word].patterns = d.top_error_patterns;
                }}
            }}
        }} else {{
            // First occurrence of this msg_type/data_word combination
            pivotData[d.msg_type][d.data_word] = {{
                count: issueCount,  // Start with this row's issue count
                occurrences: 1,  // First location combination
                single: d.single_word_changes || 0,
                multi: d.multi_word_changes || 0,
                patterns: d.top_error_patterns || '',
                locations: [{{
                    unit: d.unit_id,
                    station: d.station,
                    save: d.save,
                    count: issueCount
                }}]
            }};
        }}
        
        totalEvents += issueCount;
    }});
    
    const sortedDataWords = Array.from(dataWords).sort();
    const sortedMsgTypes = Array.from(msgTypes).sort();
    
    // Create header
    const headerHtml = `
        <tr>
            <th class="row-header">Msg Type \\ Data Word</th>
            ${{sortedDataWords.map(dw => `<th>${{dw}}</th>`).join('')}}
            <th style="background: #ff9800;">Row Total</th>
        </tr>
    `;
    document.getElementById('pivotTableHeader').innerHTML = headerHtml;
    
    // Create body
    let bodyHtml = '';
    const columnTotals = {{}};
    let nonEmptyCells = 0;
    
    sortedMsgTypes.forEach(msgType => {{
        let rowTotal = 0;
        let rowHtml = `<td class="row-header">${{msgType}}</td>`;
        
        sortedDataWords.forEach(dataWord => {{
            const cellData = pivotData[msgType] && pivotData[msgType][dataWord];
            const count = cellData ? cellData.count : 0;  // This should now be the sum of issues
            rowTotal += count;
            columnTotals[dataWord] = (columnTotals[dataWord] || 0) + count;
            
            if (count > 0) nonEmptyCells++;
            
            // Color coding based on total issues
            const bgColor = count === 0 ? '#fff' : 
                          count === 1 ? '#e8f5e9' :
                          count < 10 ? '#c8e6c9' :
                          count < 50 ? '#ffeb3b' :
                          count < 100 ? '#ff9800' : 
                          count < 500 ? '#ff5722' : '#d32f2f';
            const textColor = (count > 50 || (bgColor === '#d32f2f')) ? 'white' : 'black';
            
            // Tooltip shows location breakdown
            let tooltip = '';
            if (cellData) {{
                if (cellData.occurrences > 1) {{
                    tooltip = `${{cellData.occurrences}} locations, ${{count}} total issues`;
                }} else {{
                    tooltip = `${{count}} issues from ${{cellData.locations[0].unit}}/${{cellData.locations[0].station}}/${{cellData.locations[0].save}}`;
                }}
                tooltip += `\\nSingle: ${{cellData.single}}, Multi: ${{cellData.multi}}`;
            }}
            
            rowHtml += `<td style="background: ${{bgColor}}; color: ${{textColor}};" 
                           onclick="showCellDetails('${{msgType}}', '${{dataWord}}')"
                           title="${{tooltip}}">${{count || '-'}}</td>`;
        }});
        
        rowHtml += `<td style="background: #fff3e0; font-weight: bold;">${{rowTotal}}</td>`;
        bodyHtml += `<tr>${{rowHtml}}</tr>`;
    }});
    
    // Add totals row
    let totalRow = '<tr><td class="row-header" style="background: #ff9800; color: white;">Col TOTAL</td>';
    let grandTotal = 0;
    sortedDataWords.forEach(dw => {{
        const total = columnTotals[dw] || 0;
        grandTotal += total;
        totalRow += `<td style="background: #fff3e0; font-weight: bold;">${{total}}</td>`;
    }});
    totalRow += `<td style="background: #ff5722; color: white; font-weight: bold;">${{grandTotal}}</td></tr>`;
    
    document.getElementById('pivotTableBody').innerHTML = bodyHtml + totalRow;
    
    // Verify the math
    console.log('Total Events from raw data:', totalEvents);
    console.log('Grand Total from pivot:', grandTotal);
    if (totalEvents !== grandTotal) {{
        console.warn('Mismatch between raw total and pivot total!');
    }}
    
    // Update summary stats
    updateDataWordSummaryStats(pivotData, sortedMsgTypes, sortedDataWords, totalEvents, nonEmptyCells, grandTotal);
}}

// Also update showCellDetails to be clearer:
function showCellDetails(msgType, dataWord) {{
    const items = filteredDataWordData.filter(d => d.msg_type === msgType && d.data_word === dataWord);
    if (items.length === 0) return;
    
    // Calculate totals
    const totalIssues = items.reduce((sum, item) => sum + (item.total_issues || 0), 0);
    const totalSingle = items.reduce((sum, item) => sum + (item.single_word_changes || 0), 0);
    const totalMulti = items.reduce((sum, item) => sum + (item.multi_word_changes || 0), 0);
    
    let detailsText = `${{msgType}} - ${{dataWord}}\\n`;
    detailsText += `${{'='.repeat(40)}}\\n\\n`;
    
    detailsText += `TOTAL ISSUES: ${{totalIssues}}\\n`;
    detailsText += `Single Word Changes: ${{totalSingle}}\\n`;
    detailsText += `Multi Word Changes: ${{totalMulti}}\\n\\n`;
    
    if (items.length === 1) {{
        const item = items[0];
        detailsText += `Location: ${{item.unit_id}}/${{item.station}}/${{item.save}}\\n\\n`;
        detailsText += `Error Patterns:\\n${{item.top_error_patterns || 'N/A'}}`;
    }} else {{
        detailsText += `${{items.length}} Location Combinations:\\n`;
        detailsText += `${{'─'.repeat(30)}}\\n`;
        
        // Sort by issue count descending
        items.sort((a, b) => (b.total_issues || 0) - (a.total_issues || 0));
        
        items.forEach(item => {{
            detailsText += `\\n${{item.unit_id}}/${{item.station}}/${{item.save}}:`;
            detailsText += ` ${{item.total_issues || 0}} issues`;
            if (item.single_word_changes || item.multi_word_changes) {{
                detailsText += ` (Single:${{item.single_word_changes || 0}}, Multi:${{item.multi_word_changes || 0}})`;
            }}
        }});
        
        // Show common patterns
        const patterns = [...new Set(items
            .map(item => item.top_error_patterns)
            .filter(p => p)
            .flatMap(p => p.split(' | ')))];
        
        if (patterns.length > 0) {{
            detailsText += `\\n\\nCommon Error Patterns:\\n`;
            patterns.slice(0, 5).forEach(p => {{
                detailsText += `• ${{p}}\\n`;
            }});
        }}
    }}
    
    alert(detailsText);
}}
