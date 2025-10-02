function updateDataWordSummaryStats(pivotData, msgTypes, dataWords, totalEvents, nonEmptyCells, pivotTotal) {{
    // Aggregate issues by msg_type-data_word combination
    const aggregatedIssues = {{}};
    filteredDataWordData.forEach(d => {{
        const key = `${{d.msg_type}}-${{d.data_word}}`;
        if (!aggregatedIssues[key]) {{
            aggregatedIssues[key] = {{
                msg_type: d.msg_type,
                data_word: d.data_word,
                total: 0,
                locations: 0
            }};
        }}
        aggregatedIssues[key].total += d.total_issues || 0;
        aggregatedIssues[key].locations += 1;
    }});
    
    // Sort and get top 5
    const topIssues = Object.values(aggregatedIssues)
        .sort((a, b) => b.total - a.total)
        .slice(0, 5);
    
    // Calculate multi-word percentage from the raw data
    let totalMulti = 0;
    let totalSingle = 0;
    
    // Sum up all single and multi counts from pivot data
    Object.values(pivotData).forEach(msgTypeData => {{
        Object.values(msgTypeData).forEach(cellData => {{
            totalMulti += cellData.multi || 0;
            totalSingle += cellData.single || 0;
        }});
    }});
    
    const multiPercent = totalMulti + totalSingle > 0 ? (totalMulti / (totalMulti + totalSingle) * 100).toFixed(1) : 0;
    
    // Count unique location combinations
    const uniqueLocations = new Set(filteredDataWordData.map(d => `${{d.unit_id}}|${{d.station}}|${{d.save}}`));
    
    const statsHtml = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div><strong>Total Data Word Changes:</strong> ${{totalEvents.toLocaleString()}}</div>
            <div><strong>Pivot Table Total:</strong> ${{pivotTotal.toLocaleString()}}</div>
            <div><strong>Non-Empty Cells:</strong> ${{nonEmptyCells}}</div>
            <div><strong>Analysis Rows:</strong> ${{filteredDataWordData.length}}</div>
            <div><strong>Unique Locations:</strong> ${{uniqueLocations.size}}</div>
            <div><strong>Unique Message Types:</strong> ${{msgTypes.length}}</div>
            <div><strong>Unique Data Words:</strong> ${{dataWords.length}}</div>
            <div><strong>Multi-Word Change Rate:</strong> ${{multiPercent}}%</div>
        </div>
        <div style="margin-top: 15px;">
            <strong>Top 5 Problem Areas (Aggregated):</strong>
            <ol>
                ${{topIssues.map(d => 
                    `<li>${{d.msg_type}}-${{d.data_word}}: ${{d.total}} total changes (${{d.locations}} location${{d.locations !== 1 ? 's' : ''}})</li>`
                ).join('')}}
            </ol>
        </div>
        <div style="margin-top: 10px; padding: 10px; background: #e3f2fd; border-left: 4px solid #2196F3; border-radius: 4px;">
            <strong>Reading the Pivot Table:</strong><br>
            • Each cell shows the <strong>total count of data word changes</strong> for that msg_type/data_word combination<br>
            • Numbers represent individual change events detected during bus flips<br>
            • Hover over cells to see breakdown by single/multi-word changes<br>
            • Click cells for detailed information including location breakdown
        </div>
    `;
    
    document.getElementById('summaryStatsContent').innerHTML = statsHtml;
}}
