function drawRiskCharts() {
                if (testCaseRiskData.length === 0) {
                    document.getElementById('riskOverviewChart').innerHTML = '<p>No risk data available</p>';
                    return;
                }
                
                // This uses the globally filtered risk data
                const topRisk = filteredRiskData.slice(0, 20);
                
                // 1. Draw the first chart
                Plotly.newPlot('riskOverviewChart', [{
                    x: topRisk.map(d => d.test_case),
                    y: topRisk.map(d => d.total_bus_flips),
                    type: 'bar',
                    marker: { 
                        color: topRisk.map(d => 
                            d.total_bus_flips > 100 ? '#e74c3c' :
                            d.total_bus_flips > 50 ? '#f39c12' : '#f0ad4e'
                        )
                    },
                    text: topRisk.map(d => `${d.instances_run} runs, ${d.requirements_at_risk} reqs at risk`),
                    hoverinfo: 'none' // Your request
                }], {
                    margin: { t: 10, b: 120, l: 60, r: 20 },
                    xaxis: { title: 'Test Case', tickangle: -45 },
                    yaxis: { title: 'Total Bus Flips (Deduplicated)' }
                });
                
                // 2. Populate the "Requirements at Risk" table
                const reqRiskData = filteredRiskData.filter(d => d.requirements_at_risk > 0);
                const reqTableBody = document.getElementById('requirementsAtRiskTableBody');
                reqTableBody.innerHTML = '';
                
                if (reqRiskData.length > 0) {
                    // Sort by most at risk first
                    reqRiskData.sort((a, b) => b.requirements_at_risk - a.requirements_at_risk); 
                    
                    reqRiskData.forEach(risk => {
                        const row = reqTableBody.insertRow();
                        row.insertCell(0).textContent = risk.test_case;
                        row.insertCell(1).textContent = risk.total_bus_flips;
                        row.insertCell(2).textContent = risk.requirements_at_risk;
                        
                        // -----------------------------------------------------------------
                        // FIX: Add null check. This was the first crash.
                        // -----------------------------------------------------------------
                        let reqListHtml = 'None';
                        if (risk.requirement_list) {
                            reqListHtml = risk.requirement_list.split(', ').join('<br>');
                        }
                        const reqCell = row.insertCell(3);
                        reqCell.innerHTML = reqListHtml;
                        reqCell.style.whiteSpace = "nowrap"; 
                    });
                } else {
                    reqTableBody.innerHTML = '<tr><td colspan="4">No requirements at risk found for current filter.</td></tr>';
                }
                
                // 3. Populate the "Detailed Test Case Risk Analysis" table
                const tableBody = document.getElementById('riskDetailTableBody');
                tableBody.innerHTML = '';
                
                if (filteredRiskData.length > 0) {
                    // Sort this table by total flips
                    filteredRiskData.sort((a, b) => b.total_bus_flips - a.total_bus_flips);
                    
                    filteredRiskData.forEach(risk => {
                        const row = tableBody.insertRow();
                        
                        row.insertCell(0).textContent = risk.test_case;
                        row.insertCell(1).textContent = risk.total_bus_flips;
                        row.insertCell(2).textContent = risk.avg_flips_per_run.toFixed(1);
                        row.insertCell(3).textContent = `${risk.top_station_save_combo} (${risk.top_combo_flips})`;
                        row.insertCell(4).textContent = risk.station_save_combos || 'N/A';
                        row.insertCell(5).textContent = risk.top_msg_types || 'N/A';
                        
                        // -----------------------------------------------------------------
                        // FIX: Add null check. This was the second (or identical) crash.
                        // -----------------------------------------------------------------
                        let reqHtml = 'None';
                        if (risk.requirement_list) {
                            const reqList = risk.requirement_list.split(', ');
                            reqHtml = reqList.length > 3 
                                ? `${reqList.slice(0,3).join(', ')}... (+${reqList.length - 3} more)` 
                                : reqList.join(', ');
                        }
                        row.insertCell(6).textContent = `${risk.requirements_at_risk}: ${reqHtml}`;
    
                        const timeCell = row.insertCell(7);
                        timeCell.textContent = risk.time_ranges || 'N/A';
                        timeCell.className = 'time-range-cell';
                    });
                } else {
                    tableBody.innerHTML = '<tr><td colspan="8">No detailed risk data to display for current filter.</td></tr>';
                }
            }
