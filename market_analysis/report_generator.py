import os
import json
from datetime import datetime

class ReportGenerator:
    def __init__(self, output_path="hub/market-analysis.html"):
        self.output_path = output_path
        self.styles = """
        :root {
            --bg-dark: #0b0f19;
            --card-bg: #111827;
            --border: #1f2937;
            --text-main: #f9fafb;
            --text-dim: #9ca3af;
            --accent: #3b82f6;
            --green: #10b981;
            --red: #ef4444;
            --yellow: #f59e0b;
        }
        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Inter', -apple-system, sans-serif;
            margin: 0; padding: 0;
            overflow-x: hidden;
        }
        .container { max-width: 1300px; margin: 0 auto; padding: 30px; }
        .header { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 40px; }
        .button-link {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 8px 16px;
            border-radius: 9999px;
            border: 1px solid var(--border);
            color: var(--text-dim);
            text-decoration: none;
            background: var(--card-bg);
            font-size: 0.85rem;
            font-weight: 500;
            transition: border-color 0.2s ease, color 0.2s ease, transform 0.2s ease;
        }
        .button-link:hover {
            border-color: var(--accent);
            color: var(--text-main);
            transform: translateY(-1px);
        }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
        .card { 
            background: var(--card-bg); 
            border: 1px solid var(--border);
            border-radius: 16px; 
            padding: 24px; 
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
        }
        .card-full { grid-column: 1 / -1; }
        .card h3 { margin-top: 0; font-weight: 600; color: var(--text-main); font-size: 1.1rem; }
        
        table { width: 100%; border-collapse: separate; border-spacing: 0; margin-top: 12px; }
        th { text-align: left; color: var(--text-dim); padding: 12px 16px; font-size: 0.85rem; font-weight: 500; border-bottom: 1px solid var(--border); }
        td { padding: 14px 16px; border-bottom: 1px solid var(--border); font-size: 0.9rem; transition: background 0.2s; }
        tr.clickable { cursor: pointer; }
        tr.clickable:hover td { background: rgba(59, 130, 246, 0.1); }
        .card-note { color: var(--text-dim); font-size: 0.82rem; line-height: 1.5; margin: -4px 0 14px; }
        
        .pos { color: var(--green); font-weight: 500; }
        .neg { color: var(--red); font-weight: 500; }
        .mono { font-family: 'Fira Code', monospace; font-size: 0.85rem; }
        .badge {
            display: inline-flex;
            align-items: center;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid transparent;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            white-space: nowrap;
        }
        .badge-direct { background: rgba(16, 185, 129, 0.12); color: #6ee7b7; border-color: rgba(16, 185, 129, 0.28); }
        .badge-industry-fallback { background: rgba(59, 130, 246, 0.12); color: #93c5fd; border-color: rgba(59, 130, 246, 0.28); }
        .badge-industry-group-fallback { background: rgba(245, 158, 11, 0.12); color: #fcd34d; border-color: rgba(245, 158, 11, 0.28); }
        .badge-sector-fallback { background: rgba(239, 68, 68, 0.12); color: #fca5a5; border-color: rgba(239, 68, 68, 0.28); }
        
        /* Detail Overlay */
        #detail-overlay {
            position: fixed; top: 0; right: -100%; width: 50%; height: 100%;
            background: var(--card-bg); border-left: 1px solid var(--border);
            box-shadow: -10px 0 30px rgba(0,0,0,0.5);
            z-index: 1000; transition: right 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            padding: 40px; box-sizing: border-box;
            display: flex; flex-direction: column;
        }
        #detail-overlay.active { right: 0; }
        .close-btn { position: absolute; top: 20px; left: 20px; cursor: pointer; color: var(--text-dim); }
        .detail-header { margin-bottom: 30px; }
        .detail-meta { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
        .meta-item { background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; border: 1px solid var(--border); }
        .meta-label { font-size: 0.75rem; color: var(--text-dim); margin-bottom: 4px; }
        .meta-value { font-size: 1.2rem; font-weight: 600; }
        """

    def generate(self, regime_data, pairs_data, momentum_data, plots):
        # Prepare Data for JS injection
        market_data = {
            "regime": {
                "z_score": regime_data['z_score'],
                "status": regime_data['regime']
            },
            "pairs": pairs_data,
            "momentum": momentum_data
        }

        # Build Table Rows
        pairs_rows = ""
        for p in pairs_data:
            pairs_rows += f"""
            <tr class="clickable" onclick="showDetail('pair', '{p['id']}')">
                <td>{p['description']}</td>
                <td class="mono">{p['ticker1']}/{p['ticker2']}</td>
                <td>{p['current_ratio']:.4f}</td>
                <td class="mono {'pos' if p['z_score'] > 1.5 else 'neg' if p['z_score'] < -1.5 else ''}">{p['z_score']}</td>
                <td style="font-size: 0.8rem; color: var(--text-dim)">{p['signal']}</td>
            </tr>
            """

        momentum_rows = ""
        for m in momentum_data:
            badge_class = m["proxy_type"].lower().replace(" ", "-")
            momentum_rows += f"""
            <tr class="clickable" onclick="showDetail('momentum', '{m['id']}')">
                <td>{m['sub_industry']}</td>
                <td class="mono">{m['proxy_etf']}</td>
                <td><span class="badge badge-{badge_class}">{m['proxy_type']}</span></td>
                <td class="{'pos' if m['m_1m'] > 0 else 'neg'}">{m['m_1m']:.1%}</td>
                <td class="{'pos' if m['m_3m'] > 0 else 'neg'}">{m['m_3m']:.1%}</td>
                <td class="{'pos' if m['m_12m'] > 0 else 'neg'}">{m['m_12m']:.1%}</td>
            </tr>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Quant Intelligence | Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Fira+Code&display=swap" rel="stylesheet">
            <style>{self.styles}</style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div>
                        <div style="margin-bottom: 16px;">
                            <a class="button-link" href="index.html">← Back to Hub</a>
                        </div>
                        <h1 style="margin:0; font-size: 2rem;">Equity Intelligence Hub</h1>
                        <p style="color: var(--text-dim); margin-top:8px;">Market Intelligence & Quantitative Signals</p>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: {regime_data['color']}22; color: {regime_data['color']}; padding: 6px 16px; border-radius: 99px; border: 1px solid {regime_data['color']}44; font-weight: 600;">
                            {regime_data['regime']} (Z: {regime_data['z_score']})
                        </span>
                        <p style="margin-top:12px; font-size: 0.75rem; color: var(--text-dim);">LAST REFRESH: {datetime.now().strftime('%H:%M:%S UTC')}</p>
                    </div>
                </div>

                <div class="grid">
                    <div class="card card-full">
                        <h3>Level 1: Market Regime Monitor (Offensive vs. Defensive Sectors)</h3>
                        {plots.get('regime', '')}
                    </div>

                    <div class="card card-full">
                        <h3>Level 1: Market Breadth Monitor (S&P 500 Equal-Weight vs. Cap-Weight)</h3>
                        {plots.get('breadth', '')}
                    </div>

                    <div class="card">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <h3 style="margin:0">Level 2: Industry Pairs</h3>
                        </div>
                        <div style="max-height: 600px; overflow-y: auto;">
                            <table>
                                <thead style="position: sticky; top: 0; background: var(--card-bg); z-index: 10;">
                                    <tr><th>DESCRIPTION</th><th>PAIR</th><th>RATIO</th><th>Z-SCORE</th><th>SIGNAL</th></tr>
                                </thead>
                                <tbody>{pairs_rows}</tbody>
                            </table>
                        </div>
                    </div>

                    <div class="card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                            <div>
                                <h3 style="margin:0">Level 3: Sub-Industry Proxies</h3>
                                <p class="card-note" style="margin-top:4px;">Ranked by composite momentum. Full coverage of 160+ sub-industries.</p>
                            </div>
                            <input type="text" id="momentum-search" placeholder="Search sub-industries..." 
                                style="background: rgba(255,255,255,0.05); border: 1px solid var(--border); color: white; padding: 6px 12px; border-radius: 8px; font-size: 0.8rem; width: 200px;">
                        </div>
                        <div style="max-height: 600px; overflow-y: auto;">
                            <table id="momentum-table">
                                <thead style="position: sticky; top: 0; background: var(--card-bg); z-index: 10;">
                                    <tr><th>SUB-INDUSTRY</th><th>PROXY ETF</th><th>COVERAGE</th><th>1M</th><th>3M</th><th>12M</th></tr>
                                </thead>
                                <tbody>{momentum_rows}</tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Detail Drawer -->
            <div id="detail-overlay">
                <div class="close-btn" onclick="closeDetail()">← BACK</div>
                <div id="detail-content">
                    <div class="detail-header">
                        <h2 id="detail-title" style="margin:0"></h2>
                        <p id="detail-subtitle" style="color: var(--text-dim)"></p>
                    </div>
                    <div id="detail-chart" style="height: 400px; margin-bottom: 30px;"></div>
                    <div class="detail-meta" id="detail-meta"></div>
                </div>
            </div>

            <script>
                const MARKET_DATA = {json.dumps(market_data)};

                function badgeClass(proxyType) {{
                    return `badge-${{proxyType.toLowerCase().replace(/\\s+/g, '-')}}`;
                }}
                
                function showDetail(type, id) {{
                    const overlay = document.getElementById('detail-overlay');
                    const item = type === 'pair' 
                        ? MARKET_DATA.pairs.find(p => p.id === id)
                        : MARKET_DATA.momentum.find(m => m.id === id);
                    
                    if (!item) return;

                    // Set Metadata
                    document.getElementById('detail-title').innerText = type === 'pair' ? item.description : item.sub_industry;
                    document.getElementById('detail-subtitle').innerText = type === 'pair' 
                        ? `Relationship between ${{item.ticker1}} and ${{item.ticker2}}` 
                        : `${{item.sub_industry}} via ${{item.proxy_etf}}`;

                    let metaHtml = '';
                    if (type === 'pair') {{
                        metaHtml = `
                            <div class="meta-item"><div class="meta-label">Current Ratio</div><div class="meta-value">${{item.current_ratio}}</div></div>
                            <div class="meta-item"><div class="meta-label">1Y Z-Score</div><div class="meta-value">${{item.z_score}}</div></div>
                            <div class="meta-item" style="grid-column: span 2;"><div class="meta-label">Suggested Action</div><div class="meta-value" style="color: var(--yellow)">${{item.signal}}</div></div>
                        `;
                    }} else {{
                        const relatedTickers = item.related_tickers && item.related_tickers.length > 1
                            ? `<div class="meta-item" style="grid-column: span 2;"><div class="meta-label">Mapped ETF Set</div><div class="meta-value mono">${{item.related_tickers.join(', ')}}</div></div>`
                            : '';
                        metaHtml = `
                            <div class="meta-item"><div class="meta-label">Proxy ETF</div><div class="meta-value mono">${{item.proxy_etf}}</div></div>
                            <div class="meta-item"><div class="meta-label">Coverage</div><div class="meta-value"><span class="badge ${{badgeClass(item.proxy_type)}}">${{item.proxy_type}}</span></div></div>
                            <div class="meta-item"><div class="meta-label">Parent Industry</div><div class="meta-value">${{item.parent_industry}}</div></div>
                            <div class="meta-item"><div class="meta-label">1-Month</div><div class="meta-value ${{item.m_1m > 0 ? 'pos' : 'neg'}}">${{(item.m_1m*100).toFixed(1)}}%</div></div>
                            <div class="meta-item"><div class="meta-label">3-Month</div><div class="meta-value ${{item.m_3m > 0 ? 'pos' : 'neg'}}">${{(item.m_3m*100).toFixed(1)}}%</div></div>
                            <div class="meta-item"><div class="meta-label">12-Month</div><div class="meta-value ${{item.m_12m > 0 ? 'pos' : 'neg'}}">${{(item.m_12m*100).toFixed(1)}}%</div></div>
                            <div class="meta-item"><div class="meta-label">Momentum Score</div><div class="meta-value">${{item.score.toFixed(2)}}</div></div>
                            ${{relatedTickers}}
                        `;
                    }}
                    document.getElementById('detail-meta').innerHTML = metaHtml;

                    // Render Chart
                    // Render Chart
                    const trace = {{
                        x: item.history.dates,
                        y: type === 'pair' ? item.history.ratio : item.history.performance,
                        type: 'scatter',
                        mode: 'lines',
                        line: {{ color: '#3b82f6', width: 2.5 }},
                        fill: 'tozeroy',
                        fillcolor: 'rgba(59, 130, 246, 0.1)',
                        hovertemplate: type === 'pair' ? "Ratio: %{{y:.4f}}<extra></extra>" : "Perf: %{{y:.1%}}<extra></extra>"
                    }};

                    // Calculate 1Y range
                    const lastDate = new Date(item.history.dates[item.history.dates.length - 1]);
                    const oneYearAgo = new Date(lastDate);
                    oneYearAgo.setFullYear(lastDate.getFullYear() - 1);

                    const layout = {{
                        template: 'plotly_dark',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        margin: {{ l: 40, r: 0, t: 30, b:40 }},
                        hovermode: 'x unified',
                        hoverlabel: {{
                            bgcolor: '#111827',
                            bordercolor: '#3b82f6',
                            font: {{ color: '#f9fafb', family: 'Inter', size: 13 }}
                        }},
                        xaxis: {{ 
                            showgrid: false, 
                            range: [oneYearAgo.toISOString().split('T')[0], lastDate.toISOString().split('T')[0]],
                            rangeslider: {{ visible: true, thickness: 0.1 }},
                            rangeselector: {{
                                buttons: [
                                    {{count: 1, label: '1M', step: 'month', stepmode: 'backward'}},
                                    {{count: 6, label: '6M', step: 'month', stepmode: 'backward'}},
                                    {{count: 1, label: '1Y', step: 'year', stepmode: 'backward'}},
                                    {{step: 'all', label: 'ALL'}}
                                ],
                                bgcolor: '#1e293b',
                                activecolor: '#3b82f6',
                                font: {{ color: '#f8fafc', size: 11 }}
                            }}
                        }},
                        yaxis: {{ 
                            gridcolor: '#1f2937',
                            tickformat: type === 'pair' ? '.4f' : '.1%'
                        }}
                    }};

                    Plotly.newPlot('detail-chart', [trace], layout);
                    overlay.classList.add('active');
                }}

                function closeDetail() {{
                    document.getElementById('detail-overlay').classList.remove('active');
                }}

                // Search functionality
                document.getElementById('momentum-search').addEventListener('keyup', function() {{
                    const query = this.value.toLowerCase();
                    const rows = document.querySelectorAll('#momentum-table tbody tr');
                    
                    rows.forEach(row => {{
                        const text = row.innerText.toLowerCase();
                        row.style.display = text.includes(query) ? '' : 'none';
                    }});
                }});
            </script>
        </body>
        </html>
        """
        
        with open(self.output_path, "w") as f:
            f.write(html_content)
        return self.output_path
