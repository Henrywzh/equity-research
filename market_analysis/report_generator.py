import os
import json
from datetime import datetime

class ReportGenerator:
    def __init__(self, output_path="report/dashboard.html"):
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
        
        .pos { color: var(--green); font-weight: 500; }
        .neg { color: var(--red); font-weight: 500; }
        .mono { font-family: 'Fira Code', monospace; font-size: 0.85rem; }
        
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
        for m in momentum_data[:20]:
            momentum_rows += f"""
            <tr class="clickable" onclick="showDetail('momentum', '{m['id']}')">
                <td>{m['sub_industry']}</td>
                <td class="mono">{m['ticker']}</td>
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
                        <h3>Level 1: Market Regime Monitor</h3>
                        {plots.get('regime', '')}
                    </div>

                    <div class="card">
                        <h3>Level 2: Industry Pairs</h3>
                        <table>
                            <thead><tr><th>DESCRIPTION</th><th>PAIR</th><th>RATIO</th><th>Z-SCORE</th><th>SIGNAL</th></tr></thead>
                            <tbody>{pairs_rows}</tbody>
                        </table>
                    </div>

                    <div class="card">
                        <h3>Level 3: Sub-Industry Momentum</h3>
                        <table>
                            <thead><tr><th>SUB-INDUSTRY</th><th>TICKER</th><th>1M</th><th>3M</th><th>12M</th></tr></thead>
                            <tbody>{momentum_rows}</tbody>
                        </table>
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
                        : `Momentum analysis for ${{item.ticker}} (${{item.parent_industry}})`;

                    let metaHtml = '';
                    if (type === 'pair') {{
                        metaHtml = `
                            <div class="meta-item"><div class="meta-label">Current Ratio</div><div class="meta-value">${{item.current_ratio}}</div></div>
                            <div class="meta-item"><div class="meta-label">1Y Z-Score</div><div class="meta-value">${{item.z_score}}</div></div>
                            <div class="meta-item" style="grid-column: span 2;"><div class="meta-label">Suggested Action</div><div class="meta-value" style="color: var(--yellow)">${{item.signal}}</div></div>
                        `;
                    }} else {{
                        metaHtml = `
                            <div class="meta-item"><div class="meta-label">1-Month</div><div class="meta-value ${{item.m_1m > 0 ? 'pos' : 'neg'}}">${{(item.m_1m*100).toFixed(1)}}%</div></div>
                            <div class="meta-item"><div class="meta-label">3-Month</div><div class="meta-value ${{item.m_3m > 0 ? 'pos' : 'neg'}}">${{(item.m_3m*100).toFixed(1)}}%</div></div>
                            <div class="meta-item"><div class="meta-label">12-Month</div><div class="meta-value ${{item.m_12m > 0 ? 'pos' : 'neg'}}">${{(item.m_12m*100).toFixed(1)}}%</div></div>
                            <div class="meta-item"><div class="meta-label">Momentum Score</div><div class="meta-value">${{item.score.toFixed(2)}}</div></div>
                        `;
                    }}
                    document.getElementById('detail-meta').innerHTML = metaHtml;

                    // Render Chart
                    const trace = {{
                        x: item.history.dates,
                        y: type === 'pair' ? item.history.ratio : item.history.performance,
                        type: 'scatter',
                        mode: 'lines',
                        line: {{ color: '#3b82f6', width: 2 }},
                        fill: 'tozeroy',
                        fillcolor: 'rgba(59, 130, 246, 0.1)'
                    }};

                    const layout = {{
                        template: 'plotly_dark',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        margin: {{ l: 40, r: 0, t: 20, b:40 }},
                        xaxis: {{ showgrid: false, rangeslider: {{ visible: true, thickness: 0.1 }} }},
                        yaxis: {{ gridcolor: '#1f2937' }}
                    }};

                    Plotly.newPlot('detail-chart', [trace], layout);
                    overlay.classList.add('active');
                }}

                function closeDetail() {{
                    document.getElementById('detail-overlay').classList.remove('active');
                }}
            </script>
        </body>
        </html>
        """
        
        with open(self.output_path, "w") as f:
            f.write(html_content)
        return self.output_path
