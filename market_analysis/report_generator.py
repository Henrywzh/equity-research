import os
from datetime import datetime

class ReportGenerator:
    def __init__(self, output_path="report/dashboard.html"):
        self.output_path = output_path
        self.styles = """
        :root {
            --bg-dark: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
            --accent: #3b82f6;
            --green: #10b981;
            --red: #ef4444;
            --yellow: #f59e0b;
        }
        body {
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; border-bottom: 1px solid #334155; padding-bottom: 10px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .card { background-color: var(--card-bg); border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        .card-full { grid-column: 1 / -1; }
        .regime-badge { padding: 8px 16px; border-radius: 20px; font-weight: 600; font-size: 0.9em; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th { text-align: left; color: var(--text-dim); padding: 12px 8px; font-weight: 500; border-bottom: 1px solid #334155; }
        td { padding: 12px 8px; border-bottom: 1px solid #334155; }
        .pos { color: var(--green); }
        .neg { color: var(--red); }
        .mono { font-family: 'Fira Code', monospace; }
        """

    def generate(self, regime_data, pairs_data, momentum_data, plots):
        
        # Format Momentum Table
        momentum_rows = ""
        for item in momentum_data[:20]: # Top 20
            momentum_rows += f"""
            <tr>
                <td>{item['sub_industry']}</td>
                <td class="mono">{item['ticker']}</td>
                <td class="{ 'pos' if item['m_1m'] > 0 else 'neg' }">{item['m_1m']:.1%}</td>
                <td class="{ 'pos' if item['m_3m'] > 0 else 'neg' }">{item['m_3m']:.1%}</td>
                <td class="{ 'pos' if item['m_12m'] > 0 else 'neg' }">{item['m_12m']:.1%}</td>
            </tr>
            """

        # Format Pairs Table
        pairs_rows = ""
        for p in pairs_data:
            pairs_rows += f"""
            <tr>
                <td>{p['description']}</td>
                <td class="mono">{p['ticker1']}/{p['ticker2']}</td>
                <td>{p['current_ratio']:.4f}</td>
                <td class="mono">{p['z_score']}</td>
                <td style="color: {'var(--yellow)' if abs(p['z_score']) > 1.5 else 'inherit'}">{p['signal']}</td>
            </tr>
            """

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Quant Equity Intelligence Dashboard</title>
            <style>{self.styles}</style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div>
                        <h1 style="margin:0;">Equity Intelligence Hub</h1>
                        <p style="color: var(--text-dim); margin-top:5px;">Multi-Tier Market Analysis Dashboard</p>
                    </div>
                    <div style="text-align: right;">
                        <span class="regime-badge" style="background-color: {regime_data['color']}; color: #000;">
                            Regime: {regime_data['regime']} (Z: {regime_data['z_score']})
                        </span>
                        <p style="margin-top:10px; font-size: 0.8em; color: var(--text-dim);">Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                    </div>
                </div>

                <div class="grid">
                    <div class="card card-full">
                        <h3>Level 1: Market Regime (Sector RORO)</h3>
                        {plots.get('regime', '')}
                    </div>

                    <div class="card">
                        <h3>Level 2: Industry Pairs Monitor</h3>
                        <table>
                            <thead>
                                <tr><th>Pair</th><th>Tickers</th><th>Ratio</th><th>Z-Score</th><th>Action</th></tr>
                            </thead>
                            <tbody>{pairs_rows}</tbody>
                        </table>
                    </div>

                    <div class="card">
                        <h3>Level 3: Top Momentum (Sub-Industries)</h3>
                        <table>
                            <thead>
                                <tr><th>Sub-Industry</th><th>Tickers</th><th>1M</th><th>3M</th><th>12M</th></tr>
                            </thead>
                            <tbody>{momentum_rows}</tbody>
                        </table>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_path, "w") as f:
            f.write(html_content)
        return self.output_path
