import pandas as pd
import numpy as np
import plotly.graph_objects as go
from market_analysis.data_engine import DataEngine

class PairsTracker:
    def __init__(self, data_engine):
        self.engine = data_engine
        self.pairs = [
            ("SMH", "IGV", "Semis vs Software"),
            ("KRE", "XLF", "Regional vs Broad Financials"),
            ("XHB", "XRT", "Homebuilders vs Retail"),
            ("ITA", "XLI", "Defense vs Broad Industrials")
        ]
        self.all_tickers = list(set([t for p in self.pairs for t in (p[0], p[1])]))

    def analyze(self):
        data = self.engine.fetch_data(self.all_tickers, period="2y")
        prices = self.engine.get_close_prices(data)
        
        if prices is None: return []

        results = []
        for t1, t2, desc in self.pairs:
            if t1 not in prices.columns or t2 not in prices.columns:
                print(f"Skipping pair {t1}/{t2}: One or more tickers missing.")
                continue
                
            ratio = prices[t1] / prices[t2]
            
            # 1-year window for Z-Score (approx 252 trading days)
            rolling_mean = ratio.rolling(window=252).mean()
            rolling_std = ratio.rolling(window=252).std()
            z_score = (ratio - rolling_mean) / rolling_std
            
            curr_z = z_score.iloc[-1]
            curr_ratio = ratio.iloc[-1]
            
            # Mean reversion signal
            signal = "Neutral"
            if curr_z > 2.0: signal = f"Short {t1} / Long {t2} (Overstretched)"
            elif curr_z < -2.0: signal = f"Long {t1} / Short {t2} (Undervalued)"
            
            results.append({
                "ticker1": t1,
                "ticker2": t2,
                "description": desc,
                "current_ratio": round(curr_ratio, 4),
                "z_score": round(curr_z, 2),
                "signal": signal,
                "last_updated": ratio.index[-1].strftime('%Y-%m-%d')
            })
            
        return results

    def create_pair_plot(self, t1, t2, desc, prices):
        ratio = prices[t1] / prices[t2]
        rolling_mean = ratio.rolling(window=252).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio, name="Ratio", line=dict(color="#3b82f6")))
        fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean, name="1Y Mean", line=dict(color="#64748b", dash='dash')))
        
        fig.update_layout(
            title=f"Pairs Analysis: {desc} ({t1}/{t2})",
            template="plotly_dark",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
