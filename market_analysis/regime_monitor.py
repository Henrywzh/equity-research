import plotly.graph_objects as go
from datetime import datetime, timedelta

class RegimeMonitor:
    def __init__(self, data_engine):
        self.engine = data_engine
        self.offensive = ['XLK', 'XLY', 'XLC']
        self.defensive = ['XLP', 'XLU', 'XLV']
        self.all_tickers = self.offensive + self.defensive

    def _available_group(self, prices, tickers, label):
        available = [ticker for ticker in tickers if ticker in prices.columns]
        missing = [ticker for ticker in tickers if ticker not in prices.columns]
        if missing:
            print(f"Warning: Missing {label} tickers: {', '.join(missing)}")
        if not available:
            raise ValueError(f"No {label} tickers available after data cleanup.")
        return available

    def analyze(self, prices=None):
        if prices is None:
            prices = self.engine.fetch_data(self.all_tickers)
        if prices is None or prices.empty:
            return None

        offensive = self._available_group(prices, self.offensive, "offensive")
        defensive = self._available_group(prices, self.defensive, "defensive")

        # Calculate equal-weighted groups
        off_bench = prices[offensive].pct_change(fill_method=None).mean(axis=1).add(1).cumprod()
        def_bench = prices[defensive].pct_change(fill_method=None).mean(axis=1).add(1).cumprod()
        
        # Calculate Ratio
        ratio = off_bench / def_bench
        
        # Calculate Trend (60-day MA)
        ratio_ma = ratio.rolling(window=60).mean()
        
        # Calculate Z-Score relative to 1-year history
        rolling_mean = ratio.rolling(window=252).mean()
        rolling_std = ratio.rolling(window=252).std()
        z_score = (ratio - rolling_mean) / rolling_std
        
        current_ratio = ratio.iloc[-1]
        current_z = z_score.iloc[-1]
        
        # Determine Regime
        if current_z > 1.0:
            regime = "Aggressive Risk-On"
            color = "#00ff00"
        elif current_z > 0:
            regime = "Moderate Risk-On"
            color = "#aaffaa"
        elif current_z > -1.0:
            regime = "Cautious Risk-Off"
            color = "#ffaaaa"
        else:
            regime = "Extreme Risk-Off"
            color = "#ff0000"

        return {
            "regime": regime,
            "color": color,
            "z_score": round(current_z, 2),
            "ratio_history": ratio,
            "ma_history": ratio_ma,
            "z_history": z_score
        }

    def create_plot(self, results):
        fig = go.Figure()
        
        # Add Ratio line
        fig.add_trace(go.Scatter(
            x=results["ratio_history"].index, 
            y=results["ratio_history"],
            mode='lines',
            name='Offensive/Defensive Ratio',
            line=dict(color='#3b82f6', width=2.5),
            hovertemplate="Ratio: %{y:.4f}<extra></extra>"
        ))
        
        # Add MA line
        fig.add_trace(go.Scatter(
            x=results["ma_history"].index, 
            y=results["ma_history"],
            mode='lines',
            name='60d Moving Avg',
            line=dict(color='#94a3b8', width=1.5, dash='dot'),
            hovertemplate="MA: %{y:.4f}<extra></extra>"
        ))
        
        # Update layout for "Trading 212" style
        fig.update_layout(
            title=dict(
                text="Market Regime Monitor (Risk-On / Risk-Off)",
                font=dict(size=20, color="#f8fafc")
            ),
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=40, t=60, b=40),
            height=500,
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="#111827",
                bordercolor="#3b82f6",
                font=dict(color="#f9fafb", size=13),
                namelength=-1,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(
                range=[results["ratio_history"].index[-1] - timedelta(days=365), results["ratio_history"].index[-1]],
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                showline=True,
                showgrid=False,
                rangeslider=dict(visible=True, thickness=0.08),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="ALL")
                    ]),
                    bgcolor="#1e293b",
                    activecolor="#3b82f6",
                    font=dict(color="#f8fafc")
                )
            ),
            yaxis=dict(
                side="right",
                showgrid=True,
                gridcolor="#334155",
                zeroline=False
            )
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
