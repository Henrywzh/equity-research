import pandas as pd
import numpy as np
import os
from market_analysis.data_engine import DataEngine

class MomentumScanner:
    def __init__(self, data_engine, mapping_file="data/sub_industries_etfs.csv"):
        self.engine = data_engine
        self.mapping_file = mapping_file
        self.mapping_df = pd.read_csv(mapping_file)
        
    def analyze(self):
        # Extract unique tickers from mapping
        all_tickers_raw = self.mapping_df['Tickers'].dropna().unique()
        tickers = []
        for t_str in all_tickers_raw:
            for t in t_str.split(','):
                t = t.strip()
                if t and t not in tickers:
                    tickers.append(t)
        
        data = self.engine.fetch_data(tickers, period="2y")
        prices = self.engine.get_close_prices(data)
        
        if prices is None: return None

        # Calculate returns
        ret_1m = prices.pct_change(21).iloc[-1]
        ret_3m = prices.pct_change(63).iloc[-1]
        ret_12m = prices.pct_change(252).iloc[-1]
        
        # Combine into a results list
        scan_results = []
        for _, row in self.mapping_df.iterrows():
            sub_name = row['Name']
            primary_ticker = row['Tickers'].split(',')[0].strip()
            
            if primary_ticker in prices.columns:
                p_series = prices[primary_ticker]
                # Cumulative performance for charting
                perf_series = (p_series / p_series.iloc[0]) - 1
                
                scan_results.append({
                    "id": primary_ticker,
                    "sub_industry": sub_name,
                    "ticker": primary_ticker,
                    "parent_industry": row['Parent Industry'],
                    "m_1m": ret_1m[primary_ticker],
                    "m_3m": ret_3m[primary_ticker],
                    "m_12m": ret_12m[primary_ticker],
                    "score": (ret_1m[primary_ticker] * 0.4 + ret_3m[primary_ticker] * 0.4 + ret_12m[primary_ticker] * 0.2),
                    "history": {
                        "dates": perf_series.index.strftime('%Y-%m-%d').tolist(),
                        "performance": perf_series.fillna(0).tolist()
                    }
                })
        
        # Sort by composite score
        scan_results.sort(key=lambda x: x['score'], reverse=True)
        return scan_results

    def get_top_movers(self, results, top_n=20):
        return results[:top_n]
