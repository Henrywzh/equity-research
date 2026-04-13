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
        # Extract unique tickers from mapping, handling multiple tickers per category
        all_tickers_raw = self.mapping_df['Tickers'].dropna().unique()
        tickers = []
        for t_str in all_tickers_raw:
            # Tickers are comma-separated in the CSV
            for t in t_str.split(','):
                t = t.strip()
                if t and t not in tickers:
                    tickers.append(t)
        
        # We need 1 year + extra for start-of-month calcs
        data = self.engine.fetch_data(tickers, period="2y")
        prices = self.engine.get_close_prices(data)
        
        if prices is None: return None

        # Calculate returns for different periods
        # Use simple end-to-end return
        ret_1m = prices.pct_change(21).iloc[-1]  # ~1 month
        ret_3m = prices.pct_change(63).iloc[-1]  # ~3 months
        ret_12m = prices.pct_change(252).iloc[-1] # ~12 months
        
        # Combine into a results list
        scan_results = []
        # We iterate over the mapping to keep sub-industry names
        for _, row in self.mapping_df.iterrows():
            sub_name = row['Name']
            primary_ticker = row['Tickers'].split(',')[0].strip()
            
            if primary_ticker in prices.columns:
                scan_results.append({
                    "sub_industry": sub_name,
                    "ticker": primary_ticker,
                    "parent_industry": row['Parent Industry'],
                    "m_1m": ret_1m[primary_ticker],
                    "m_3m": ret_3m[primary_ticker],
                    "m_12m": ret_12m[primary_ticker],
                    "score": (ret_1m[primary_ticker] * 0.4 + ret_3m[primary_ticker] * 0.4 + ret_12m[primary_ticker] * 0.2)
                })
        
        # Sort by composite score
        scan_results.sort(key=lambda x: x['score'], reverse=True)
        return scan_results

    def get_top_movers(self, results, top_n=20):
        return results[:top_n]
