import pandas as pd

class MomentumScanner:
    def __init__(self, data_engine, mapping_file="data/sub_industries_etfs.csv"):
        self.engine = data_engine
        self.mapping_file = mapping_file
        self.mapping_df = pd.read_csv(mapping_file)

    def _build_mappings(self):
        mappings = []
        for _, row in self.mapping_df.iterrows():
            tickers = [t.strip() for t in str(row["Tickers"]).split(",") if t.strip()]
            if not tickers:
                continue

            mappings.append(
                {
                    "id": str(row["GICS Code"]).strip(),
                    "sub_industry": row["Name"],
                    "proxy_etf": tickers[0],
                    "proxy_type": row["Mapping Type"],
                    "parent_industry": row["Parent Industry"],
                    "related_tickers": tickers,
                }
            )
        return mappings
        
    def analyze(self, prices=None):
        mappings = self._build_mappings()
        tickers = []
        for mapping in mappings:
            proxy_etf = mapping["proxy_etf"]
            if proxy_etf not in tickers:
                tickers.append(proxy_etf)
        
        if prices is None:
            prices = self.engine.fetch_data(tickers)
        if prices is None or prices.empty:
            return None

        # Calculate returns
        ret_1m = prices.pct_change(21, fill_method=None).iloc[-1]
        ret_3m = prices.pct_change(63, fill_method=None).iloc[-1]
        ret_12m = prices.pct_change(252, fill_method=None).iloc[-1]
        
        # Combine into a results list
        scan_results = []
        for mapping in mappings:
            proxy_etf = mapping["proxy_etf"]
            if proxy_etf in prices.columns:
                p_series = prices[proxy_etf]
                # Cumulative performance for charting
                perf_series = (p_series / p_series.iloc[0]) - 1
                
                scan_results.append({
                    "id": mapping["id"],
                    "sub_industry": mapping["sub_industry"],
                    "proxy_etf": proxy_etf,
                    "proxy_type": mapping["proxy_type"],
                    "parent_industry": mapping["parent_industry"],
                    "related_tickers": mapping["related_tickers"],
                    "m_1m": ret_1m[proxy_etf],
                    "m_3m": ret_3m[proxy_etf],
                    "m_12m": ret_12m[proxy_etf],
                    "score": (ret_1m[proxy_etf] * 0.4 + ret_3m[proxy_etf] * 0.4 + ret_12m[proxy_etf] * 0.2),
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
