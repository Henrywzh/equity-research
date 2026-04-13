import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

class DataEngine:
    def __init__(self, cache_dir="data/cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def fetch_data(self, tickers, period="2y", interval="1d", force_refresh=False):
        """
        Fetch historical data for a list of tickers. 
        Uses a simple file-based cache to avoid redundant network calls.
        """
        # Determine if we should use cache
        cache_file = os.path.join(self.cache_dir, f"etf_data_{period}.pkl")
        
        if os.path.exists(cache_file) and not force_refresh:
            # Check if cache is from today
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if mtime.date() == datetime.now().date():
                print(f"Loading data from cache: {cache_file}")
                return pd.read_pickle(cache_file)

        print(f"Fetching data from YFinance for {len(tickers)} tickers...")
        try:
            # Default structure: Level 0 = Price Attribute, Level 1 = Ticker
            data = yf.download(tickers, period=period, interval=interval, progress=False)
            
            if data.empty:
                print("Warning: Downloaded data is empty.")
                return None
                
            # Save to cache
            data.to_pickle(cache_file)
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def get_close_prices(self, data):
        """Extracts Adjusted Close or Close prices from yfinance DataFrame."""
        if data is None or data.empty: 
            print("Warning: No data provided to get_close_prices.")
            return None
        
        # Determine the price column to use
        if isinstance(data.columns, pd.MultiIndex):
            # Level 0 is the attribute (Open, High, ..., Adj Close)
            # Level 1 is the Ticker
            
            # We want to be robust: take Adj Close if it exists, otherwise Close
            if 'Adj Close' in data.columns.get_level_values(0):
                adj_close = data['Adj Close']
            else:
                adj_close = pd.DataFrame()
                
            if 'Close' in data.columns.get_level_values(0):
                close = data['Close']
            else:
                close = pd.DataFrame()
            
            # Merge: Use close as baseline, update with adj_close where not null
            prices = close.copy()
            for col in adj_close.columns:
                if col in prices.columns:
                    # Only fill non-null values to be safe
                    prices[col].update(adj_close[col].dropna())
                else:
                    prices[col] = adj_close[col]
        else:
            # Single ticker case
            print("Extracting from single-level Index...")
            if 'Adj Close' in data.columns:
                prices = data['Adj Close'].to_frame()
            elif 'Close' in data.columns:
                prices = data['Close'].to_frame()
            else:
                prices = data
        
        # Clean up columns
        if isinstance(prices, pd.Series):
             prices = prices.to_frame()
             
        prices.columns = [str(c).strip() for c in prices.columns]
        prices.columns.name = None
        
        print(f"Initial tickers in prices DF: {len(prices.columns)}")
            
        # Standardize: Drop tickers with all NaNs
        valid_prices = prices.dropna(axis=1, how='all')
        missing = set(prices.columns) - set(valid_prices.columns)
        if missing:
            print(f"Warning: Dropped {len(missing)} tickers due to all NaNs.")
            
        print(f"Final valid tickers: {len(valid_prices.columns)}")
        return valid_prices
