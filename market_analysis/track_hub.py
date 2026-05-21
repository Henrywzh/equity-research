import sys
import os
from pathlib import Path

# Fix path for running as a script from repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from market_analysis.data_engine import DataEngine
from market_analysis.regime_monitor import RegimeMonitor, BreadthMonitor
from market_analysis.pairs_tracker import PairsTracker
from market_analysis.momentum_scanner import MomentumScanner
from market_analysis.report_generator import ReportGenerator

def main():
    print("🚀 Initializing Quant Equity Intelligence Tracker...")
    
    # Initialize components
    engine = DataEngine()
    regime_mon = RegimeMonitor(engine)
    breadth_mon = BreadthMonitor(engine)
    pairs_mon = PairsTracker(engine)
    momentum_mon = MomentumScanner(engine)
    reporter = ReportGenerator()

    # Step 0: Pre-fetch all tickers to ensure data consistency and populate cache
    print("📥 Pre-fetching all required data...")
    all_tickers = list(set(
        regime_mon.all_tickers + 
        breadth_mon.all_tickers +
        pairs_mon.all_tickers + 
        [t for t_str in momentum_mon.mapping_df['Tickers'].dropna() for t in t_str.split(',')]
    ))
    all_tickers = [t.strip() for t in all_tickers if t.strip()]
    shared_prices, refresh_meta = engine.refresh_market_data(all_tickers)
    if shared_prices is None or shared_prices.empty:
        raise RuntimeError("Market analysis refresh produced no usable price data.")
    print(f"Cache file: {refresh_meta['cache_file']}")
    print(f"Refresh mode: {refresh_meta['mode']}")
    if refresh_meta["refreshed_tickers"]:
        print(f"Fresh tickers: {', '.join(refresh_meta['refreshed_tickers'])}")
    if refresh_meta["fallback_tickers"]:
        print(f"Cached fallback tickers: {', '.join(refresh_meta['fallback_tickers'])}")
    if refresh_meta["unavailable_tickers"]:
        print(f"Degraded run warning: unavailable tickers {', '.join(refresh_meta['unavailable_tickers'])}")
    print(f"Latest market data date: {refresh_meta['last_date']}")

    # Step 1: Regime Analysis
    print("📊 LEVEL 1: Running Sector Regime Analysis...")
    regime_results = regime_mon.analyze(shared_prices)
    regime_plot = regime_mon.create_plot(regime_results)

    # Run Breadth Analysis
    print("📊 LEVEL 1: Running Market Breadth Analysis...")
    breadth_results = breadth_mon.analyze(shared_prices)
    breadth_plot = breadth_mon.create_plot(breadth_results)

    # Step 2: Pairs Tracking
    print("🔗 LEVEL 2: Tracking Industry Pairs...")
    pairs_results = pairs_mon.analyze(shared_prices)
    
    # Step 3: Momentum Scanning
    print("📈 LEVEL 3: Scanning Sub-Industry Momentum...")
    momentum_results = momentum_mon.analyze(shared_prices)

    # Step 4: Generate Dashboard
    print("📝 Generating HTML Report...")
    plots = {
        "regime": regime_plot,
        "breadth": breadth_plot
    }
    report_path = reporter.generate(
        regime_results, 
        pairs_results, 
        momentum_results, 
        plots
    )

    print(f"\n✅ Analysis Complete! Dashboard available at: {os.path.abspath(report_path)}")
    print(f"Current Regime: {regime_results['regime']} (Z-Score: {regime_results['z_score']})")
    print(f"Current Breadth: {breadth_results['regime']} (Z-Score: {breadth_results['z_score']})")


if __name__ == "__main__":
    main()
