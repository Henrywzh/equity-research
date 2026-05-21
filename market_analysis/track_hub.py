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
    
    # Initialize components with separate bootstrap periods
    engine_l1 = DataEngine(bootstrap_period="max")
    engine_l23 = DataEngine(bootstrap_period="5y")
    
    regime_mon = RegimeMonitor(engine_l1)
    breadth_mon = BreadthMonitor(engine_l1)
    pairs_mon = PairsTracker(engine_l23)
    momentum_mon = MomentumScanner(engine_l23)
    reporter = ReportGenerator()

    # Step 0: Pre-fetch Level 1 data (max history)
    print("📥 Pre-fetching Level 1 data (max)...")
    tickers_l1 = list(set(regime_mon.all_tickers + breadth_mon.all_tickers))
    tickers_l1 = [t.strip() for t in tickers_l1 if t.strip()]
    prices_l1, meta_l1 = engine_l1.refresh_market_data(tickers_l1)
    if prices_l1 is None or prices_l1.empty:
        raise RuntimeError("Level 1 market analysis refresh produced no usable price data.")
    print(f"Level 1 Cache file: {meta_l1['cache_file']}")
    print(f"Level 1 Refresh mode: {meta_l1['mode']}")
    print(f"Level 1 Latest market data date: {meta_l1['last_date']}")
        
    # Pre-fetch Level 2/3 data (5y history)
    print("📥 Pre-fetching Level 2/3 data (5y)...")
    tickers_l23 = list(set(
        pairs_mon.all_tickers + 
        [t for t_str in momentum_mon.mapping_df['Tickers'].dropna() for t in t_str.split(',')]
    ))
    tickers_l23 = [t.strip() for t in tickers_l23 if t.strip()]
    prices_l23, meta_l23 = engine_l23.refresh_market_data(tickers_l23)
    if prices_l23 is None or prices_l23.empty:
        raise RuntimeError("Level 2/3 market analysis refresh produced no usable price data.")
    print(f"Level 2/3 Cache file: {meta_l23['cache_file']}")
    print(f"Level 2/3 Refresh mode: {meta_l23['mode']}")
    print(f"Level 2/3 Latest market data date: {meta_l23['last_date']}")

    # Step 1: Regime Analysis
    print("📊 LEVEL 1: Running Sector Regime Analysis...")
    regime_results = regime_mon.analyze(prices_l1)
    regime_plot = regime_mon.create_plot(regime_results)

    # Run Breadth Analysis
    print("📊 LEVEL 1: Running Market Breadth Analysis...")
    breadth_results = breadth_mon.analyze(prices_l1)
    breadth_plot = breadth_mon.create_plot(breadth_results)

    # Step 2: Pairs Tracking
    print("🔗 LEVEL 2: Tracking Industry Pairs...")
    pairs_results = pairs_mon.analyze(prices_l23)
    
    # Step 3: Momentum Scanning
    print("📈 LEVEL 3: Scanning Sub-Industry Momentum...")
    momentum_results = momentum_mon.analyze(prices_l23)

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
    print(f"Current Sector Regime: {regime_results['regime']} (Z-Score: {regime_results['z_score']})")
    print(f"Current Market Breadth: {breadth_results['regime']} (Z-Score: {breadth_results['z_score']})")



if __name__ == "__main__":
    main()
