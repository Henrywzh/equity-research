import sys
import os
from pathlib import Path

# Fix path for running as a script from repo root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from market_analysis.data_engine import DataEngine
from market_analysis.regime_monitor import RegimeMonitor
from market_analysis.pairs_tracker import PairsTracker
from market_analysis.momentum_scanner import MomentumScanner
from market_analysis.report_generator import ReportGenerator

def main():
    print("🚀 Initializing Quant Equity Intelligence Tracker...")
    
    # Initialize components
    engine = DataEngine()
    regime_mon = RegimeMonitor(engine)
    pairs_mon = PairsTracker(engine)
    momentum_mon = MomentumScanner(engine)
    reporter = ReportGenerator()

    # Step 0: Pre-fetch all tickers to ensure data consistency and populate cache
    print("📥 Pre-fetching all required data...")
    all_tickers = list(set(
        regime_mon.all_tickers + 
        pairs_mon.all_tickers + 
        [t for t_str in momentum_mon.mapping_df['Tickers'].dropna() for t in t_str.split(',')]
    ))
    all_tickers = [t.strip() for t in all_tickers if t.strip()]
    engine.fetch_data(all_tickers, period="2y", force_refresh=True)

    # Step 1: Regime Analysis
    print("📊 LEVEL 1: Running Regime Analysis...")
    regime_results = regime_mon.analyze()
    regime_plot = regime_mon.create_plot(regime_results)

    # Step 2: Pairs Tracking
    print("🔗 LEVEL 2: Tracking Industry Pairs...")
    pairs_results = pairs_mon.analyze()
    
    # Step 3: Momentum Scanning
    print("📈 LEVEL 3: Scanning Sub-Industry Momentum...")
    momentum_results = momentum_mon.analyze()

    # Step 4: Generate Dashboard
    print("📝 Generating HTML Report...")
    plots = {"regime": regime_plot}
    report_path = reporter.generate(
        regime_results, 
        pairs_results, 
        momentum_results, 
        plots
    )

    print(f"\n✅ Analysis Complete! Dashboard available at: {os.path.abspath(report_path)}")
    print(f"Current Regime: {regime_results['regime']} (Z-Score: {regime_results['z_score']})")

if __name__ == "__main__":
    main()
