# PROJECT_MAP

## 📅 Daily Progress
- Integrated the quant dashboard into the hub flow: [`market_analysis/track_hub.py`](/Users/henrywzh/Desktop/Quant/equity-research/market_analysis/track_hub.py) now orchestrates regime, pairs, and momentum analysis into [`hub/market-analysis.html`](/Users/henrywzh/Desktop/Quant/equity-research/hub/market-analysis.html), while [`hub/index.html`](/Users/henrywzh/Desktop/Quant/equity-research/hub/index.html) links that page from the Market Momentum card.
- Hardened macro event coverage: [`daily-macro/src/daily_macro/release_calendar.py`](/Users/henrywzh/Desktop/Quant/equity-research/daily-macro/src/daily_macro/release_calendar.py) merges FRED releases with scraped FOMC calendar events, and [`daily-macro/src/daily_macro/nowcasting.py`](/Users/henrywzh/Desktop/Quant/equity-research/daily-macro/src/daily_macro/nowcasting.py) expands Cleveland Fed nowcasts plus GDPNow into `macro.sqlite`.
- Refreshed the live data surfaces that feed the hub: `daily-market` added new `polymarket_runs` plus snapshot summaries, `marine-traffic-monitor` refreshed the Hormuz traffic log, `youtube-intake` added Goldman Sachs analysis artifacts, and [`hub/aggregator.py`](/Users/henrywzh/Desktop/Quant/equity-research/hub/aggregator.py) continued publishing [`hub/data/signals.json`](/Users/henrywzh/Desktop/Quant/equity-research/hub/data/signals.json), [`hub/data/hormuz.json`](/Users/henrywzh/Desktop/Quant/equity-research/hub/data/hormuz.json), and [`hub/data/polymarket.json`](/Users/henrywzh/Desktop/Quant/equity-research/hub/data/polymarket.json).

## 🏗️ System Architecture
```mermaid
graph TD
    MarineCLI["marine-traffic-monitor/run.py + run_api.py"] --> MarineData["marine-traffic-monitor/data/hormuz_traffic_log.csv"]
    MacroCLI["daily-macro/src/daily_macro/cli.py"] --> ReleaseCal["daily_macro/release_calendar.py"]
    MacroCLI --> Nowcast["daily_macro/nowcasting.py"]
    ReleaseCal --> MacroDB["daily-macro/data/macro.sqlite"]
    Nowcast --> MacroDB
    MacroCLI --> MacroAnalysis["daily-macro/data/analyses/*.json"]

    MarketCLI["daily-market/src/daily_market/cli.py"] --> MarketPipe["fetcher.py + formatter.py + storage.py"]
    MarketPipe --> MarketDB["daily-market/data/market.sqlite"]
    MarketPipe --> PolyRuns["daily-market/data/polymarket_runs/*.json"]
    MarketPipe --> MarketSummaries["daily-market/data/snapshots/*.json + summaries/*.json"]

    YTCLI["youtube-intake/src/youtube_intake/cli.py"] --> YTPipe["pipeline.py + analyst.py + youtube_client.py"]
    YTPipe --> YTData["youtube-intake/data/analysis/*.json"]

    QuantRun["market_analysis/track_hub.py"] --> DataEngine["market_analysis/data_engine.py"]
    QuantRun --> Regime["market_analysis/regime_monitor.py"]
    QuantRun --> Pairs["market_analysis/pairs_tracker.py"]
    QuantRun --> Momentum["market_analysis/momentum_scanner.py"]
    DataEngine --> ETFCache["data/cache/etf_data_2y.pkl"]
    Regime --> ReportGen["market_analysis/report_generator.py"]
    Pairs --> ReportGen
    Momentum --> ReportGen
    ReportGen --> QuantHTML["hub/market-analysis.html"]

    MarineData --> Aggregator["hub/aggregator.py"]
    MacroDB --> Aggregator
    MacroAnalysis --> Aggregator
    MarketDB --> Aggregator
    PolyRuns --> Aggregator
    MarketSummaries --> Aggregator
    YTData --> Aggregator

    Aggregator --> HubData["hub/data/signals.json + hormuz.json + polymarket.json"]
    HubData --> HubUI["hub/index.html"]
    QuantHTML --> HubUI
```

## 🧠 Context Memo
- The FOMC merge in [`daily-macro/src/daily_macro/release_calendar.py`](/Users/henrywzh/Desktop/Quant/equity-research/daily-macro/src/daily_macro/release_calendar.py) exists because FRED release metadata alone misses Fed meeting structure. Pulling the Federal Reserve calendar into the same digest keeps warning logic aligned with actual statement days instead of treating macro risk as a pure release-ID problem.
- The nowcast expansion in [`daily-macro/src/daily_macro/nowcasting.py`](/Users/henrywzh/Desktop/Quant/equity-research/daily-macro/src/daily_macro/nowcasting.py) is about making `macro.sqlite` a reusable store, not a one-off fetch. Cleveland Fed month/quarter/year variants and GDPNow are normalized into the same schema so the hub and future macro tooling can read one table instead of scraping multiple sites on demand.
- [`market_analysis/track_hub.py`](/Users/henrywzh/Desktop/Quant/equity-research/market_analysis/track_hub.py) pre-fetches the full ticker universe before running regime, pairs, and momentum modules so all three layers operate on the same cached price window. That avoids inconsistent calculations where one module sees fresher bars or a partial download and another does not.
- The hub integration is intentionally loose-coupled: [`hub/aggregator.py`](/Users/henrywzh/Desktop/Quant/equity-research/hub/aggregator.py) still publishes compact JSON payloads for the command-center view, while the quant lane renders a separate HTML page. That keeps the new analysis stack shippable without forcing the main hub to absorb heavy charting logic or a larger client-side data contract yet.

## 🔗 Obsidian Links
- No new `.md` files were created in the last 24 hours.
