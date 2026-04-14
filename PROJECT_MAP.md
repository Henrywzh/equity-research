# PROJECT_MAP

## 📅 Daily Progress
- Added a new quant analysis lane: `market_analysis/` now provides `DataEngine`, `RegimeMonitor`, `PairsTracker`, `MomentumScanner`, and `ReportGenerator`, with [`track_hub.py`](/Users/henrywzh/Desktop/Quant/equity-research/track_hub.py) orchestrating the run into [`report/dashboard.html`](/Users/henrywzh/Desktop/Quant/equity-research/report/dashboard.html).
- Refactored and hardened macro release handling: [`daily-macro/src/daily_macro/prompts.py`](/Users/henrywzh/Desktop/Quant/equity-research/daily-macro/src/daily_macro/prompts.py) was split out of analysis code, and [`daily-macro/src/daily_macro/release_calendar.py`](/Users/henrywzh/Desktop/Quant/equity-research/daily-macro/src/daily_macro/release_calendar.py) now merges FRED releases with parsed Federal Reserve FOMC events.
- Tightened the hub’s data contract: [`hub/aggregator.py`](/Users/henrywzh/Desktop/Quant/equity-research/hub/aggregator.py) gained QQQ/BTC track-record and richer Polymarket history logic, while current local edits remove fallback mock FRED items and add coverage in [`hub/tests/test_aggregator.py`](/Users/henrywzh/Desktop/Quant/equity-research/hub/tests/test_aggregator.py).

## 🏗️ System Architecture
```mermaid
graph TD
    subgraph GH["GitHub Actions / Scheduled Jobs"]
        WMarine["Hormuz monitor jobs"]
        WMacro["daily-macro scrape + analysis jobs"]
        WMarket["daily-market refresh jobs"]
        WYT["youtube-intake jobs"]
        WHub["hub refresh job"]
        WQuant["local/manual quant run"]
    end

    subgraph Marine["marine-traffic-monitor"]
        MRun["run.py / run_api.py"]
        MAIS["ais_client.py"]
        MAnalyst["analyst.py"]
        MPolicy["policy_engine.py + state_manager.py"]
        MOut["data/hormuz_traffic_log.csv + logs/state/*"]
    end

    subgraph Macro["daily-macro"]
        DMCLI["src/daily_macro/cli.py"]
        DMAnalysis["analysis.py"]
        DMPrompts["prompts.py"]
        DMCalendar["release_calendar.py"]
        DMNotify["notifier.py"]
        DMOut["data/analyses/* + data/news.sqlite"]
    end

    subgraph Market["daily-market"]
        DKCLI["src/daily_market/cli.py"]
        DKPipe["pipeline.py + fetcher.py + formatter.py + storage.py"]
        DKOut["data/summaries/* + data/polymarket_runs/* + market.sqlite"]
    end

    subgraph YT["youtube-intake"]
        YCLI["src/youtube_intake/cli.py"]
        YPipe["pipeline.py + youtube_client.py + analyst.py"]
        YOut["data/analysis/* + state/channels.json"]
    end

    subgraph Quant["market_analysis"]
        QEngine["data_engine.py"]
        QRegime["regime_monitor.py"]
        QPairs["pairs_tracker.py"]
        QMomentum["momentum_scanner.py"]
        QReport["report_generator.py"]
        QRun["track_hub.py"]
        QOut["report/dashboard.html"]
    end

    subgraph Hub["hub"]
        HAgg["aggregator.py"]
        HSignals["data/signals.json + hormuz.json + polymarket.json"]
        HUI["index.html + polymarket.html"]
        HTests["tests/test_aggregator.py"]
    end

    WMarine --> MRun
    WMacro --> DMCLI
    WMarket --> DKCLI
    WYT --> YCLI
    WQuant --> QRun
    WHub --> HAgg

    MRun --> MAIS
    MRun --> MAnalyst
    MAnalyst --> MPolicy
    MPolicy --> MOut

    DMCLI --> DMAnalysis
    DMAnalysis --> DMPrompts
    DMAnalysis --> DMCalendar
    DMCalendar --> DMNotify
    DMAnalysis --> DMOut
    DMCalendar --> DMOut

    DKCLI --> DKPipe
    DKPipe --> DKOut

    YCLI --> YPipe
    YPipe --> YOut

    QRun --> QEngine
    QRun --> QRegime
    QRun --> QPairs
    QRun --> QMomentum
    QRun --> QReport
    QReport --> QOut

    MOut --> HAgg
    DMOut --> HAgg
    DKOut --> HAgg
    YOut --> HAgg
    DMCalendar --> HAgg
    HAgg --> HSignals
    HSignals --> HUI
    HTests --> HAgg
```

## 🧠 Context Memo
- The FOMC work in [`daily-macro/src/daily_macro/release_calendar.py`](/Users/henrywzh/Desktop/Quant/equity-research/daily-macro/src/daily_macro/release_calendar.py) exists because FRED release metadata is not enough for Fed meeting-day coverage. The code now scrapes the Federal Reserve calendar directly, then merges those events into the same digest so macro warnings can distinguish actual statement days from generic release IDs or stale press-release noise.
- The prompt split into [`daily-macro/src/daily_macro/prompts.py`](/Users/henrywzh/Desktop/Quant/equity-research/daily-macro/src/daily_macro/prompts.py) is architectural, not cosmetic. It separates long prompt templates from `analysis.py` so the analysis pipeline can keep evolving without burying business logic inside LLM strings.
- The new quant stack is intentionally isolated in `market_analysis/` and surfaced through [`track_hub.py`](/Users/henrywzh/Desktop/Quant/equity-research/track_hub.py). That keeps cross-sectional regime, pairs, and momentum logic reusable as a standalone dashboard pipeline instead of coupling it directly into the main hub before the signal contract is stable.
- The current local hub edits remove synthetic FRED fallback rows from [`hub/aggregator.py`](/Users/henrywzh/Desktop/Quant/equity-research/hub/aggregator.py) and add tests in [`hub/tests/test_aggregator.py`](/Users/henrywzh/Desktop/Quant/equity-research/hub/tests/test_aggregator.py). The point is to make missing API credentials visible as missing data, not silently replaced with fabricated calendar items that would pollute operational decisions.

## 🔗 Obsidian Links
- No new project `.md` files were created in the last 24 hours.
- Existing note surfaces still map as follows:
  [`README.md`](/Users/henrywzh/Desktop/Quant/equity-research/README.md) covers the repo entry points and operator workflow.
  [`daily-macro/README.md`](/Users/henrywzh/Desktop/Quant/equity-research/daily-macro/README.md) documents the `daily_macro` pipeline and its generated macro artifacts.
  [`daily-market/README.md`](/Users/henrywzh/Desktop/Quant/equity-research/daily-market/README.md) tracks the market snapshot and Polymarket ingestion path that now feeds the hub.
  [`youtube-intake/README.md`](/Users/henrywzh/Desktop/Quant/equity-research/youtube-intake/README.md) maps to the ingestion and analysis pipeline under `youtube-intake/src/youtube_intake/`.
  [`marine-traffic-monitor/PROJECT_README.md`](/Users/henrywzh/Desktop/Quant/equity-research/marine-traffic-monitor/PROJECT_README.md) maps to the Hormuz monitoring code and its CSV/state outputs.
