"""Daily Macro scraper package."""

from .analysis import run_analysis
from .notifier import send_analysis_summary_email, send_test_email
from .pipeline import cleanup_old_snapshots, run_scrape, run_smoke
from .site import build_site
from .storage import Storage

__all__ = [
    "Storage",
    "build_site",
    "cleanup_old_snapshots",
    "run_analysis",
    "run_scrape",
    "run_smoke",
    "send_analysis_summary_email",
    "send_test_email",
]
