from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import get_db_path, get_state_dir, get_truthsocial_credentials
from .models import PipelineResult
from .scraper import Scraper
from .storage import Storage

logger = logging.getLogger(__name__)

_CURSOR_FILE = "cursor.json"


def _load_cursor(state_dir: Path) -> str | None:
    path = state_dir / _CURSOR_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data["last_seen_id"]) if data.get("last_seen_id") else None
    except Exception:
        return None


def _save_cursor(state_dir: Path, last_seen_id: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / _CURSOR_FILE
    path.write_text(
        json.dumps(
            {
                "last_seen_id": last_seen_id,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        ),
        encoding="utf-8",
    )


def run_once(
    *,
    db_path: Path | None = None,
    state_dir: Path | None = None,
) -> PipelineResult:
    """Fetch new posts since last cursor, save to SQLite, update cursor. Returns PipelineResult."""
    db_path = db_path or get_db_path()
    state_dir = state_dir or get_state_dir()

    now_iso = datetime.now(timezone.utc).isoformat()
    storage = Storage(db_path)
    run_id = storage.start_run(now_iso)

    try:
        username, password = get_truthsocial_credentials()
        if not username or not password:
            raise RuntimeError(
                "TruthSocial credentials not set. "
                "Add TRUTHSOCIAL_USERNAME and TRUTHSOCIAL_PASSWORD to your .config or environment."
            )

        scraper = Scraper(username, password)
        last_seen_id = _load_cursor(state_dir)
        posts = scraper.fetch_since(last_seen_id)

        if not posts:
            storage.finish_run(run_id, datetime.now(timezone.utc).isoformat(), "empty", 0)
            logger.debug("No new posts.")
            return PipelineResult(status="empty", new_posts=[], new_posts_count=0)

        inserted = storage.insert_posts(posts, seen_at=now_iso)
        newest_id = max(p.post_id for p in posts)
        _save_cursor(state_dir, newest_id)

        storage.finish_run(run_id, datetime.now(timezone.utc).isoformat(), "ok", inserted)
        logger.info("Saved %d new post(s). Newest id: %s", inserted, newest_id)
        return PipelineResult(status="ok", new_posts=posts, new_posts_count=inserted)

    except Exception as exc:
        error_msg = str(exc)
        logger.error("Pipeline error: %s", error_msg)
        storage.finish_run(run_id, datetime.now(timezone.utc).isoformat(), "error", 0, error_msg)
        return PipelineResult(status="error", new_posts=[], new_posts_count=0, error=error_msg)
    finally:
        storage.close()
