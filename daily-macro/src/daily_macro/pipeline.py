from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import DEFAULT_RETENTION_DAYS, get_article_backup_dir, get_data_dir, get_db_path
from .http import build_session, fetch_text
from .models import ArticleDetails, PlacementCandidate
from .site_adapters import HkejAdapter
from .storage import Storage

LOGGER = logging.getLogger(__name__)


def run_scrape(
    *,
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
    adapter: HkejAdapter | None = None,
) -> dict[str, int | str | list[str]]:
    adapter = adapter or HkejAdapter()
    resolved_data_dir = get_data_dir(data_dir)
    resolved_db_path = get_db_path(db_path, resolved_data_dir)
    article_backup_dir = get_article_backup_dir(resolved_data_dir)
    resolved_data_dir.mkdir(parents=True, exist_ok=True)
    article_backup_dir.mkdir(parents=True, exist_ok=True)

    storage = Storage(resolved_db_path)
    session = build_session()
    started_at = utc_now()
    run_id = storage.start_run(started_at)
    errors: list[str] = []
    article_count = 0
    placement_count = 0

    try:
        homepage_html = fetch_text(session, adapter.homepage_url)
        placements = adapter.parse_homepage(homepage_html)
        grouped = _group_placements_by_url(placements)

        for url, related_placements in grouped.items():
            try:
                article_html = fetch_text(session, url)
                article = adapter.parse_article(article_html, url)
                article.summary_snippet = _best_summary_snippet(related_placements)
                article_id = storage.upsert_article(article, started_at)
                backup_payload = _build_article_backup_payload(
                    article=article,
                    placements=related_placements,
                    scraped_at=started_at,
                )
                article_backup_path = _write_article_backup(
                    article_backup_dir=article_backup_dir,
                    run_id=run_id,
                    article=article,
                    payload=backup_payload,
                    scraped_at=started_at,
                )
                storage.record_backup(
                    run_id=run_id,
                    backup_type="parsed_article_json",
                    relative_path=article_backup_path,
                    content_hash=_sha256_json(backup_payload),
                    scraped_at=started_at,
                    article_id=article_id,
                )
                for placement in related_placements:
                    storage.record_placement(article_id, run_id, placement, started_at)
                    placement_count += 1
                article_count += 1
            except Exception as exc:  # pragma: no cover - handled in tests via summary
                LOGGER.exception("Failed to scrape article %s", url)
                errors.append(f"{url}: {exc}")

        status = "success" if not errors else "partial_success"
        finished_at = utc_now()
        storage.finish_run(
            run_id=run_id,
            finished_at=finished_at,
            status=status,
            article_count=article_count,
            placement_count=placement_count,
            error_summary=errors,
        )
        return {
            "run_id": run_id,
            "status": status,
            "article_count": article_count,
            "placement_count": placement_count,
            "errors": errors,
        }
    except Exception as exc:
        finished_at = utc_now()
        errors.append(str(exc))
        storage.finish_run(
            run_id=run_id,
            finished_at=finished_at,
            status="failed",
            article_count=article_count,
            placement_count=placement_count,
            error_summary=errors,
        )
        raise
    finally:
        storage.close()
        session.close()


def run_smoke(adapter: HkejAdapter | None = None) -> dict[str, object]:
    adapter = adapter or HkejAdapter()
    session = build_session()
    try:
        homepage_html = fetch_text(session, adapter.homepage_url)
        placements = adapter.parse_homepage(homepage_html)
        head_news = [item for item in placements if item.collection == "head_news"]
        latest = [item for item in placements if item.collection == "latest"]
        if len(head_news) != 5:
            raise ValueError(f"Expected 5 head news items, found {len(head_news)}")
        if not latest:
            raise ValueError("No latest items were found on the homepage.")
        return {
            "head_news_count": len(head_news),
            "latest_count": len(latest),
            "head_titles": [item.title for item in head_news],
            "latest_first_title": latest[0].title,
        }
    finally:
        session.close()


def cleanup_old_snapshots(
    *,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    data_dir: str | Path | None = None,
    db_path: str | Path | None = None,
) -> dict[str, int]:
    resolved_data_dir = get_data_dir(data_dir)
    resolved_db_path = get_db_path(db_path, resolved_data_dir)
    article_backup_dir = get_article_backup_dir(resolved_data_dir)
    storage = Storage(resolved_db_path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_iso = cutoff.isoformat()

    try:
        rows = storage.get_backups_older_than(cutoff_iso)
        deleted_files = 0
        deleted_rows: list[int] = []
        for row in rows:
            backup_path = article_backup_dir / row["relative_path"]
            if backup_path.exists():
                backup_path.unlink()
                deleted_files += 1
            deleted_rows.append(int(row["id"]))
        storage.delete_backups(deleted_rows)
        _remove_empty_dirs(article_backup_dir)
        return {"deleted_backup_rows": len(deleted_rows), "deleted_files": deleted_files}
    finally:
        storage.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _group_placements_by_url(placements: list[PlacementCandidate]) -> "OrderedDict[str, list[PlacementCandidate]]":
    grouped: OrderedDict[str, list[PlacementCandidate]] = OrderedDict()
    for placement in placements:
        grouped.setdefault(placement.url, []).append(placement)
    return grouped


def _best_summary_snippet(placements: list[PlacementCandidate]) -> str | None:
    for placement in placements:
        if placement.summary_snippet:
            return placement.summary_snippet
    return None


def _article_backup_label(article: ArticleDetails) -> str:
    if article.source_article_id:
        return article.source_article_id
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", article.title).strip("-").lower()
    return slug or "article"


def _write_article_backup(
    *,
    article_backup_dir: Path,
    run_id: int,
    article: ArticleDetails,
    payload: dict[str, object],
    scraped_at: str,
) -> str:
    timestamp = datetime.fromisoformat(scraped_at.replace("Z", "+00:00"))
    run_dir = article_backup_dir / timestamp.strftime("%Y/%m/%d") / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^a-zA-Z0-9._-]+", "-", _article_backup_label(article)).strip("-") or "article"
    file_name = f"article-{safe_label}.json"
    backup_path = run_dir / file_name
    backup_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(backup_path.relative_to(article_backup_dir))


def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sha256_json(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _build_article_backup_payload(
    *,
    article: ArticleDetails,
    placements: list[PlacementCandidate],
    scraped_at: str,
) -> dict[str, object]:
    return {
        "backup_schema_version": 1,
        "scraped_at": scraped_at,
        "canonical_url": article.canonical_url,
        "source_site": article.source_site,
        "source_article_id": article.source_article_id,
        "title": article.title,
        "article_section": article.article_section,
        "published_at": article.published_at,
        "summary_snippet": article.summary_snippet,
        "content_text": article.content_text,
        "content_hash": article.content_hash,
        "placements": [
            {
                "collection": placement.collection,
                "rank": placement.rank,
                "homepage_section": placement.homepage_section,
                "summary_snippet": placement.summary_snippet,
            }
            for placement in placements
        ],
        "parser_metadata": {
            "extraction_method": article.extraction_method,
            "malformed_jsonld_recovered": article.malformed_jsonld_recovered,
        },
    }


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted((candidate for candidate in root.rglob("*") if candidate.is_dir()), reverse=True):
        if any(path.iterdir()):
            continue
        path.rmdir()
