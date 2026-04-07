from __future__ import annotations

import json
import logging
import re
from typing import Any

from .models import TruthPost

logger = logging.getLogger(__name__)

_HANDLE = "realDonaldTrump"


def _strip_html(html: str | None) -> str:
    if not html:
        return ""
    return re.sub(r"<[^>]+>", " ", html).strip()


def _parse_post(raw: Any) -> TruthPost:
    """Parse a truthbrush status object (dict or object with attrs) into TruthPost."""
    if isinstance(raw, dict):
        get = raw.get
    else:
        get = lambda key, default=None: getattr(raw, key, default)  # noqa: E731

    return TruthPost(
        post_id=str(get("id", "")),
        created_at=str(get("created_at", "")),
        content=_strip_html(get("content", "")),
        reblogs_count=int(get("reblogs_count") or 0),
        favourites_count=int(get("favourites_count") or 0),
        replies_count=int(get("replies_count") or 0),
        url=str(get("url", "") or ""),
        sensitive=bool(get("sensitive", False)),
        raw_json=json.dumps(raw if isinstance(raw, dict) else vars(raw), default=str, ensure_ascii=False),
    )


class Scraper:
    def __init__(self, username: str, password: str) -> None:
        self._username = username
        self._password = password
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from truthbrush import Api  # type: ignore[import]
            self._client = Api(username=self._username, password=self._password)
        return self._client

    def _reset_client(self) -> None:
        self._client = None

    def fetch_since(self, last_seen_id: str | None) -> list[TruthPost]:
        """Fetch new posts since last_seen_id (exclusive). Returns list sorted ascending by post_id."""
        try:
            return self._fetch(last_seen_id)
        except Exception as exc:
            exc_str = str(exc).lower()
            if any(token in exc_str for token in ("401", "unauthorized", "auth", "login", "credential")):
                logger.warning("Auth error detected (%s) — re-authenticating and retrying once.", exc)
                self._reset_client()
                return self._fetch(last_seen_id)
            raise

    def _fetch(self, last_seen_id: str | None) -> list[TruthPost]:
        client = self._get_client()
        kwargs: dict[str, Any] = {}
        if last_seen_id:
            kwargs["since_id"] = last_seen_id

        raw_posts = list(client.pull_statuses(_HANDLE, **kwargs) or [])
        posts = [_parse_post(p) for p in raw_posts if p]
        # Sort ascending by post_id so newest is last; safe for snowflake IDs (lexicographic == numeric order)
        posts.sort(key=lambda p: p.post_id)
        logger.debug("Fetched %d post(s) since id=%s", len(posts), last_seen_id or "none")
        return posts
