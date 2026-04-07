from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TruthPost:
    post_id: str
    created_at: str          # ISO8601
    content: str             # plain text (HTML stripped)
    reblogs_count: int
    favourites_count: int
    replies_count: int
    url: str
    sensitive: bool
    raw_json: str            # full API payload as JSON string (for Phase II NLP)


@dataclass
class PipelineResult:
    status: str              # 'ok' | 'error' | 'empty'
    new_posts: list[TruthPost]
    new_posts_count: int
    error: str | None = None
