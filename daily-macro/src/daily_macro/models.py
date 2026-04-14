from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlacementCandidate:
    collection: str
    rank: int
    title: str
    url: str
    homepage_section: str | None = None
    summary_snippet: str | None = None


@dataclass
class LatestPageSnapshot:
    active_title: str | None
    items: list[PlacementCandidate]


@dataclass
class ArticleDetails:
    canonical_url: str
    title: str
    source_site: str
    source_article_id: str | None
    article_section: str | None
    published_at: str | None
    content_text: str
    content_hash: str
    summary_snippet: str | None = None
    extraction_method: str = "fallback_html"
    malformed_jsonld_recovered: bool = False
