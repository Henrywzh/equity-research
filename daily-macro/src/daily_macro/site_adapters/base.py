from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import ArticleDetails, PlacementCandidate


class SiteAdapter(ABC):
    source_site: str
    homepage_url: str

    @abstractmethod
    def parse_homepage(self, html: str) -> list[PlacementCandidate]:
        raise NotImplementedError

    @abstractmethod
    def parse_article(self, html: str, url: str) -> ArticleDetails:
        raise NotImplementedError

    @abstractmethod
    def normalize_url(self, url: str) -> str:
        raise NotImplementedError
