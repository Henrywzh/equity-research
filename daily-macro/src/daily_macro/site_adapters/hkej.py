from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from ..config import DEFAULT_HOMEPAGE_URL, DEFAULT_SOURCE_SITE
from ..models import ArticleDetails, LatestPageSnapshot, PlacementCandidate
from .base import SiteAdapter

BOILERPLATE_SNIPPETS = (
    "《信報》印刷版出報日為星期一至六",
    "其餘日子照常出刊",
    "網上版及流動版於休刊日將如常更新",
)


class HkejAdapter(SiteAdapter):
    source_site = DEFAULT_SOURCE_SITE
    homepage_url = DEFAULT_HOMEPAGE_URL

    def normalize_url(self, url: str) -> str:
        absolute = urljoin(self.homepage_url, url)
        parsed = urlparse(absolute)
        return parsed._replace(fragment="").geturl()

    def parse_homepage(self, html: str) -> list[PlacementCandidate]:
        soup = BeautifulSoup(html, "html.parser")
        placements: list[PlacementCandidate] = []
        seen_pairs: set[tuple[str, str]] = set()

        hero = self._parse_hero_story(soup)
        if hero is not None:
            pair = (hero.collection, hero.url)
            seen_pairs.add(pair)
            placements.append(hero)

        featured_rank = 2
        for anchor in soup.select(".hkej_hl-news_topic_2014 a[href]"):
            candidate = self._build_featured_candidate(anchor, featured_rank)
            if candidate is None:
                continue
            pair = (candidate.collection, candidate.url)
            if pair in seen_pairs:
                continue
            placements.append(candidate)
            seen_pairs.add(pair)
            featured_rank += 1
            if featured_rank > 5:
                break

        latest_snapshot = self.parse_latest_page(html, start_rank=1)
        for candidate in latest_snapshot.items:
            pair = (candidate.collection, candidate.url)
            if pair in seen_pairs:
                continue
            placements.append(candidate)
            seen_pairs.add(pair)

        return placements

    def parse_latest_page(self, html: str, start_rank: int = 1) -> LatestPageSnapshot:
        soup = BeautifulSoup(html, "html.parser")
        active_title = self._extract_active_listing_title(soup)
        latest_items = soup.select("div.hkej_toc_listingAll_news2_2014")
        if not latest_items:
            raise ValueError("Could not locate latest news cards on the page.")

        candidates: list[PlacementCandidate] = []
        current_rank = start_rank
        for item in latest_items:
            candidate = self._build_latest_candidate(item, current_rank)
            if candidate is None:
                continue
            candidates.append(candidate)
            current_rank += 1

        return LatestPageSnapshot(active_title=active_title, items=candidates)

    def parse_article(self, html: str, url: str) -> ArticleDetails:
        soup = BeautifulSoup(html, "html.parser")

        news_article, malformed_jsonld_recovered = self._extract_news_article_jsonld(soup)
        canonical_url = self._extract_canonical_url(soup, url)
        source_article_id = self._extract_source_article_id(canonical_url)

        if news_article is not None:
            title = self._clean_inline_text(str(news_article.get("headline", "")))
            article_section = self._clean_inline_text(str(news_article.get("articleSection", ""))) or None
            published_at = self._normalize_datetime(news_article.get("datePublished"))
            content_text = self._clean_article_body(str(news_article.get("articleBody", "")))
            extraction_method = "jsonld"
        else:
            title = self._clean_inline_text(self._safe_text(soup.select_one("h1")))
            article_section = self._extract_fallback_section(soup, canonical_url)
            published_at = self._extract_fallback_published_at(soup)
            content_text = self._extract_fallback_content(soup)
            extraction_method = "fallback_html"
            malformed_jsonld_recovered = False

        if not title:
            raise ValueError(f"Could not extract article title from {url}")
        if not content_text:
            raise ValueError(f"Could not extract article body from {url}")

        return ArticleDetails(
            canonical_url=canonical_url,
            title=title,
            source_site=self.source_site,
            source_article_id=source_article_id,
            article_section=article_section,
            published_at=published_at,
            content_text=content_text,
            content_hash=hashlib.sha256(content_text.encode("utf-8")).hexdigest(),
            extraction_method=extraction_method,
            malformed_jsonld_recovered=malformed_jsonld_recovered,
        )

    def _parse_hero_story(self, soup: BeautifulSoup) -> PlacementCandidate | None:
        container = soup.select_one(".in_news_u_2015")
        if container is None:
            return None

        anchor = container.select_one("h3 a[href]")
        if anchor is None:
            return None

        return PlacementCandidate(
            collection="head_news",
            rank=1,
            title=self._clean_inline_text(anchor.get_text(" ", strip=True)),
            url=self.normalize_url(anchor["href"]),
            homepage_section=self._safe_text(container.select_one(".in_news_u_cat a")),
        )

    def _build_featured_candidate(self, anchor: Tag, rank: int) -> PlacementCandidate | None:
        title = self._clean_inline_text(anchor.get_text(" ", strip=True))
        href = anchor.get("href")
        if not title or not href:
            return None

        item = anchor.find_parent("li")
        section = None
        if item is not None:
            section = self._safe_text(item.select_one(".hkej_hl-news_section_2014 a"))

        return PlacementCandidate(
            collection="head_news",
            rank=rank,
            title=title,
            url=self.normalize_url(href),
            homepage_section=section,
        )

    def _build_latest_candidate(self, item: Tag, rank: int) -> PlacementCandidate | None:
        anchor = item.select_one("h3 a[href]")
        if anchor is None:
            return None

        title = self._clean_inline_text(anchor.get_text(" ", strip=True))
        href = anchor.get("href")
        if not title or not href:
            return None

        snippet = self._safe_text(item.select_one(".hkej_toc_listingAll_news2_recap_2014"))
        snippet = snippet.replace("全文", "").strip() if snippet else None
        section = self._safe_text(item.select_one(".hkej_toc_top2_2014 a"))

        return PlacementCandidate(
            collection="latest",
            rank=rank,
            title=title,
            url=self.normalize_url(href),
            homepage_section=section,
            summary_snippet=snippet,
        )

    def _extract_active_listing_title(self, soup: BeautifulSoup) -> str | None:
        node = soup.select_one(
            "ul.hkej_online-news-menu_2014 li.hkej_onewsCat_2014_on span.listing_new_title"
        )
        return self._safe_text(node)

    def _extract_news_article_jsonld(self, soup: BeautifulSoup) -> tuple[dict | None, bool]:
        for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
            raw = script.string or script.get_text()
            if not raw:
                continue
            parsed, malformed_jsonld_recovered = self._parse_json_blob(raw)
            candidate = self._find_news_article(parsed)
            if candidate is not None:
                return candidate, malformed_jsonld_recovered
        return None, False

    def _parse_json_blob(self, raw: str) -> tuple[object | None, bool]:
        try:
            return json.loads(raw), False
        except json.JSONDecodeError:
            return self._extract_news_article_via_regex(raw), True

    def _extract_news_article_via_regex(self, raw: str) -> dict | None:
        if '"NewsArticle"' not in raw:
            return None

        article: dict[str, str] = {"@type": "NewsArticle"}
        for field in ("headline", "articleBody", "articleSection", "datePublished"):
            value = self._extract_json_like_value(raw, field)
            if value:
                article[field] = value
        return article if "headline" in article and "articleBody" in article else None

    def _extract_json_like_value(self, raw: str, field: str) -> str | None:
        quoted = re.search(
            rf'"{field}"\s*:\s*("(?:(?:\\.)|[^"\\])*")',
            raw,
            flags=re.S,
        )
        if quoted:
            return self._decode_relaxed_json_string(quoted.group(1))

        unquoted = re.search(
            rf'"{field}"\s*:\s*([^,\n}}]+)',
            raw,
            flags=re.S,
        )
        if unquoted:
            return unquoted.group(1).strip().strip('"')
        return None

    def _decode_relaxed_json_string(self, token: str) -> str:
        if len(token) >= 2 and token[0] == '"' and token[-1] == '"':
            token = token[1:-1]
        return (
            token.replace("\\/", "/")
            .replace('\\"', '"')
            .replace("\\n", "\n")
            .replace("\\r", "\r")
            .replace("\\t", "\t")
        )

    def _find_news_article(self, node: object) -> dict | None:
        if isinstance(node, dict):
            type_value = node.get("@type")
            if type_value == "NewsArticle" or (
                isinstance(type_value, list) and "NewsArticle" in type_value
            ):
                return node
            for value in node.values():
                found = self._find_news_article(value)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = self._find_news_article(item)
                if found is not None:
                    return found
        return None

    def _extract_canonical_url(self, soup: BeautifulSoup, fallback_url: str) -> str:
        canonical = soup.select_one('link[rel="canonical"]')
        href = canonical.get("href") if canonical is not None else fallback_url
        return self.normalize_url(href)

    def _extract_source_article_id(self, url: str) -> str | None:
        match = re.search(r"/article/(\d+)", url)
        return match.group(1) if match else None

    def _extract_fallback_section(self, soup: BeautifulSoup, canonical_url: str) -> str | None:
        meta = soup.select_one('meta[property="article:section"]')
        if meta and meta.get("content"):
            return self._clean_inline_text(meta["content"])

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            if href.startswith("/instantnews/") and "/article/" not in href:
                text = self._clean_inline_text(anchor.get_text(" ", strip=True))
                if text:
                    return text

        parsed = urlparse(canonical_url)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return parts[1]
        return None

    def _extract_fallback_published_at(self, soup: BeautifulSoup) -> str | None:
        meta = soup.select_one('meta[property="article:published_time"]')
        if meta and meta.get("content"):
            return self._normalize_datetime(meta["content"])

        time_node = soup.select_one("time")
        if time_node is not None:
            return self._normalize_datetime(time_node.get("datetime") or time_node.get_text(" ", strip=True))
        return None

    def _extract_fallback_content(self, soup: BeautifulSoup) -> str:
        paragraphs: list[str] = []
        seen: set[str] = set()
        for paragraph in soup.find_all("p"):
            text = self._clean_inline_text(paragraph.get_text(" ", strip=True))
            if not text:
                continue
            if len(text) < 20:
                continue
            if self._is_boilerplate(text):
                continue
            if text in seen:
                continue
            seen.add(text)
            paragraphs.append(text)
        return "\n\n".join(paragraphs)

    def _safe_text(self, node: Tag | None) -> str | None:
        if node is None:
            return None
        text = self._clean_inline_text(node.get_text(" ", strip=True))
        return text or None

    def _clean_inline_text(self, text: str | None) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def _clean_article_body(self, text: str | None) -> str:
        if not text:
            return ""
        pieces = [self._clean_inline_text(piece) for piece in re.split(r"\n+", text)]
        filtered = [piece for piece in pieces if piece and not self._is_boilerplate(piece)]
        return "\n\n".join(filtered)

    def _is_boilerplate(self, text: str) -> bool:
        return any(snippet in text for snippet in BOILERPLATE_SNIPPETS)

    def _normalize_datetime(self, value: str | None) -> str | None:
        if value is None:
            return None
        value = str(value).strip()
        if not value:
            return None
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(value).isoformat()
        except ValueError:
            return value
