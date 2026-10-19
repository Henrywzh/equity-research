#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import smtplib
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_BASE_URL = "https://guolaoxing.com"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "state" / "seen_posts.json"

GMAIL_SENDER_ENV = "GMAIL_SENDER"
GMAIL_APP_PASSWORD_ENV = "GMAIL_APP_PASSWORD"
GMAIL_RECIPIENT_ENV = "GMAIL_RECIPIENT"

MEMBER_MARKERS = (
    "专属内部研究报告",
    "本文仅限有效年度会员阅读",
    "仅限有效年度会员阅读",
    "登录会员体系",
    "my-account?redirect_to=",
    "post-password-form",
    "受密码保护",
)

ARCHIVE_ID_RE = re.compile(r"https?://(?:www\.)?guolaoxing\.com/archives/(\d+)(?:[/?#]|$)")
GUID_ID_RE = re.compile(r"[?&]p=(\d+)(?:&|$)")


@dataclass
class ArticleAlertItem:
    post_id: int
    title: str
    url: str
    is_member_only: bool
    member_reason: str = ""
    categories: list[str] = field(default_factory=list)
    published_at: str = ""

    @property
    def member_label_zh(self) -> str:
        return "是（仅限年度会员阅读 🔒）" if self.is_member_only else "否（公开免费阅读 🔓）"


def load_local_config(repo_root: Path = REPO_ROOT) -> None:
    config_path = repo_root / ".config"
    if not config_path.exists():
        return
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def build_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
    )
    return session


def fetch_text(url: str, session: requests.Session | None = None, timeout: int = 20) -> str:
    try:
        from scrapling.fetchers import Fetcher  # type: ignore

        resp = Fetcher.get(url, timeout=timeout)
        if getattr(resp, "status", 0) == 200 and getattr(resp, "body", None):
            return resp.body.decode("utf-8", errors="ignore")
    except Exception:
        pass

    own_session = session is None
    http = session or build_http_session()
    try:
        resp = http.get(url, timeout=timeout)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return resp.text
    finally:
        if own_session:
            http.close()


def extract_post_id(url_or_guid: str) -> int | None:
    if not url_or_guid:
        return None
    m = ARCHIVE_ID_RE.search(url_or_guid.strip())
    if m:
        return int(m.group(1))
    m2 = GUID_ID_RE.search(url_or_guid.strip())
    if m2:
        return int(m2.group(1))
    return None


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    soup = BeautifulSoup(unescaped, "html.parser")
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def parse_wp_api_posts(payload: list[dict[str, Any]], base_url: str = DEFAULT_BASE_URL) -> dict[int, dict[str, Any]]:
    discovered: dict[int, dict[str, Any]] = {}
    for item in payload:
        post_id = item.get("id")
        if not isinstance(post_id, int):
            continue
        title = clean_text(item.get("title", {}).get("rendered", ""))
        link = item.get("link") or f"{base_url.rstrip('/')}/archives/{post_id}"
        published_at = item.get("date") or ""
        cats: list[str] = []
        for group in item.get("_embedded", {}).get("wp:term", []):
            if isinstance(group, list):
                for term in group:
                    if isinstance(term, dict) and term.get("taxonomy") == "category" and term.get("name"):
                        cats.append(str(term["name"]).strip())
        discovered[post_id] = {
            "post_id": post_id,
            "title": title,
            "url": link,
            "published_at": published_at,
            "categories": cats,
        }
    return discovered


def parse_rss_feed(xml_text: str, base_url: str = DEFAULT_BASE_URL) -> dict[int, dict[str, Any]]:
    discovered: dict[int, dict[str, Any]] = {}
    root = ET.fromstring(xml_text)
    for item in root.findall("./channel/item"):
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or "").strip()
        post_id = extract_post_id(link) or extract_post_id(guid)
        if not post_id:
            continue
        title = clean_text(item.findtext("title") or "")
        pub_date = (item.findtext("pubDate") or "").strip()
        cats = [clean_text(c.text or "") for c in item.findall("category") if (c.text or "").strip()]
        canonical_url = f"{base_url.rstrip('/')}/archives/{post_id}"
        discovered[post_id] = {
            "post_id": post_id,
            "title": title,
            "url": canonical_url,
            "published_at": pub_date,
            "categories": cats,
        }
    return discovered


def parse_homepage_links(html_text: str, base_url: str = DEFAULT_BASE_URL) -> dict[int, dict[str, Any]]:
    discovered: dict[int, dict[str, Any]] = {}
    soup = BeautifulSoup(html_text, "html.parser")
    for a_tag in soup.find_all("a", href=True):
        href = urljoin(base_url, a_tag["href"].strip())
        if "/category/" in href or "comment-page" in href or "#" in href:
            continue
        post_id = extract_post_id(href)
        if not post_id:
            continue
        link_text = re.sub(r"\s+", " ", a_tag.get_text(" ", strip=True)).strip()
        if link_text in ("", "更多", "前一页", "后一页"):
            link_text = ""
        existing = discovered.get(post_id)
        canonical_url = f"{base_url.rstrip('/')}/archives/{post_id}"
        if not existing:
            discovered[post_id] = {
                "post_id": post_id,
                "title": link_text,
                "url": canonical_url,
                "published_at": "",
                "categories": [],
            }
        elif not existing["title"] and link_text:
            existing["title"] = link_text
    return discovered


def detect_membership_from_article_html(
    html_text: str,
    post_id: int,
    base_url: str = DEFAULT_BASE_URL,
    fallback_title: str = "",
    fallback_categories: list[str] | None = None,
    fallback_published_at: str = "",
) -> ArticleAlertItem:
    soup = BeautifulSoup(html_text, "html.parser")
    container = soup.select_one(".bloginner-content-part") or soup.select_one("article") or soup

    title = ""
    title_el = container.select_one(".title-data h2") or soup.select_one("h1")
    if title_el:
        title = re.sub(r"\s+", " ", title_el.get_text(" ", strip=True)).strip()
    if not title and soup.title and soup.title.string:
        title = soup.title.string.replace("果老星宗&七政四余", "").strip()
    if not title:
        title = fallback_title or f"文章 #{post_id}"

    categories: list[str] = []
    published_at = fallback_published_at
    for li in container.select(".title-data li"):
        li_text = re.sub(r"\s+", " ", li.get_text(" ", strip=True)).strip()
        if li_text.startswith("发布时间："):
            published_at = li_text.replace("发布时间：", "").strip() or published_at
        elif li_text.startswith("博文分类："):
            cat_links = [
                re.sub(r"\s+", " ", a.get_text(" ", strip=True)).strip()
                for a in li.find_all("a")
                if a.get_text(strip=True)
            ]
            if cat_links:
                categories = cat_links
    if not categories and fallback_categories:
        categories = list(fallback_categories)

    container_html = str(container)
    container_text = re.sub(r"\s+", " ", container.get_text(" ", strip=True)).strip()

    is_member_only = False
    member_reason = "公开正文"
    for marker in MEMBER_MARKERS:
        if marker in container_text or marker in container_html:
            is_member_only = True
            member_reason = "检测到会员专享限制提示（本文仅限有效年度会员阅读）"
            break

    url = f"{base_url.rstrip('/')}/archives/{post_id}"
    return ArticleAlertItem(
        post_id=post_id,
        title=title,
        url=url,
        is_member_only=is_member_only,
        member_reason=member_reason,
        categories=categories,
        published_at=published_at,
    )


def discover_site_posts(
    base_url: str = DEFAULT_BASE_URL,
    session: requests.Session | None = None,
) -> dict[int, dict[str, Any]]:
    own_session = session is None
    http = session or build_http_session()
    merged: dict[int, dict[str, Any]] = {}
    try:
        # 1. WordPress REST API (most structured)
        try:
            wp_url = f"{base_url.rstrip('/')}/wp-json/wp/v2/posts?per_page=25&_embed=1&orderby=date&order=desc"
            wp_raw = fetch_text(wp_url, session=http)
            wp_data = json.loads(wp_raw)
            if isinstance(wp_data, list):
                merged.update(parse_wp_api_posts(wp_data, base_url=base_url))
        except Exception:
            pass

        # 2. RSS Feed
        try:
            rss_url = f"{base_url.rstrip('/')}/feed"
            rss_raw = fetch_text(rss_url, session=http)
            for pid, meta in parse_rss_feed(rss_raw, base_url=base_url).items():
                if pid not in merged:
                    merged[pid] = meta
                else:
                    if not merged[pid].get("title") and meta.get("title"):
                        merged[pid]["title"] = meta["title"]
                    if not merged[pid].get("categories") and meta.get("categories"):
                        merged[pid]["categories"] = meta["categories"]
        except Exception:
            pass

        # 3. Homepage links across all sections
        try:
            home_raw = fetch_text(f"{base_url.rstrip('/')}/", session=http)
            for pid, meta in parse_homepage_links(home_raw, base_url=base_url).items():
                if pid not in merged:
                    merged[pid] = meta
                elif not merged[pid].get("title") and meta.get("title"):
                    merged[pid]["title"] = meta["title"]
        except Exception:
            pass

        if not merged:
            raise RuntimeError(f"Failed to discover any articles from {base_url}")
        return merged
    finally:
        if own_session:
            http.close()


def inspect_article(
    post_id: int,
    meta: dict[str, Any] | None = None,
    base_url: str = DEFAULT_BASE_URL,
    session: requests.Session | None = None,
) -> ArticleAlertItem:
    meta = meta or {}
    url = f"{base_url.rstrip('/')}/archives/{post_id}"
    html_text = fetch_text(url, session=session)
    return detect_membership_from_article_html(
        html_text=html_text,
        post_id=post_id,
        base_url=base_url,
        fallback_title=str(meta.get("title") or ""),
        fallback_categories=list(meta.get("categories") or []),
        fallback_published_at=str(meta.get("published_at") or ""),
    )


def load_state(state_path: Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    if not state_path.exists():
        return {
            "site": DEFAULT_BASE_URL,
            "updated_at_utc": None,
            "baseline_max_post_id": 0,
            "seen_post_ids": [],
            "recent_articles": [],
        }
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any], state_path: Path = DEFAULT_STATE_PATH) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def find_new_post_ids(discovered_ids: set[int], state: dict[str, Any]) -> list[int]:
    seen_ids = {int(x) for x in state.get("seen_post_ids", [])}
    baseline_max = int(state.get("baseline_max_post_id") or 0)
    new_ids = [
        pid
        for pid in discovered_ids
        if pid not in seen_ids and pid > baseline_max
    ]
    return sorted(new_ids, reverse=True)


def build_email_subject(items: list[ArticleAlertItem], is_test: bool = False) -> str:
    prefix = "[果老星宗监控测试]" if is_test else "[果老星宗新文章]"
    if len(items) == 1:
        item = items[0]
        tag = "会员专享" if item.is_member_only else "公开可读"
        return f"{prefix} {item.title} ({tag})"
    first_title = items[0].title if items else "新文章更新"
    return f"{prefix} 发现 {len(items)} 篇新文章：{first_title} 等"


def build_plain_body(items: list[ArticleAlertItem], is_test: bool = False) -> str:
    header = (
        "【果老星宗 (guolaoxing.com) 邮件监控测试】\n以下为当前网站最新文章检测结果："
        if is_test
        else f"【果老星宗 (guolaoxing.com) 新文章提醒】\n共检测到 {len(items)} 篇新发布文章："
    )
    lines = [header, ""]
    for idx, item in enumerate(items, start=1):
        cat_str = "、".join(item.categories) if item.categories else "未分类"
        lines.extend(
            [
                f"{idx}. 标题 (Title): {item.title}",
                f"   链接 (Link): {item.url}",
                f"   是否会员可见: {item.member_label_zh}",
                f"   所属板块: {cat_str}",
                f"   发布时间: {item.published_at or '未知'}",
                "",
            ]
        )
    lines.append("— 自动监控任务 (每 30 分钟检查一次 https://guolaoxing.com/)")
    return "\n".join(lines)


def build_html_body(items: list[ArticleAlertItem], is_test: bool = False) -> str:
    heading = (
        "果老星宗 (guolaoxing.com) 邮件监控测试"
        if is_test
        else f"果老星宗 (guolaoxing.com) 新文章提醒（{len(items)} 篇）"
    )
    subtitle = (
        "以下为当前网站最新文章的实时抓取与会员权限识别结果："
        if is_test
        else "监控任务检测到以下新发布文章："
    )
    cards: list[str] = []
    for item in items:
        badge_bg = "#fef2f2" if item.is_member_only else "#ecfdf5"
        badge_fg = "#b91c1c" if item.is_member_only else "#047857"
        badge_border = "#fecaca" if item.is_member_only else "#a7f3d0"
        cat_str = html.escape("、".join(item.categories) if item.categories else "未分类")
        pub_str = html.escape(item.published_at or "未知")
        title_esc = html.escape(item.title)
        url_esc = html.escape(item.url)
        member_esc = html.escape(item.member_label_zh)
        cards.append(
            f"""
            <div style="border:1px solid #e2e8f0; border-radius:10px; padding:16px 18px; margin-bottom:14px; background:#ffffff;">
              <div style="font-size:17px; font-weight:700; color:#0f172a; margin-bottom:10px;">
                <a href="{url_esc}" style="color:#0f172a; text-decoration:none;">{title_esc}</a>
              </div>
              <table style="border-collapse:collapse; font-size:14px; color:#334155; width:100%;">
                <tr>
                  <td style="padding:4px 0; width:130px; color:#64748b;">是否会员才能看到：</td>
                  <td style="padding:4px 0;">
                    <span style="display:inline-block; padding:2px 10px; border-radius:999px; font-weight:600; font-size:13px; background:{badge_bg}; color:{badge_fg}; border:1px solid {badge_border};">
                      {member_esc}
                    </span>
                  </td>
                </tr>
                <tr>
                  <td style="padding:4px 0; color:#64748b;">文章链接：</td>
                  <td style="padding:4px 0;"><a href="{url_esc}" style="color:#2563eb;">{url_esc}</a></td>
                </tr>
                <tr>
                  <td style="padding:4px 0; color:#64748b;">所属板块：</td>
                  <td style="padding:4px 0;">{cat_str}</td>
                </tr>
                <tr>
                  <td style="padding:4px 0; color:#64748b;">发布时间：</td>
                  <td style="padding:4px 0;">{pub_str}</td>
                </tr>
              </table>
            </div>
            """
        )
    cards_html = "\n".join(cards)
    return f"""<!DOCTYPE html>
<html>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'PingFang SC','Microsoft YaHei',sans-serif; background:#f8fafc; padding:20px; color:#0f172a;">
  <div style="max-width:640px; margin:0 auto; background:#f8fafc;">
    <h2 style="margin:0 0 8px 0; font-size:20px; color:#0f172a;">{html.escape(heading)}</h2>
    <p style="margin:0 0 16px 0; font-size:14px; color:#475569;">{html.escape(subtitle)}</p>
    {cards_html}
    <p style="margin-top:18px; font-size:12px; color:#94a3b8;">
      自动监控任务 · 每 30 分钟检查一次 <a href="https://guolaoxing.com/" style="color:#64748b;">https://guolaoxing.com/</a>
    </p>
  </div>
</body>
</html>"""


def get_gmail_credentials() -> tuple[str, str, str]:
    load_local_config()
    sender = os.getenv(GMAIL_SENDER_ENV, "").strip()
    app_password = os.getenv(GMAIL_APP_PASSWORD_ENV, "").strip()
    recipient = os.getenv(GMAIL_RECIPIENT_ENV, "").strip() or sender
    if not sender or not app_password or not recipient:
        raise RuntimeError(
            f"Missing Gmail credentials. Please set {GMAIL_SENDER_ENV}, "
            f"{GMAIL_APP_PASSWORD_ENV}, and {GMAIL_RECIPIENT_ENV}."
        )
    return sender, app_password, recipient


def send_alert_email(items: list[ArticleAlertItem], is_test: bool = False) -> tuple[bool, str]:
    if not items:
        return False, "No articles to email."
    sender, app_password, recipient = get_gmail_credentials()

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = build_email_subject(items, is_test=is_test)
    msg.attach(MIMEText(build_plain_body(items, is_test=is_test), "plain", "utf-8"))
    msg.attach(MIMEText(build_html_body(items, is_test=is_test), "html", "utf-8"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
        server.login(sender, app_password)
        server.sendmail(sender, [recipient], msg.as_string())

    return True, f"Sent email alert ({len(items)} article(s)) to {recipient}."


def run_monitor(
    base_url: str = DEFAULT_BASE_URL,
    state_path: Path = DEFAULT_STATE_PATH,
    init_baseline: bool = False,
    send_latest_test: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    session = build_http_session()
    try:
        discovered = discover_site_posts(base_url=base_url, session=session)
        all_ids = sorted(discovered.keys(), reverse=True)
        max_id = max(all_ids) if all_ids else 0
        state = load_state(state_path)

        is_uninitialized = not state.get("seen_post_ids") and not state.get("baseline_max_post_id")
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        if init_baseline or is_uninitialized:
            sample_ids = all_ids[:5]
            recent_items = [
                inspect_article(pid, discovered.get(pid), base_url=base_url, session=session)
                for pid in sample_ids
            ]
            new_state = {
                "site": base_url,
                "updated_at_utc": now_utc,
                "baseline_max_post_id": max_id,
                "seen_post_ids": sorted(all_ids),
                "recent_articles": [asdict(item) for item in recent_items],
            }
            save_state(new_state, state_path)
            email_sent = False
            email_msg = "Baseline initialized without sending email."
            if send_latest_test and recent_items and not dry_run:
                test_items = recent_items[:2]
                email_sent, email_msg = send_alert_email(test_items, is_test=True)
            return {
                "mode": "init_baseline",
                "discovered_count": len(all_ids),
                "baseline_max_post_id": max_id,
                "new_articles": [asdict(i) for i in recent_items[:2]] if send_latest_test else [],
                "email_sent": email_sent,
                "email_message": email_msg,
            }

        new_ids = find_new_post_ids(set(all_ids), state)
        new_items: list[ArticleAlertItem] = []
        for pid in new_ids:
            new_items.append(
                inspect_article(pid, discovered.get(pid), base_url=base_url, session=session)
            )

        email_sent = False
        email_msg = "No new articles."
        if new_items:
            if dry_run:
                email_msg = f"Dry run: would send alert for {len(new_items)} new article(s)."
            else:
                email_sent, email_msg = send_alert_email(new_items, is_test=False)

            seen_union = sorted({int(x) for x in state.get("seen_post_ids", [])} | set(all_ids))
            prev_recent = list(state.get("recent_articles", []))
            updated_recent = ([asdict(item) for item in new_items] + prev_recent)[:20]
            updated_state = {
                "site": base_url,
                "updated_at_utc": now_utc,
                "baseline_max_post_id": max(int(state.get("baseline_max_post_id") or 0), max_id),
                "seen_post_ids": seen_union,
                "recent_articles": updated_recent,
            }
            save_state(updated_state, state_path)
        elif send_latest_test:
            test_ids = all_ids[:2]
            test_items = [
                inspect_article(pid, discovered.get(pid), base_url=base_url, session=session)
                for pid in test_ids
            ]
            if not dry_run:
                email_sent, email_msg = send_alert_email(test_items, is_test=True)
            return {
                "mode": "test_email",
                "discovered_count": len(all_ids),
                "baseline_max_post_id": state.get("baseline_max_post_id"),
                "new_articles": [asdict(i) for i in test_items],
                "email_sent": email_sent,
                "email_message": email_msg,
            }

        return {
            "mode": "check",
            "discovered_count": len(all_ids),
            "baseline_max_post_id": state.get("baseline_max_post_id"),
            "new_articles": [asdict(i) for i in new_items],
            "email_sent": email_sent,
            "email_message": email_msg,
        }
    finally:
        session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check https://guolaoxing.com/ for new articles and send Gmail alerts."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument(
        "--init-baseline",
        action="store_true",
        help="Initialize or reset the baseline state file with currently published articles.",
    )
    parser.add_argument(
        "--send-latest-test",
        action="store_true",
        help="Send a test alert email for the latest articles on the site.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check for new articles without sending emails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_monitor(
        base_url=args.base_url,
        state_path=args.state_path,
        init_baseline=args.init_baseline,
        send_latest_test=args.send_latest_test,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
