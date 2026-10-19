from __future__ import annotations

from guolaoxing.email_alert import (
    ArticleAlertItem,
    build_email_subject,
    build_html_body,
    build_plain_body,
    detect_membership_from_article_html,
    extract_post_id,
    find_new_post_ids,
    parse_homepage_links,
    parse_rss_feed,
)


def test_extract_post_id() -> None:
    assert extract_post_id("https://guolaoxing.com/archives/84287") == 84287
    assert extract_post_id("https://guolaoxing.com/?p=84269") == 84269
    assert extract_post_id("https://guolaoxing.com/archives/category/gsjj") is None


def test_detect_membership_member_only_and_public() -> None:
    member_html = """
    <div class="bloginner-content-part" id="post-84287">
      <div class="title-data">
        <h2>股市行情与趋势分析周报10/19</h2>
        <ul>
          <li>作者：照见五蕴皆空</li>
          <li>发布时间：10/19/2026</li>
          <li>博文分类：<a href="https://guolaoxing.com/archives/category/gsjj">股市经济</a></li>
        </ul>
      </div>
      <div>
        <h3>🔒 专属内部研究报告</h3>
        <p>本文仅限有效年度会员阅读。请先登录您的账号。</p>
        <a href="https://guolaoxing.com/my-account?redirect_to=https%3A%2F%2Fguolaoxing.com%2Farchives%2F84287">登录会员体系</a>
      </div>
    </div>
    """
    member_item = detect_membership_from_article_html(member_html, post_id=84287)
    assert member_item.title == "股市行情与趋势分析周报10/19"
    assert member_item.url == "https://guolaoxing.com/archives/84287"
    assert member_item.is_member_only is True
    assert member_item.categories == ["股市经济"]
    assert member_item.published_at == "10/19/2026"

    public_html = """
    <div class="bloginner-content-part" id="post-84269">
      <div class="title-data">
        <h2>即将迎来几天极其强劲的反弹。</h2>
        <ul>
          <li>作者：照见五蕴皆空</li>
          <li>发布时间：10/15/2026</li>
          <li>博文分类：<a href="https://guolaoxing.com/archives/category/zjwy">照见微语</a></li>
        </ul>
      </div>
      <p>一个典型的多头陷阱、空头陷阱，是双向的双重陷阱。</p>
    </div>
    """
    public_item = detect_membership_from_article_html(public_html, post_id=84269)
    assert public_item.title == "即将迎来几天极其强劲的反弹。"
    assert public_item.is_member_only is False
    assert public_item.categories == ["照见微语"]


def test_find_new_post_ids_guards_against_old_homepage_links() -> None:
    state = {
        "baseline_max_post_id": 84287,
        "seen_post_ids": [84246, 84269, 84287],
    }
    discovered = {1331, 84269, 84287, 84301}
    assert find_new_post_ids(discovered, state) == [84301]


def test_parse_rss_and_homepage_and_email_formatting() -> None:
    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>股市行情与趋势分析周报10/19</title>
        <link>https://guolaoxing.com/archives/84287</link>
        <pubDate>Sun, 18 Oct 2026 19:44:34 +0000</pubDate>
        <category><![CDATA[股市经济]]></category>
      </item>
    </channel></rss>"""
    parsed_rss = parse_rss_feed(rss_xml)
    assert 84287 in parsed_rss
    assert parsed_rss[84287]["title"] == "股市行情与趋势分析周报10/19"

    home_html = '<a href="https://guolaoxing.com/archives/84269">即将迎来几天极其强劲的反弹。</a>'
    parsed_home = parse_homepage_links(home_html)
    assert parsed_home[84269]["title"] == "即将迎来几天极其强劲的反弹。"

    item = ArticleAlertItem(
        post_id=84287,
        title="股市行情与趋势分析周报10/19",
        url="https://guolaoxing.com/archives/84287",
        is_member_only=True,
        categories=["股市经济"],
        published_at="10/19/2026",
    )
    subject = build_email_subject([item], is_test=False)
    plain = build_plain_body([item], is_test=False)
    html_body = build_html_body([item], is_test=False)
    assert "股市行情与趋势分析周报10/19" in subject
    assert "会员专享" in subject
    assert "https://guolaoxing.com/archives/84287" in plain
    assert "是（仅限年度会员阅读" in plain
    assert "https://guolaoxing.com/archives/84287" in html_body
