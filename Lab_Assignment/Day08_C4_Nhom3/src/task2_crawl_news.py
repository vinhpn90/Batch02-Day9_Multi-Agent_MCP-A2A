"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import gzip
import json
import re
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://tienphong.vn/nghe-si-dinh-ma-tuy-khoang-trong-sau-nhung-cu-truot-nga-post1845503.tpo",
    "https://tienphong.vn/ket-qua-xet-nghiem-ma-tuy-cua-ca-si-ngoc-son-post1845495.tpo",
    "https://tienphong.vn/tu-ket-qua-xet-nghiem-5-loai-ma-tuy-cua-ngoc-son-va-dong-thai-cua-nhieu-nghe-si-post1845555.tpo",
    "https://tienphong.vn/30-nguoi-lien-quan-vu-ca-si-chi-dan-va-anh-trai-to-chuc-su-dung-ma-tuy-post1771252.tpo",
    "https://tienphong.vn/miu-le-duong-tinh-cung-luc-3-loai-chat-cam-tiem-an-nguy-co-mat-kiem-soat-hanh-vi-post1842611.tpo",
]


class ArticleTextParser(HTMLParser):
    """Extract readable text from common news article tags."""

    BLOCK_TAGS = {"p", "h1", "h2", "h3", "li", "blockquote"}
    SKIP_TAGS = {"script", "style", "noscript", "svg"}

    def __init__(self):
        super().__init__()
        self._capture = False
        self._skip_depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag in self.BLOCK_TAGS:
            self._capture = True

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag in self.BLOCK_TAGS:
            self._capture = False
            self.parts.append("\n")

    def handle_data(self, data):
        if self._capture and not self._skip_depth:
            text = re.sub(r"\s+", " ", data).strip()
            if text:
                self.parts.append(text)

    def get_text(self) -> str:
        text = " ".join(self.parts)
        lines = [line.strip() for line in re.split(r"\s*\n\s*", text) if line.strip()]
        return "\n\n".join(lines)


def _extract_between(pattern: str, html: str) -> str:
    match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
    return unescape(match.group(1)).strip() if match else ""


def _strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _extract_article(html: str, url: str) -> dict:
    title = (
        _extract_between(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html)
        or _strip_tags(_extract_between(r"<h1[^>]*>(.*?)</h1>", html))
        or _strip_tags(_extract_between(r"<title[^>]*>(.*?)</title>", html))
        or "Unknown"
    )
    published_at = (
        _extract_between(r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']', html)
        or _extract_between(r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2})", html)
    )

    description = (
        _extract_between(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html)
        or _extract_between(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']', html)
    )
    article_html = (
        _extract_between(
            r'<div[^>]+class=["\'][^"\']*cms-body[^"\']*["\'][^>]*>(.*?)(?:<div class=["\']article-footer|<div class=["\']related-news|</article>)',
            html,
        )
        or _extract_between(r"<article[^>]*>(.*?)</article>", html)
        or html
    )
    parser = ArticleTextParser()
    parser.feed(article_html)
    body = parser.get_text()
    content = "\n\n".join(part for part in [description, body] if part)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "published_at": published_at,
        "content_markdown": f"# {title}\n\n{content}",
    }


async def _crawl_with_crawl4ai(url: str) -> dict:
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        metadata = getattr(result, "metadata", {}) or {}
        return {
            "url": url,
            "title": metadata.get("title", "Unknown"),
            "date_crawled": datetime.now().isoformat(timespec="seconds"),
            "published_at": metadata.get("published_time") or metadata.get("date"),
            "content_markdown": result.markdown,
        }


async def _crawl_with_stdlib(url: str) -> dict:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
            ),
            "Accept-Encoding": "gzip, identity",
        },
    )
    with urlopen(request, timeout=30) as response:
        raw_html = response.read()
        if response.headers.get("Content-Encoding", "").lower() == "gzip":
            raw_html = gzip.decompress(raw_html)
        html = raw_html.decode("utf-8", errors="replace")
    return _extract_article(html, url)


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    try:
        return await _crawl_with_crawl4ai(url)
    except ImportError:
        return await _crawl_with_stdlib(url)


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2))
        print(f"  Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("Hay dien ARTICLE_URLS truoc khi chay!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
