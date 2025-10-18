"""Web crawling tool for fetching and extracting content from URLs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from loguru import logger

from agents import function_tool

DEFAULT_TIMEOUT = 30
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class WebCrawlError(RuntimeError):
    """Raised when web crawling fails."""


@dataclass
class WebCrawlConfig:
    """Configuration for the web crawler."""

    headers: dict[str, str] = None  # type: ignore[assignment]
    timeout: int = DEFAULT_TIMEOUT
    max_content_length: int = 1_000_000  # 1MB max

    def __post_init__(self) -> None:
        merged_headers = dict(DEFAULT_HEADERS)
        if self.headers:
            merged_headers.update(self.headers)
        self.headers = merged_headers


class WebCrawler:
    """Web crawler that fetches and parses HTML content."""

    def __init__(self, config: Optional[WebCrawlConfig] = None) -> None:
        self.config = config or WebCrawlConfig()

    async def crawl(self, url: str) -> dict[str, Any]:
        """Fetch and parse content from a URL.

        Args:
            url: The URL to crawl

        Returns:
            Dictionary containing:
                - url: The URL that was crawled
                - title: Page title
                - text: Main text content
                - headings: List of headings (h1, h2, h3)
                - links: List of links found on the page
                - metadata: Dict of meta tags
        """
        if not url or not url.strip():
            raise ValueError("URL must be a non-empty string.")

        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")

        html = await self._fetch_html(url)
        parsed_content = self._parse_html(html, url)

        return parsed_content

    async def _fetch_html(self, url: str) -> str:
        """Fetch HTML content from URL."""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.get(
                    url,
                    headers=self.config.headers,
                    allow_redirects=True,
                ) as response:
                    response.raise_for_status()

                    # Check content length
                    content_length = response.headers.get("Content-Length")
                    if content_length and int(content_length) > self.config.max_content_length:
                        raise WebCrawlError(
                            f"Content too large: {content_length} bytes "
                            f"(max: {self.config.max_content_length})"
                        )

                    html = await response.text(encoding="utf-8", errors="ignore")

                    # Double-check actual content length
                    if len(html) > self.config.max_content_length:
                        html = html[:self.config.max_content_length]
                        logger.warning(f"Truncated content from {url} to {self.config.max_content_length} chars")

                    return html

            except aiohttp.ClientError as exc:
                logger.error(f"Failed to fetch {url}: {exc}")
                raise WebCrawlError(f"Failed to fetch URL: {exc}") from exc

    def _parse_html(self, html: str, url: str) -> dict[str, Any]:
        """Parse HTML and extract structured content."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        # Extract meta tags
        metadata = {}
        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property")
            content = meta.get("content")
            if name and content:
                metadata[name] = content

        # Extract headings
        headings = []
        for heading_tag in soup.find_all(["h1", "h2", "h3"]):
            heading_text = heading_tag.get_text(strip=True)
            if heading_text:
                headings.append({
                    "level": heading_tag.name,
                    "text": heading_text
                })

        # Extract main text content
        # Prioritize main, article, or body content
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"id": "content"})
            or soup.find("div", {"class": "content"})
            or soup.body
        )

        if main_content:
            # Get text from paragraphs and other text elements
            text_elements = []
            for elem in main_content.find_all(["p", "div", "span", "li", "td", "th"]):
                text = elem.get_text(separator=" ", strip=True)
                if text and len(text) > 10:  # Skip very short snippets
                    text_elements.append(text)

            # Join with newlines and clean up
            text = "\n".join(text_elements)
        else:
            text = soup.get_text(separator="\n", strip=True)

        # Extract links
        links = []
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            link_text = link.get_text(strip=True)
            if href and not href.startswith("#"):  # Skip anchor links
                links.append({
                    "text": link_text,
                    "url": href
                })

        # Limit number of links to avoid huge output
        if len(links) > 100:
            links = links[:100]

        return {
            "url": url,
            "title": title,
            "text": text[:50000],  # Limit text to 50k chars
            "headings": headings[:50],  # Limit headings
            "links": links,
            "metadata": metadata,
        }


@function_tool
async def crawl_website(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Crawl a web page and extract its content.

    This tool fetches HTML from a URL and extracts structured information including
    the page title, main text content, headings, links, and metadata.

    Args:
        url: The URL to crawl and extract content from.
        timeout: Request timeout in seconds (default: 30).

    Returns:
        Dictionary containing:
            - url: The URL that was crawled
            - title: Page title extracted from <title> tag
            - text: Main text content from the page (cleaned of scripts/styles)
            - headings: List of headings with 'level' (h1/h2/h3) and 'text'
            - links: List of links with 'text' and 'url'
            - metadata: Dictionary of meta tags (e.g., description, keywords)

    Example:
        result = await web_crawl("https://example.com/article")
        print(result["title"])
        print(result["text"][:500])  # First 500 chars
    """
    config = WebCrawlConfig(timeout=timeout)
    crawler = WebCrawler(config)
    return await crawler.crawl(url)


__all__ = [
    "crawl_website",
    "WebCrawler",
    "WebCrawlConfig",
    "WebCrawlError",
]
