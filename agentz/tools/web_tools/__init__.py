"""Web tools for search and web content retrieval."""

from .search import web_search
from .crawl import crawl_website

__all__ = [
    "web_search",
    "crawl_website",
]
