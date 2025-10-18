"""Web tools for search and web content retrieval."""

from .search import web_search
from .web_search import search_google
from .crawl import web_crawl

__all__ = [
    "web_search",
    "search_google",
    "web_crawl",
]
