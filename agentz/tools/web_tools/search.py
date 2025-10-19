import asyncio
import os
import ssl
from typing import List, Optional, Union

import aiohttp
from agents import function_tool
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()
CONTENT_LENGTH_LIMIT = 10000  # Trim scraped content to this length to avoid large context / token limit issues

# ------- DEFINE TYPES -------


class ScrapeResult(BaseModel):
    url: str = Field(description="The URL of the webpage")
    text: str = Field(description="The full text content of the webpage")
    title: str = Field(description="The title of the webpage")
    description: str = Field(description="A short description of the webpage")


class WebpageSnippet(BaseModel):
    url: str = Field(description="The URL of the webpage")
    title: str = Field(description="The title of the webpage")
    description: Optional[str] = Field(description="A short description of the webpage")


# ------- DEFINE UNDERLYING TOOL LOGIC -------

# Create a shared connector
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
ssl_context.set_ciphers(
    "DEFAULT:@SECLEVEL=1"
)  # Add this line to allow older cipher suites


class SerperClient:
    """A client for the Serper API to perform Google searches."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set SERPER_API_KEY environment variable."
            )

        self.url = "https://google.serper.dev/search"
        self.headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

    async def search(
        self, query: str, filter_for_relevance: bool = True, max_results: int = 5
    ) -> List[WebpageSnippet]:
        """Perform a Google search using Serper API and fetch basic details for top results.

        Args:
            query: The search query
            filter_for_relevance: Unused, kept for API compatibility
            max_results: Maximum number of results to return (max 10)

        Returns:
            List of WebpageSnippet objects
        """
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                self.url, headers=self.headers, json={"q": query, "autocorrect": False}
            ) as response:
                response.raise_for_status()
                results = await response.json()
                results_list = [
                    WebpageSnippet(
                        url=result.get("link", ""),
                        title=result.get("title", ""),
                        description=result.get("snippet", ""),
                    )
                    for result in results.get("organic", [])
                ]

        return results_list[:max_results]


class SearchXNGClient:
    """A client for the SearchXNG API to perform searches."""

    def __init__(self):
        self.host = os.getenv("SEARCHXNG_HOST")
        if not self.host:
            raise ValueError("SEARCHXNG_HOST environment variable not set")
        if not self.host.endswith("/search"):
            self.host = (
                f"{self.host}/search"
                if not self.host.endswith("/")
                else f"{self.host}search"
            )

    async def search(
        self, query: str, filter_for_relevance: bool = True, max_results: int = 5
    ) -> List[WebpageSnippet]:
        """Perform a search using SearchXNG API.

        Args:
            query: The search query
            filter_for_relevance: Unused, kept for API compatibility
            max_results: Maximum number of results to return

        Returns:
            List of WebpageSnippet objects
        """
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            params = {
                "q": query,
                "format": "json",
            }

            async with session.get(self.host, params=params) as response:
                response.raise_for_status()
                results = await response.json()

                results_list = [
                    WebpageSnippet(
                        url=result.get("url", ""),
                        title=result.get("title", ""),
                        description=result.get("content", ""),
                    )
                    for result in results.get("results", [])
                ]

        return results_list[:max_results]


async def scrape_urls(items: List[WebpageSnippet]) -> List[ScrapeResult]:
    """Fetch text content from provided URLs.

    Args:
        items: List of SearchEngineResult items to extract content from

    Returns:
        List of ScrapeResult objects which have the following fields:
            - url: The URL of the search result
            - title: The title of the search result
            - description: The description of the search result
            - text: The full text content of the search result
    """
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Create list of tasks for concurrent execution
        tasks = []
        for item in items:
            if item.url:  # Skip empty URLs
                tasks.append(fetch_and_process_url(session, item))

        # Execute all tasks concurrently and gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors and return successful results
        return [r for r in results if isinstance(r, ScrapeResult)]


async def fetch_and_process_url(
    session: aiohttp.ClientSession, item: WebpageSnippet
) -> ScrapeResult:
    """Helper function to fetch and process a single URL."""

    if not is_valid_url(item.url):
        return ScrapeResult(
            url=item.url,
            title=item.title,
            description=item.description,
            text="Error fetching content: URL contains restricted file extension",
        )

    try:
        async with session.get(item.url, timeout=8) as response:
            if response.status == 200:
                content = await response.text()
                # Run html_to_text in a thread pool to avoid blocking
                text_content = await asyncio.get_event_loop().run_in_executor(
                    None, html_to_text, content
                )
                text_content = text_content[
                    :CONTENT_LENGTH_LIMIT
                ]  # Trim content to avoid exceeding token limit
                return ScrapeResult(
                    url=item.url,
                    title=item.title,
                    description=item.description,
                    text=text_content,
                )
            else:
                # Instead of raising, return a WebSearchResult with an error message
                return ScrapeResult(
                    url=item.url,
                    title=item.title,
                    description=item.description,
                    text=f"Error fetching content: HTTP {response.status}",
                )
    except Exception as e:
        # Instead of raising, return a WebSearchResult with an error message
        return ScrapeResult(
            url=item.url,
            title=item.title,
            description=item.description,
            text=f"Error fetching content: {str(e)}",
        )


def html_to_text(html_content: str) -> str:
    """
    Strips out all of the unnecessary elements from the HTML context to prepare it for text extraction / LLM processing.
    """
    # Parse the HTML using lxml for speed
    soup = BeautifulSoup(html_content, "lxml")

    # Extract text from relevant tags
    tags_to_extract = ("h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote")

    # Use a generator expression for efficiency
    extracted_text = "\n".join(
        element.get_text(strip=True)
        for element in soup.find_all(tags_to_extract)
        if element.get_text(strip=True)
    )

    return extracted_text


def is_valid_url(url: str) -> bool:
    """Check that a URL does not contain restricted file extensions."""
    if any(
        ext in url
        for ext in [
            ".pdf",
            ".doc",
            ".xls",
            ".ppt",
            ".zip",
            ".rar",
            ".7z",
            ".txt",
            ".js",
            ".xml",
            ".css",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".svg",
            ".webp",
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
            ".wmv",
            ".flv",
            ".wma",
            ".wav",
            ".m4a",
            ".m4v",
            ".m4b",
            ".m4p",
            ".m4u",
        ]
    ):
        return False
    return True


# ------- INITIALIZE SEARCH CLIENT AND DEFINE TOOL -------

# Get search provider from environment (default to serper)
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "serper")

# Initialize the search client based on provider
if SEARCH_PROVIDER == "serper":
    _search_client = SerperClient()
elif SEARCH_PROVIDER == "searchxng":
    _search_client = SearchXNGClient()
else:
    raise ValueError(f"Invalid search provider: {SEARCH_PROVIDER}. Must be 'serper' or 'searchxng'")


@function_tool
async def web_search(query: str) -> Union[List[ScrapeResult], str]:
    """Perform a web search for a given query and get back the URLs along with their titles, descriptions and text contents.

    Args:
        query: The search query

    Returns:
        List of ScrapeResult objects which have the following fields:
            - url: The URL of the search result
            - title: The title of the search result
            - description: The description of the search result
            - text: The full text content of the search result
    """
    try:
        search_results = await _search_client.search(
            query, filter_for_relevance=False, max_results=5
        )
        results = await scrape_urls(search_results)
        return results
    except Exception as e:
        # Return a user-friendly error message
        return f"Sorry, I encountered an error while searching: {str(e)}"