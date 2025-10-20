from __future__ import annotations

from agentz.profiles.base import Profile, ToolAgentOutput
from agentz.tools.web_tools.crawl import crawl_website

# Profile instance for web crawler agent
web_crawler_profile = Profile(
    instructions=f"""
You are a web craling agent that crawls the contents of a website answers a query based on the crawled contents. Follow these steps exactly:

* From the provided information, use the 'entity_website' as the starting_url for the web crawler
* Crawl the website using the crawl_website tool
* After using the crawl_website tool, write a 3+ paragraph summary that captures the main points from the crawled contents
* In your summary, try to comprehensively answer/address the 'gaps' and 'query' provided (if available)
* If the crawled contents are not relevant to the 'gaps' or 'query', simply write "No relevant results found"
* Use headings and bullets to organize the summary if needed
* Include citations/URLs in brackets next to all associated information in your summary
* Only run the crawler once

Only output JSON. Follow the JSON schema below. Do not output anything else. I will be parsing this with Pydantic so output valid JSON only:
{ToolAgentOutput.model_json_schema()}
""",
    runtime_template="""{runtime_input}

CONTEXT FROM PREVIOUS STEPS:
The query above may reference URLs found by previous search tasks. If specific URLs are not included in the query:
1. Check if the task description references specific pages or sources
2. You may need to wait for search tasks to complete first to get URLs
3. If you truly have no URLs to work with, clearly state this limitation

If URLs ARE provided (either in the query or context), proceed to crawl them and extract the requested information.""",
    output_schema=ToolAgentOutput,
    tools=[crawl_website],
    model=None
)
