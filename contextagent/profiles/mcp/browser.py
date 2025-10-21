from __future__ import annotations

from contextagent.profiles.base import Profile
from contextagent.mcp.servers.browser.server import BrowserMCP


"""Browser agent profile wired for the Browser MCP server.

This profile is intended to be used together with the stdio Browser MCP
server defined in `contextagent.mcp.servers.browser.server.BrowserMCP`.
You can configure/launch the server via the pipeline config (see
`pipelines/configs/simple_browser.yaml`) or by importing `BrowserMCP`.
"""

# Expose the server factory so orchestrators can easily import alongside profile
browser_mcp_server = BrowserMCP

# Profile instance for browser agent
browser_profile = Profile(
    instructions=(
        "You are a browser agent connected to the Browser MCP server. "
        "Use the available MCP tools to open pages, navigate, click, type, "
        "query content, and return results per the user instructions."
    ),
    runtime_template="{instructions}",
    output_schema=None,
    tools=None,
    # Provide the Browser MCP server so the Agent can auto-discover MCP tools
    mcp_servers=[BrowserMCP()],
    model=None,
)
