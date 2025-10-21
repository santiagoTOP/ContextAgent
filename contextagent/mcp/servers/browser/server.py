from __future__ import annotations

"""
Browser MCP server factory.

Spawns the Browser MCP CLI via `npx @browsermcp/mcp@latest` using stdio transport,
so it can be consumed by the "agents" runtime through MCP.

Note: We proactively apply a small patch to the Browser MCP CLI cache to avoid a
known recursive `server.close` issue when shutting down. This is a no-op if the
cache is not present or already patched.
"""

from agents.mcp import MCPServer, MCPServerStdio

from contextagent.mcp.patches import apply_browsermcp_close_patch


def BrowserMCP() -> MCPServer:
    """Return a configured Browser MCP server (stdio over npx).

    Uses `npx -y @browsermcp/mcp@latest` to execute the server binary from npm.
    """
    # Best-effort patch for the Browser MCP CLI close handler (safe if unavailable)
    try:
        apply_browsermcp_close_patch()
    except Exception:
        # Non-fatal: if patching fails we still try to start the server
        pass

    return MCPServerStdio(
        cache_tools_list=True,
        params={
            "command": "npx",
            # Include "-y" for non-interactive installation of the package
            "args": ["-y", "@browsermcp/mcp@latest"],
        },
    )

