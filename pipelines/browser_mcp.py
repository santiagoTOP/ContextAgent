from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from contextagent.agent import ContextAgent
from contextagent.context import Context
from pipelines.base import BasePipeline, autotracing


class BrowserTask(BaseModel):
    """Lightweight query model for Browser MCP tasks."""
    prompt: str

    def format(self) -> str:
        # Pass the user prompt straight through to the agent
        return self.prompt


class BrowserMCPPipeline(BasePipeline):
    """Simple pipeline that exercises the Browser MCP server via the browser profile."""

    def __init__(self, config):
        super().__init__(config)

        # Initialize context and the browser agent (profile wires MCP server)
        self.context = Context(["profiles", "states"])
        llm = self.config.llm.main_model
        self.browser_agent = ContextAgent(context=self.context, profile="browser", llm=llm, name="browser_agent")

    @autotracing()
    async def run(self, query: Any = None) -> Any:
        # Normalize to BrowserTask
        if query is None:
            prompt = self.config.prompt or ""
            q = BrowserTask(prompt=prompt)
        elif isinstance(query, BrowserTask):
            q = query
        else:
            prompt = getattr(query, "prompt", None) or getattr(query, "message", None) or str(query)
            q = BrowserTask(prompt=prompt)

        # Record query in state for template rendering and history
        self.context.state.set_query(q)

        self.update_printer("initialization", "Browser MCP pipeline initialized", is_done=True)
        self.update_printer("browser", "Running browser tasks via MCP...")

        # Execute the browser agent; underlying Agent discovers MCP tools from profile
        result = await self.browser_agent(q)

        self.update_printer("browser", "Browser MCP run complete", is_done=True)

        # Extract final output conservatively
        final_result = getattr(result, "response", None) or getattr(result, "output", None) or result
        if self.reporter is not None:
            try:
                self.reporter.set_final_result(final_result)
            except Exception:
                pass

        return final_result

