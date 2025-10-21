from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from contextagent.agent import ContextAgent
from contextagent.context import Context
from pipelines.base import BasePipeline, autotracing


class ChatQuery(BaseModel):
    """Lightweight query model for vanilla chat."""
    message: str

    def format(self) -> str:
        return self.message


class VanillaChatPipeline(BasePipeline):
    """Minimal chat pipeline using a single vanilla_chat agent profile."""

    def __init__(self, config):
        super().__init__(config)

        # Initialize context and chat agent
        self.context = Context(["profiles", "states"])
        llm = self.config.llm.main_model
        self.chat_agent = ContextAgent(context=self.context, profile="vanilla_chat", llm=llm)

    @autotracing()
    async def run(self, query: Any = None) -> Any:
        # Normalize input message
        if query is None:
            message = self.config.prompt or ""
            query_obj = ChatQuery(message=message)
        elif isinstance(query, ChatQuery):
            query_obj = query
        else:
            message = getattr(query, "prompt", None) or getattr(query, "message", None) or str(query)
            query_obj = ChatQuery(message=message)

        # Record query in state for profile runtime templates
        self.context.state.set_query(query_obj)

        self.update_printer("initialization", "Pipeline initialized", is_done=True)
        self.update_printer("chat", "Chatting...")

        # Execute chat agent
        result = await self.chat_agent(query_obj)

        self.update_printer("chat", "Chat complete", is_done=True)

        # Publish final result to reporter if active
        final_result = getattr(result, "response", None) or getattr(result, "output", None) or result
        if self.reporter is not None:
            try:
                self.reporter.set_final_result(final_result)
            except Exception:
                pass

        return final_result

