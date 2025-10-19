from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from agentz.agent.base import ContextAgent
from agentz.context.context import Context
from agentz.profiles.manager.routing import AgentSelectionPlan, AgentTask
from pipelines.base import BasePipeline


class RoutingInput(BaseModel):
    """Input schema for routing agent."""
    query: str = Field(description="Original user query")
    gap: str = Field(description="Knowledge gap to address")
    history: str = Field(description="History of actions, findings and thoughts")


class SimpleQuery(BaseModel):
    """Lightweight query model for the simple pipeline."""
    prompt: str

    def format(self) -> str:
        """Render the query into the routing-friendly prompt."""
        return f"Task: {self.prompt}"


class SimplePipeline(BasePipeline):
    """Single-pass pipeline that routes directly to the web searcher tool."""

    def __init__(self, config):
        super().__init__(config)

        # Initialize shared context (profiles + conversation state)
        self.context = Context(["profiles", "states"])
        llm = self.config.llm.main_model

        # Bind agents from registered profiles with explicit dependencies
        self.routing_agent = ContextAgent.from_profile(
            context=self.context,
            config=self.config,
            role="routing",
            llm=llm,
        )
        self.tool_agent = ContextAgent.from_profile(
            context=self.context,
            config=self.config,
            role="web_searcher",
            llm=llm,
        )

    async def run(self, query: Any = None) -> Any:
        """Route the query once and execute the web searcher agent.

        Args:
            query: Optional SimpleQuery input

        Returns:
            Tool agent output
        """
        # Initialize query
        if query is None:
            prompt = self.config.prompt or "Analyze the dataset and provide insights."
            query = SimpleQuery(prompt=prompt)
        elif not isinstance(query, SimpleQuery):
            # Coerce arbitrary input into SimpleQuery for consistent formatting
            prompt = getattr(query, "prompt", None) or str(query)
            query = SimpleQuery(prompt=prompt)

        # Set query in context
        if query is not None:
            self.context.state.set_query(query)

        logger.info(f"User prompt: {self.config.prompt}")

        # Start single iteration for structured logging
        self.begin_iteration(title="Single Pass")

        # Get pre-formatted query from state
        query_str = self.context.state.formatted_query or ""

        routing_input = RoutingInput(
            query=query_str,
            gap="Route the query to the web_searcher_agent",
            history=self.context.state.iteration_history(include_current=False) or "",
        )

        routing_plan = await self.routing_agent(routing_input, group_id=self._current_group_id, tracker=self.runtime_tracker)
        task = self._select_task(routing_plan)

        # Just pass the task query string directly - agent will handle it
        tool_payload = task.query

        result = await self.tool_agent(tool_payload, group_id=self._current_group_id, tracker=self.runtime_tracker)

        if self.state:
            self.state.final_report = result.output
            self.state.mark_research_complete()

        self.end_iteration()

        logger.info("Simple pipeline completed")

        # Return final result
        final_result = result.output if hasattr(result, 'output') else result

        if self.reporter is not None:
            self.reporter.set_final_result(final_result)

        return final_result

    @staticmethod
    def _select_task(plan: AgentSelectionPlan) -> AgentTask:
        """Pick the first web searcher task from the routing plan."""
        if not plan or not plan.tasks:
            raise ValueError("Routing agent did not return any tasks.")

        for task in plan.tasks:
            if task.agent == "web_searcher_agent":
                return task

        # Fallback: take the first task when a specific agent isn't assigned
        return plan.tasks[0]
