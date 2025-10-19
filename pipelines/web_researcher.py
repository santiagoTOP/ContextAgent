from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentz.agent.base import ContextAgent
from agentz.context.context import Context
from agentz.runner import execute_tools
from pipelines.base import BasePipeline


class WebSearchQuery(BaseModel):
    """Query model for data science tasks."""
    prompt: str

    def format(self) -> str:
        """Format web search query."""
        return (
            f"Web search query: {self.prompt}\n"
            "Provide a comprehensive web search workflow"
        )


class WebSearcherPipeline(BasePipeline):
    """Web search pipeline using manager-tool pattern with multi-task planning.

    This pipeline demonstrates web search with parallel query execution:
    - __init__: Setup agents (observe, evaluate, planning, writer) and tool agents (web_searcher)
    - run: Implement the workflow logic (observe → evaluate → plan → execute multiple searches → write)
    - WebSearchQuery.format(): Format query (handled automatically by BasePipeline)

    The planning agent generates multiple web search tasks that are executed in parallel.
    All other logic (iteration, tool execution, memory save) is handled by BasePipeline.
    """

    def __init__(self, config):
        """Initialize pipeline with explicit manager agents and tool agent dictionary."""
        super().__init__(config)

        # Initialize context and profiles
        self.context = Context(["profiles", "states"])
        llm = self.config.llm.main_model

        # Create callbacks for agent execution
        callbacks = {
            'agent_step': self.agent_step,
            'get_current_group_id': lambda: self._current_group_id,
        }

        # Create manager agents with explicit dependencies
        self.observe_agent = ContextAgent.from_profile(
            context=self.context,
            config=self.config,
            role="observe",
            llm=llm,
            callbacks=callbacks,
        )
        self.evaluate_agent = ContextAgent.from_profile(
            context=self.context,
            config=self.config,
            role="evaluate",
            llm=llm,
            callbacks=callbacks,
        )
        self.planning_agent = ContextAgent.from_profile(
            context=self.context,
            config=self.config,
            role="web_planning",
            llm=llm,
            callbacks=callbacks,
        )
        self.writer_agent = ContextAgent.from_profile(
            context=self.context,
            config=self.config,
            role="writer",
            llm=llm,
            callbacks=callbacks,
        )

        # Create tool agents as dictionary
        tool_agent_names = [
            "web_searcher",
            "web_crawler",
        ]
        self.tool_agents = {
            f"{name}_agent": ContextAgent.from_profile(
                context=self.context,
                config=self.config,
                role=name,
                llm=llm,
                callbacks=callbacks,
            )
            for name in tool_agent_names
        }

        # Update all agents with tool_agents reference
        for agent in [self.observe_agent, self.evaluate_agent, self.planning_agent, self.writer_agent]:
            agent._tool_agents = self.tool_agents

    async def run(self, query: Any = None) -> Any:
        """Execute web search workflow - full implementation in one function.

        Args:
            query: Optional WebSearchQuery input

        Returns:
            Final report from state
        """
        self.update_printer("research", "Executing web search workflow...")

        # Iterative loop: observe → evaluate → plan → execute multiple searches
        while self.iteration < self.max_iterations and not self.context.state.complete:
            # Begin iteration - group_id managed automatically
            self.begin_iteration()

            # Get pre-formatted query from state
            query = self.context.state.formatted_query or ""

            # Observe → Evaluate → Plan → Execute Multiple Tools
            observe_output = await self.observe_agent(query)
            evaluate_output = await self.evaluate_agent(observe_output)

            if not self.context.state.complete:
                planning_output = await self.planning_agent(evaluate_output)
                await execute_tools(
                    route_plan=planning_output,
                    tool_agents=self.tool_agents,
                    group_id=self._current_group_id,
                    context=self.context,
                    agent_step_fn=self.agent_step,
                    update_printer_fn=self.update_printer,
                )

            # End iteration - group_id managed automatically
            self.end_iteration()

        # Final report
        self.update_printer("research", "Web search workflow complete", is_done=True)
        await self.writer_agent(self.context.state.findings_text())

        # Return final result
        final_result = self.context.state.final_report

        if self.reporter is not None:
            self.reporter.set_final_result(final_result)

        return final_result
