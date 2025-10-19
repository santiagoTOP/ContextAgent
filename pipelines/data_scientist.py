from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentz.agent.base import ContextAgent
from agentz.context.context import Context
from pipelines.base import BasePipeline, autotracing


class DataScienceQuery(BaseModel):
    """Query model for data science tasks."""
    prompt: str
    data_path: str

    def format(self) -> str:
        """Format data science query."""
        return (
            f"Task: {self.prompt}\n"
            f"Dataset path: {self.data_path}\n"
            "Provide a comprehensive data science workflow"
        )

class DataScientistPipeline(BasePipeline):
    """Data science pipeline using manager-tool pattern.

    This pipeline demonstrates the minimal implementation needed:
    - __init__: Setup agents and context
    - run(): Complete workflow implementation with query formatting, iteration, and finalization
    """

    def __init__(self, config):
        """Initialize pipeline with explicit manager agents and tool agent dictionary."""
        super().__init__(config)

        # Initialize context and profiles
        self.context = Context(["profiles", "states"])
        llm = self.config.llm.main_model

        # Create manager agents - automatically bound to pipeline with role
        self.observe_agent = ContextAgent.from_profile(self, "observe", llm)
        self.evaluate_agent = ContextAgent.from_profile(self, "evaluate", llm)
        self.routing_agent = ContextAgent.from_profile(self, "routing", llm)
        self.writer_agent = ContextAgent.from_profile(self, "writer", llm)

        # Create tool agents as dictionary - automatically bound to pipeline
        tool_agents = [
            "data_loader_agent",
            "data_analysis_agent",
            "preprocessing_agent",
            "model_training_agent",
            "evaluation_agent",
            "visualization_agent",
        ]
        self.tool_agents = {
            f"{name}": ContextAgent.from_profile(self, name.removesuffix("_agent"), llm)
            for name in tool_agents
        }

    @autotracing()
    async def run(self, query: Any = None) -> Any:
        """Execute data science workflow - full pipeline logic in one method.

        Implements the complete workflow:
        1. Initialize query in state
        2. Iterative loop: observe → evaluate → route → tools
        3. Final report generation
        4. Finalization

        This method demonstrates the @autotracing() decorator pattern, which automatically
        handles pipeline initialization (reporter, printer, tracing) and cleanup.

        For advanced use cases requiring explicit control, you can still use the context manager:
            with self.run_context(enable_printer=False):
                # custom logic here

        Args:
            query: DataScienceQuery input

        Returns:
            Final report from state
        """
        # Phase 1: Initialize query in state
        if query is not None:
            self.context.state.set_query(query)
            formatted_query = query.format()
        else:
            formatted_query = ""

        self.update_printer("initialization", "Pipeline initialized", is_done=True)
        self.update_printer("research", "Executing research workflow...")

        # Phase 2: Iterative loop - observe → evaluate → route → tools
        while self.iteration < self.max_iterations and not self.context.state.complete:
            # Begin iteration - group_id managed automatically
            self.begin_iteration()

            # Observe → Evaluate → Route → Tools
            observe_output = await self.observe_agent(formatted_query)
            evaluate_output = await self.evaluate_agent(observe_output)

            if not self.context.state.complete:
                routing_output = await self.routing_agent(evaluate_output)
                await self._execute_tools(routing_output, self.tool_agents)

            # End iteration - group_id managed automatically
            self.end_iteration()

        # Phase 3: Final report generation
        self.begin_final_report()
        self.update_printer("research", "Research workflow complete", is_done=True)

        await self.writer_agent(self.context.state.findings_text())

        self.end_final_report()

        # Phase 4: Finalization
        final_result = self.context.state.final_report

        if self.reporter is not None:
            self.reporter.set_final_result(final_result)

        return final_result
