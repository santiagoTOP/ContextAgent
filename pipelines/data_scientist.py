from __future__ import annotations

from typing import Any
from pydantic import BaseModel

from agentz.agent import ContextAgent, execute_tools
from agentz.context import Context
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

        # Create manager agents with explicit dependencies
        self.observe_agent = ContextAgent(self.context, profile="observe", llm=llm)
        self.evaluate_agent = ContextAgent(self.context, profile="evaluate", llm=llm)
        self.routing_agent = ContextAgent(self.context, profile="routing", llm=llm)
        self.writer_agent = ContextAgent(self.context, profile="writer", llm=llm)

        # Create tool agents as dictionary
        tool_agent_names = [
            "data_loader_agent",
            "data_analysis_agent",
            "preprocessing_agent",
            "model_training_agent",
            "evaluation_agent",
            "visualization_agent",
        ]
        self.tool_agents = {
            name: ContextAgent(self.context, profile=name.removesuffix("_agent"), llm=llm, name=name)
            for name in tool_agent_names
        }


    @autotracing()
    async def run(self, query: DataScienceQuery) -> Any:
        # Phase 1: Initialize query in state
        self.context.state.set_query(query)

        self.update_printer("initialization", "Pipeline initialized", is_done=True)
        self.update_printer("research", "Executing research workflow...")

        # Phase 2: Iterative loop - observe → evaluate → route → tools
        while self.iteration < self.max_iterations and not self.context.state.complete:
            # Smart iteration management - single command!
            self.iterate()

            # Observe → Evaluate → Route → Tools
            observe_output = await self.observe_agent(query, group_id=self._current_group_id)
            evaluate_output = await self.evaluate_agent(observe_output, group_id=self._current_group_id)

            if not self.context.state.complete:
                routing_output = await self.routing_agent(evaluate_output, group_id=self._current_group_id)
                await execute_tools(
                    route_plan=routing_output,
                    tool_agents=self.tool_agents,
                    group_id=self._current_group_id,
                    context=self.context,
                    tracker=self.runtime_tracker,
                    update_printer_fn=self.update_printer,
                )

        # Phase 3: Final report generation
        self.update_printer("research", "Research workflow complete", is_done=True)
        await self.writer_agent(self.context.state.findings_text(), group_id=self._current_group_id)

        # Phase 4: Finalization
        final_result = self.context.state.final_report

        if self.reporter is not None:
            self.reporter.set_final_result(final_result)

        return final_result
