from __future__ import annotations

import asyncio

from typing import Any
from pydantic import BaseModel

from agentz.agent import ContextAgent
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

        # Set available agents with descriptions for routing
        self.context.state.available_agents = {
            "data_loader_agent": "Loads and inspects datasets from file paths",
            "data_analysis_agent": "Performs exploratory data analysis to uncover patterns and relationships in datasets",
            "preprocessing_agent": "Cleans and transforms datasets for analysis and modeling",
            "model_training_agent": "Trains and evaluates machine learning models on prepared datasets",
            "evaluation_agent": "Provides comprehensive performance assessments of machine learning models",
            "visualization_agent": "Creates visual representations of data patterns and insights",
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
            # No need for group_id - tracker auto-derives from context!
            observe_output = await self.observe_agent(query)
            evaluate_output = await self.evaluate_agent(observe_output)

            if not self.context.state.complete:
                routing_output = await self.routing_agent(evaluate_output)
                # import ipdb; ipdb.set_trace()
                plan_tasks = routing_output.tasks

                if plan_tasks:
                    self.context.state.current_iteration.tools.clear()
                    coroutines = [self.tool_agents[task.agent](task) for task in plan_tasks]
                    for coroutine in asyncio.as_completed(coroutines):
                        await coroutine

        # Phase 3: Final report generation
        self.update_printer("research", "Research workflow complete", is_done=True)
        await self.writer_agent(self.context.state.findings_text())

        # Phase 4: Finalization
        final_result = self.context.state.final_report

        if self.reporter is not None:
            self.reporter.set_final_result(final_result)

        return final_result
