"""High-level execution patterns for common pipeline workflows."""

import asyncio
from typing import Any, Dict, Optional

from loguru import logger

from agentz.agent.executor import agent_step


async def execute_tool_plan(
    plan: Any,
    tool_agents: Dict[str, Any],
    context: Any,
    tracker: Any,
    update_printer_fn: Optional[Any] = None,
) -> None:
    """Execute a routing plan with tool agents.

    Args:
        plan: AgentSelectionPlan with tasks to execute
        tool_agents: Dict mapping agent names to agent instances
        context: Pipeline context with state
        tracker: RuntimeTracker for agent execution
        update_printer_fn: Optional function for printer updates
    """
    # Import here to avoid circular dependency
    from agentz.profiles.base import ToolAgentOutput


    state = context.state
    state.current_iteration.tools.clear()

    async def run_single(task: Any) -> ToolAgentOutput:
        agent = tool_agents.get(task.agent)
        if agent is None:
            output = ToolAgentOutput(
                output=f"No implementation found for agent {task.agent}",
                sources=[],
            )
            if update_printer_fn:
                update_printer_fn(
                    key=f"tool:{task.agent}",
                    message=f"Completed {task.agent}",
                    is_done=True,
                )
            return output

        raw_result = await agent_step(
            tracker=tracker,
            agent=agent,
            instructions=task.model_dump_json(),
            span_name=task.agent,
            span_type="tool",
            output_model=ToolAgentOutput,
            printer_key=f"tool:{task.agent}",
            printer_title=f"Tool: {task.agent}",
        )
        # import ipdb
        # ipdb.set_trace()

        if isinstance(raw_result, ToolAgentOutput):
            output = raw_result
        elif hasattr(raw_result, "final_output_as"):
            output = raw_result.final_output_as(ToolAgentOutput)
        elif hasattr(raw_result, "final_output"):
            output = ToolAgentOutput(output=str(raw_result.final_output), sources=[])
        else:
            output = ToolAgentOutput(output=str(raw_result), sources=[])

        try:
            state.record_payload(output)
        except Exception as exc:
            logger.debug(f"Failed to record tool payload for {task.agent}: {exc}")

        if update_printer_fn:
            update_printer_fn(
                key=f"tool:{task.agent}",
                message=f"Completed {task.agent}",
                is_done=True,
            )
        return output

    coroutines = [run_single(task) for task in plan.tasks]
    for coro in asyncio.as_completed(coroutines):
        tool_output = await coro
        state.current_iteration.tools.append(tool_output)


async def execute_tools(
    route_plan: Any,
    tool_agents: Dict[str, Any],
    context: Any,
    tracker: Any,
    update_printer_fn: Optional[Any] = None,
) -> None:
    """Execute tool agents based on routing plan.

    Args:
        route_plan: The routing plan (can be AgentSelectionPlan or other)
        tool_agents: Dict mapping agent names to agent instances
        context: Pipeline context with state
        tracker: RuntimeTracker for agent execution
        update_printer_fn: Optional function for printer updates
    """
    plan = route_plan

    if plan and plan.tasks:
        await execute_tool_plan(
            plan, tool_agents, context, tracker, update_printer_fn
        )


