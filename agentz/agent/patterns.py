"""High-level execution patterns for common pipeline workflows."""

import asyncio
from typing import Any, Dict, Optional

from loguru import logger


async def execute_tool_plan(
    plan: Any,
    tool_agents: Dict[str, Any],
    group_id: str,
    context: Any,
    agent_step_fn: Any,
    update_printer_fn: Optional[Any] = None,
) -> None:
    """Execute a routing plan with tool agents.

    Args:
        plan: AgentSelectionPlan with tasks to execute
        tool_agents: Dict mapping agent names to agent instances
        group_id: Group ID for printer updates
        context: Pipeline context with state
        agent_step_fn: Function to execute agent steps
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
                    key=f"{group_id}:tool:{task.agent}",
                    message=f"Completed {task.agent}",
                    is_done=True,
                    group_id=group_id,
                )
            return output

        raw_result = await agent_step_fn(
            agent=agent,
            instructions=task.model_dump_json(),
            span_name=task.agent,
            span_type="tool",
            output_model=ToolAgentOutput,
            printer_key=f"tool:{task.agent}",
            printer_title=f"Tool: {task.agent}",
            printer_group_id=group_id,
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
                key=f"{group_id}:tool:{task.agent}",
                message=f"Completed {task.agent}",
                is_done=True,
                group_id=group_id,
            )
        return output

    coroutines = [run_single(task) for task in plan.tasks]
    for coro in asyncio.as_completed(coroutines):
        tool_output = await coro
        state.current_iteration.tools.append(tool_output)


async def execute_tools(
    route_plan: Any,
    tool_agents: Dict[str, Any],
    group_id: str,
    context: Any,
    agent_step_fn: Any,
    update_printer_fn: Optional[Any] = None,
) -> None:
    """Execute tool agents based on routing plan.

    Args:
        route_plan: The routing plan (can be AgentSelectionPlan or other)
        tool_agents: Dict mapping agent names to agent instances
        group_id: Group ID for printer updates
        context: Pipeline context with state
        agent_step_fn: Function to execute agent steps
        update_printer_fn: Optional function for printer updates
    """
    plan = route_plan

    if plan and plan.tasks:
        await execute_tool_plan(
            plan, tool_agents, group_id, context, agent_step_fn, update_printer_fn
        )


