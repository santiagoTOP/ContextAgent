"""Core agent execution primitives."""

import asyncio
from typing import Any, Optional

from agents import Runner
from agents.tracing.create import agent_span, function_span
from pydantic import BaseModel

from contextagent.agent.tracker import RuntimeTracker
from contextagent.utils.helpers import extract_final_output, parse_to_model, serialize_content


async def agent_step(
    tracker: RuntimeTracker,
    agent,
    instructions: str,
    span_name: Optional[str] = None,
    span_type: str = "agent",
    output_model: Optional[type[BaseModel]] = None,
    sync: bool = False,
    printer_key: Optional[str] = None,
    printer_title: Optional[str] = None,
    printer_border_style: Optional[str] = None,
    **span_kwargs
) -> Any:
    """Run an agent with span tracking and optional output parsing.

    Args:
        tracker: RuntimeTracker for tracing, printing, etc.
        agent: The agent to run
        instructions: Instructions/prompt for the agent
        span_name: Name for the span (auto-detected from agent.name if not provided)
        span_type: Type of span - "agent" or "function"
        output_model: Optional pydantic model to parse output
        sync: Whether to run synchronously
        printer_key: Optional key for printer updates (auto-detected from agent.name if not provided)
        printer_title: Optional title for printer display (auto-detected from agent.name if not provided)
        printer_border_style: Optional border color
        **span_kwargs: Additional kwargs for span (e.g., tools, input)

    Returns:
        Parsed output if output_model provided, otherwise Runner result
    """
    span_factory = agent_span if span_type == "agent" else function_span

    handle = tracker.start_agent_step(
        agent=agent,
        span_name=span_name,
        span_factory=span_factory,
        span_kwargs=span_kwargs,
        printer_key=printer_key,
        printer_title=printer_title,
        printer_border_style=printer_border_style,
    )

    status = "success"
    error_message: Optional[str] = None

    try:
        with tracker.span_scope(handle) as span:
            # Activate context so tools can access it
            with tracker.activate():
                if sync:
                    result = Runner.run_sync(agent, instructions, context=tracker.data_store)
                else:
                    result = await Runner.run(agent, instructions, context=tracker.data_store)
                
                # if agent.name == "web_searcher_agent":
                #     import ipdb; ipdb.set_trace()

                # Handle ContextAgent parse_output (for legacy string parsers)
                from contextagent.agent.agent import ContextAgent
                if isinstance(agent, ContextAgent):
                    result = await agent.parse_output(result)

            raw_output = extract_final_output(result)
            panel_content = serialize_content(raw_output)

            tracker.log_agent_panel(handle, panel_content)

            if output_model:
                return parse_to_model(raw_output, output_model, span)
            else:
                tracker.preview_output(handle, panel_content[:200])
                return result
    except Exception as exc:  # noqa: BLE001 - propagate after logging
        status = "error"
        error_message = str(exc)
        raise
    finally:
        tracker.finish_agent_step(
            handle,
            status=status,
            error=error_message,
        )
