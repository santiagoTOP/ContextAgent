"""Core agent execution primitives."""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from agents import Runner
from agents.tracing.create import agent_span, function_span
from pydantic import BaseModel

from agentz.agent.tracker import RuntimeTracker


@dataclass
class PrinterConfig:
    """Configuration for printer updates during step execution."""
    key: Optional[str] = None
    title: Optional[str] = None
    start_message: Optional[str] = None
    done_message: Optional[str] = None


@dataclass
class AgentStep:
    """Represents a single agent execution step.

    This encapsulates all the information needed to execute an agent:
    - The agent instance
    - Instructions (static or dynamic via callable)
    - Span configuration for tracing
    - Output model for parsing
    - Printer configuration for status updates
    """

    agent: Any
    instructions: Union[str, Callable[[], str]]
    span_name: str
    span_type: str = "agent"
    output_model: Optional[type[BaseModel]] = None
    sync: bool = False
    printer_config: Optional[PrinterConfig] = None
    span_kwargs: Dict[str, Any] = field(default_factory=dict)

    def get_instructions(self) -> str:
        """Get instructions, evaluating callable if needed.

        Returns:
            Instructions string
        """
        if callable(self.instructions):
            return self.instructions()
        return self.instructions

    def get_printer_key(self, iteration: int = 0) -> Optional[str]:
        """Get the printer key, adding iteration prefix if configured.

        Args:
            iteration: Current iteration number

        Returns:
            Printer key with iteration prefix, or None if not configured
        """
        if not self.printer_config or not self.printer_config.key:
            return None
        return f"iter:{iteration}:{self.printer_config.key}"


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
    group_id: Optional[str] = None,
    printer_group_id: Optional[str] = None,
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
        group_id: Optional group to nest this item in (alias for printer_group_id)
        printer_group_id: Optional group to nest this item in (deprecated, use group_id)
        printer_border_style: Optional border color
        **span_kwargs: Additional kwargs for span (e.g., tools, input)

    Returns:
        Parsed output if output_model provided, otherwise Runner result
    """
    # Extract agent name for auto-detection
    agent_name = getattr(agent, "name", getattr(agent, "__class__", type("obj", (), {})).__name__)

    # Auto-detect span_name from agent if not provided
    if span_name is None:
        span_name = str(agent_name)

    # Auto-detect printer_key from agent if not provided
    if printer_key is None:
        printer_key = str(agent_name)

    # Auto-detect printer_title from agent if not provided
    if printer_title is None:
        # Capitalize first letter and add "ing" suffix
        title_base = str(agent_name).capitalize()
        printer_title = f"{title_base}"

    # Support both group_id and printer_group_id (group_id takes precedence)
    if group_id is not None:
        printer_group_id = group_id

    span_factory = agent_span if span_type == "agent" else function_span

    reporter = tracker.reporter
    step_id: Optional[str] = None
    if reporter:
        step_id = f"{tracker.iteration}-{span_name}-{time.time_ns()}"
        reporter.record_agent_step_start(
            step_id=step_id,
            agent_name=str(agent_name),
            span_name=span_name,
            iteration=tracker.iteration,
            group_id=printer_group_id,
            printer_title=printer_title,
        )

    full_printer_key: Optional[str] = None
    if printer_key:
        full_printer_key = f"iter:{tracker.iteration}:{printer_key}"
        tracker.update_printer(
            full_printer_key,
            "Working...",
            title=printer_title or printer_key,
            border_style=printer_border_style,
            group_id=printer_group_id,
        )

    status = "success"
    error_message: Optional[str] = None
    start_time = time.perf_counter()

    try:
        with tracker.span_context(span_factory, name=span_name, **span_kwargs) as span:
            # Activate context so tools can access it
            with tracker.activate():
                if sync:
                    result = Runner.run_sync(agent, instructions, context=tracker.data_store)
                else:
                    result = await Runner.run(agent, instructions, context=tracker.data_store)

                # Handle ContextAgent parse_output (for legacy string parsers)
                from agentz.agent.agent import ContextAgent
                if isinstance(agent, ContextAgent):
                    result = await agent.parse_output(result)

            raw_output = getattr(result, "final_output", result)

            # Update printer status and emit detailed output as a standalone panel
            if full_printer_key:
                tracker.update_printer(
                    full_printer_key,
                    "Completed",
                    is_done=True,
                    title=printer_title or printer_key,
                    group_id=printer_group_id,
                    border_style=printer_border_style,
                )

                # Extract content from raw_output
                if hasattr(raw_output, 'output'):
                    panel_content = str(raw_output.output)
                elif isinstance(raw_output, BaseModel):
                    panel_content = raw_output.model_dump_json(indent=2)
                elif isinstance(raw_output, dict):
                    panel_content = json.dumps(raw_output, indent=2)
                else:
                    panel_content = str(raw_output)

                if panel_content.strip():
                    tracker.log_panel(
                        printer_title or printer_key,
                        panel_content,
                        border_style=printer_border_style,
                        iteration=tracker.iteration,
                        group_id=printer_group_id,
                    )

            if output_model:
                if isinstance(raw_output, output_model):
                    output = raw_output
                elif isinstance(raw_output, BaseModel):
                    output = output_model.model_validate(raw_output.model_dump())
                elif isinstance(raw_output, (dict, list)):
                    output = output_model.model_validate(raw_output)
                elif isinstance(raw_output, (str, bytes, bytearray)):
                    output = output_model.model_validate_json(raw_output)
                else:
                    output = output_model.model_validate(raw_output)
                if span and hasattr(span, "set_output"):
                    span.set_output(output.model_dump())
                return output
            else:
                if span and hasattr(span, "set_output"):
                    span.set_output({"output_preview": str(getattr(result, "final_output", result))[:200]})
                return result
    except Exception as exc:  # noqa: BLE001 - propagate after logging
        status = "error"
        error_message = str(exc)
        raise
    finally:
        if reporter and step_id is not None:
            reporter.record_agent_step_end(
                step_id=step_id,
                status=status,
                duration_seconds=time.perf_counter() - start_time,
                error=error_message,
            )
