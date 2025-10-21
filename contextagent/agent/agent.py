from __future__ import annotations

import json
import re
from typing import Any, Optional, Callable

from pydantic import BaseModel

from agents import Agent, RunResult
from agents.run_context import TContext
from contextagent.llm.llm_setup import model_supports_json_and_tool_calls
from contextagent.utils.parsers import create_type_parser
from contextagent.context.conversation import identity_wrapper
from contextagent.utils.helpers import extract_final_output, parse_to_model


class ContextAgent(Agent[TContext]):
    """Augmented Agent class with context-aware capabilities.

    ContextAgent extends the base Agent class with:
    - Automatic context injection into instructions
    - Profile-based configuration (tools, instructions, output schema)
    - Automatic iteration tracking and state management
    - Runtime template rendering with state placeholders

    Usage:
        agent = ContextAgent(
            context=context,
            profile="observe",
            llm="gpt-4"
        )

    All Agent parameters can be passed via **agent_kwargs to override profile defaults:
        agent = ContextAgent(
            context=context,
            profile="observe",
            llm="gpt-4",
            tools=[custom_tool],  # Overrides profile tools
            model="gpt-4-turbo"   # Overrides llm parameter
        )
    """

    def __init__(
        self,
        context: Any,
        *,
        profile: str,
        llm: str,
        **agent_kwargs: Any,
    ) -> None:
        """Initialize ContextAgent with context and profile identifier.

        Args:
            context: Context object containing profiles and state
            profile: Profile identifier for lookup in context.profiles
            llm: LLM model name (e.g., "gpt-4", "claude-3-5-sonnet")
            **agent_kwargs: Additional Agent parameters that override profile defaults
                          (name, tools, instructions, output_type, model, etc.)
        """
        # Lookup profile from context
        resolved_profile = context.profiles[profile]
        resolved_identifier = profile

        # Build base agent configuration directly from profile
        tools = resolved_profile.tools or []
        base_agent_kwargs = {
            "instructions": resolved_profile.instructions,
            "tools": tools,
            "model": llm,
        }

        # Pass through MCP servers if defined on the profile (enables MCP tools)
        mcp_servers = getattr(resolved_profile, "mcp_servers", None)
        if mcp_servers:
            base_agent_kwargs["mcp_servers"] = mcp_servers

        # Handle output schema and parser
        output_parser = None
        output_schema = getattr(resolved_profile, "output_schema", None)

        if output_schema:
            if tools and not model_supports_json_and_tool_calls(llm):
                output_parser = create_type_parser(output_schema)
            else:
                base_agent_kwargs["output_type"] = output_schema

        # Determine final agent name
        agent_name = resolved_identifier if resolved_identifier.endswith("_agent") else f"{resolved_identifier}_agent"

        # Extract name override if provided, otherwise use derived name
        agent_kwargs_copy = dict(agent_kwargs)
        final_name = agent_kwargs_copy.pop("name", agent_name)

        # Merge agent_kwargs on top of profile config (agent_kwargs wins)
        base_agent_kwargs.update(agent_kwargs_copy)

        # Initialize parent Agent class
        super().__init__(name=final_name, **base_agent_kwargs)

        # Store ContextAgent-specific attributes
        self.output_parser = output_parser
        self._context = context  # Context reference for state access
        self._identifier = resolved_identifier  # Identifier used for profile lookup/iteration tracking
        self._profile = resolved_profile  # Profile metadata for runtime templates
        
        self._context_wrappers = {}
    
    def register_context_wrapper(self, field_name: str, wrapper: Callable[[Any], Any] = identity_wrapper) -> None:
        """Register a context wrapper for a context field."""
        self._context_wrappers[field_name] = wrapper

    def get_context_with_wrapper(self, field_name: str) -> Any:
        """Get a context wrapper for a field name."""
        return self._context.get_with_wrapper(field_name, self._context_wrappers.get(field_name, identity_wrapper))

    @property
    def role(self) -> str:
        return self._identifier.removesuffix("_agent")

    @staticmethod
    def _serialize_payload(payload: Any) -> str | None:
        """Normalize supported payload types into a string for LLM consumption."""
        if payload is None:
            return None
        if isinstance(payload, str):
            return payload
        if isinstance(payload, BaseModel):
            return payload.model_dump_json(indent=2)
        if isinstance(payload, dict):
            return json.dumps(payload, indent=2)
        return str(payload)

    def build_contextual_instructions(self, payload: Any = None) -> str:
        """Build instructions with automatic context injection from pipeline state.

        This method compiles instructions that include:
        - Runtime template rendering with placeholders filled from state (if profile has runtime_template)
        - Original query from pipeline.context.state.query
        - Previous iteration history from pipeline.context.state.iteration_history()
        - Current input payload

        Args:
            payload: Current input data for the agent

        Returns:
            Formatted instructions string with full context

        Note:
            This method requires self._context to be set.
        """
        # Convert payload to string
        current_input = self._serialize_payload(payload)

        state = self._context.state

        # Check if profile has runtime_template
        profile = getattr(self, '_profile', None)
        if profile and hasattr(profile, 'runtime_template') and profile.runtime_template:
            # Extract placeholder names from runtime_template
            template = profile.runtime_template
            placeholders = set(re.findall(r'\{([a-z_]+)\}', template))

            # Build context dict dynamically with intelligent fallbacks
            context_dict = {}

            # Always register the runtime input payload upfront
            if 'runtime_input' in placeholders and payload is not None:
                context_dict['runtime_input'] = current_input

            # Get values for each placeholder
            for placeholder in placeholders:
                # Skip if already set (e.g., runtime_input)
                if placeholder in context_dict:
                    continue

                # Try state attribute first
                # TODO: Use get_with_wrapper instead
                value = getattr(state, placeholder, None)
                # value = state.get_with_wrapper(placeholder, self._context_wrappers.get(placeholder, identity_wrapper))
                if value is not None:
                    context_dict[placeholder] = str(value)
                    continue

                # Default: empty string for unknown placeholders
                context_dict[placeholder] = ''

            # Extract BaseModel fields (only if template needs them)
            serialized_payload: str | None = None

            if payload is not None and isinstance(payload, BaseModel):
                try:
                    payload_dict = payload.model_dump()
                    for field_name, field_value in payload_dict.items():
                        lowercased_key = field_name.lower()
                        # Only add if template actually needs this field
                        if lowercased_key in placeholders:
                            context_dict[lowercased_key] = str(field_value) if field_value is not None else ''
                except Exception:
                    pass

            if serialized_payload is None and payload is not None:
                serialized_payload = self._serialize_payload(payload)

            if serialized_payload is not None:
                for key in ("task", "payload", "input"):
                    if key in placeholders and key not in context_dict:
                        context_dict[key] = serialized_payload

            # Render the runtime_template with context values
            return profile.render(**context_dict)

        # Fallback to original format_context_prompt if no runtime_template
        return state.format_context_prompt(current_input=current_input)


    async def __call__(
        self,
        payload: Any = None,
        *,
        tracker: Optional[Any] = None,
        span_name: Optional[str] = None,
        span_type: Optional[str] = None,
        output_model: Optional[type[BaseModel]] = None,
        printer_key: Optional[str] = None,
        printer_title: Optional[str] = None,
        printer_border_style: Optional[str] = None,
        record_payload: Optional[bool] = None,
        sync: bool = False,
        **span_kwargs: Any,
    ) -> Any:
        """Make ContextAgent callable directly.

        This allows usage like: result = await agent(input_data)

        When called with tracker provided (or available from context), uses the agent_step
        function for full tracking/tracing. Otherwise, uses ContextRunner.

        Note: When calling directly without tracker, input validation
        is relaxed to allow string inputs even if agent has a defined input_model.

        Args:
            payload: Input data for the agent
            tracker: Optional RuntimeTracker for execution with tracking.
                    If not provided, will attempt to get from context via get_current_tracker().

        Returns:
            Parsed output if in pipeline context, otherwise RunResult
        """
        # Build instructions with automatic context injection if enabled
        instructions = self.build_contextual_instructions(payload)

        # Auto-detect tracker from context if not explicitly provided
        if tracker is None:
            from contextagent.agent.tracker import get_current_tracker
            tracker = get_current_tracker()

        # If tracker is available (explicitly or from context), use agent_step for full tracking
        if tracker:
            from contextagent.agent.executor import agent_step

            is_tool_agent = bool(self.tools)
            resolved_span_name = span_name or self.name
            resolved_span_type = span_type or ("tool" if is_tool_agent else "agent")

            resolved_printer_key = printer_key or (f"tool:{resolved_span_name}" if is_tool_agent else None)
            resolved_printer_title = printer_title or (f"Tool: {resolved_span_name}" if is_tool_agent else None)

            resolved_output_model = output_model
            if resolved_output_model is None and is_tool_agent:
                from contextagent.profiles.base import ToolAgentOutput

                resolved_output_model = ToolAgentOutput

            resolved_record_payload = record_payload if record_payload is not None else is_tool_agent

            # Ensure MCP servers are connected for this call, and cleaned up after
            connected_servers = []
            try:
                mcp_servers = getattr(self, "mcp_servers", None)
                if mcp_servers:
                    for server in mcp_servers:
                        # Connect only if the server exposes connect() and is not already connected
                        connect = getattr(server, "connect", None)
                        session = getattr(server, "session", None)
                        if callable(connect) and session is None:
                            await connect()
                            connected_servers.append(server)

                result = await agent_step(
                    tracker=tracker,
                    agent=self,
                    instructions=instructions,
                    span_name=resolved_span_name,
                    span_type=resolved_span_type,
                    output_model=resolved_output_model,
                    sync=sync,
                    printer_key=resolved_printer_key,
                    printer_title=resolved_printer_title,
                    printer_border_style=printer_border_style,
                    **span_kwargs,
                )
            finally:
                # Cleanup only the servers we connected here
                for server in connected_servers:
                    cleanup = getattr(server, "cleanup", None)
                    if callable(cleanup):
                        try:
                            await cleanup()
                        except Exception:
                            pass
            # if self.name == "web_searcher_agent":
            #     import ipdb; ipdb.set_trace()

            if resolved_output_model and isinstance(result, resolved_output_model):
                final_output = result
            else:
                final_output = extract_final_output(result)
                if (
                    resolved_output_model
                    and not isinstance(final_output, resolved_output_model)
                ):
                    try:
                        final_output = parse_to_model(final_output, resolved_output_model)
                    except Exception:
                        # Preserve original output if parsing fails
                        pass

            if resolved_record_payload and hasattr(self, "_context"):
                state = getattr(self._context, "state", None)
                if state is not None:
                    try:
                        state.record_payload(final_output)
                    except Exception:
                        pass
                    try:
                        iteration = state.current_iteration
                    except Exception:
                        iteration = None
                    if iteration is not None:
                        try:
                            iteration.tools.append(final_output)
                        except Exception:
                            pass

            return final_output


    async def parse_output(self, run_result: RunResult) -> RunResult:
        """Apply legacy string parser only when no structured output is configured."""
        if self.output_parser and self.output_type is None:
            # import ipdb
            # ipdb.set_trace()
            run_result.final_output = self.output_parser(run_result.final_output)
        return run_result
