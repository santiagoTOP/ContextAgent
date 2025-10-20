from __future__ import annotations

import json
import re
from typing import Any, Optional, Callable

from pydantic import BaseModel

from agents import Agent, RunResult
from agents.run_context import TContext
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils.parsers import create_type_parser
from agentz.context.conversation import identity_wrapper
from agentz.utils.helpers import extract_final_output


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

            # Get values for each placeholder
            for placeholder in placeholders:
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

            # Render the runtime_template with context values
            return profile.render(**context_dict)

        # Fallback to original format_context_prompt if no runtime_template
        return state.format_context_prompt(current_input=current_input)


    async def __call__(
        self,
        payload: Any = None,
        *,
        tracker: Optional[Any] = None,
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
            from agentz.agent.tracker import get_current_tracker
            tracker = get_current_tracker()

        # If tracker is available (explicitly or from context), use agent_step for full tracking
        if tracker:
            from agentz.agent.executor import agent_step

            result = await agent_step(
                tracker=tracker,
                agent=self,
                instructions=instructions,
                # No group_id - tracker auto-derives from context!
            )
            # Extract final output for cleaner API
            output = extract_final_output(result)

            return output


    async def parse_output(self, run_result: RunResult) -> RunResult:
        """Apply legacy string parser only when no structured output is configured."""
        if self.output_parser and self.output_type is None:
            # import ipdb
            # ipdb.set_trace()
            run_result.final_output = self.output_parser(run_result.final_output)
        return run_result
