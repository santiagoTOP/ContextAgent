from __future__ import annotations

import re
from typing import Any, Callable, Optional

from pydantic import BaseModel

from agents import Agent, RunResult
from agents.run_context import TContext
from agentz.llm.llm_setup import model_supports_json_and_tool_calls
from agentz.utils.parsers import create_type_parser

PromptBuilder = Callable[[Any, Any, "ContextAgent"], str]


class ContextAgent(Agent[TContext]):
    """Capability-centric wrapper that binds LLM + tools + typed IO contract."""

    def __init__(
        self,
        *args: Any,
        output_model: type[BaseModel] | None = None,
        prompt_builder: PromptBuilder | None = None,
        default_span_type: str = "agent",
        output_parser: Optional[Callable[[str], Any]] = None,
        auto_inject_context: bool = True,
        context: Optional[Any] = None,
        config: Optional[Any] = None,
        tool_agents: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if output_model and kwargs.get("output_type"):
            raise ValueError("Use either output_model or output_type, not both.")
        if output_model is not None:
            kwargs["output_type"] = output_model

        super().__init__(*args, **kwargs)

        self.output_model = self._coerce_output_model(output_model or getattr(self, "output_type", None))
        self.prompt_builder = prompt_builder
        self.default_span_type = default_span_type
        self.output_parser = output_parser
        self.auto_inject_context = auto_inject_context  # Whether to automatically inject context in __call__
        self._context = context  # Context reference for state access
        self._config = config  # Config reference for settings access
        self._tool_agents = tool_agents  # Available tool agents
        self._role = None  # Optional role identifier for automatic iteration tracking

    @classmethod
    def from_profile(
        cls,
        context: Any,
        config: Any,
        role: str,
        llm: str,
        tool_agents: Optional[dict[str, Any]] = None,
    ) -> "ContextAgent":
        """Create a ContextAgent from context and role.

        Automatically looks up the profile from context.profiles[role],
        derives agent name, and configures the agent with explicit dependencies.

        Args:
            context: Context instance (must have profiles attribute)
            config: Config instance for settings access
            role: Role name that maps to a profile key (e.g., "observe", "evaluate")
            llm: LLM model name (e.g., "gpt-4", "claude-3-5-sonnet")
            tool_agents: Optional dictionary of available tool agents

        Returns:
            ContextAgent instance configured from the profile

        Example:
            agent = ContextAgent.from_profile(
                context=self.context,
                config=self.config,
                role="observe",
                llm="gpt-4",
            )
        """
        # Look up profile from context
        profile = context.profiles[role]

        # Auto-derive name from role
        agent_name = role + "_agent" if role != "agent" else "agent"

        # Get tools and output schema from profile
        tools = profile.tools or []
        output_schema = getattr(profile, "output_schema", None)

        # Check if model supports both structured output and tools
        # If not, use output_parser instead of output_model
        output_model = None
        output_parser = None
        instructions = profile.instructions

        if output_schema and tools and not model_supports_json_and_tool_calls(llm):
            # Model doesn't support both - use parser instead
            output_parser = create_type_parser(output_schema)
        elif output_schema:
            # Model supports both or no tools present - use structured output
            output_model = output_schema

        agent = cls(
            name=agent_name,
            instructions=instructions,
            output_model=output_model,
            tools=tools,
            model=llm,
            output_parser=output_parser,
            context=context,
            config=config,
            tool_agents=tool_agents,
        )

        # Set role and profile for runtime_template access
        agent._role = role
        agent._profile = profile  # Store profile for runtime_template access

        return agent

    @staticmethod
    def _coerce_output_model(candidate: Any) -> type[BaseModel] | None:
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
        return None

    @staticmethod
    def _to_prompt_payload(payload: Any) -> dict[str, Any]:
        if payload is None:
            return {}
        if isinstance(payload, BaseModel):
            return payload.model_dump()
        if isinstance(payload, dict):
            return payload
        return {"input": payload}

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
        if isinstance(payload, str):
            current_input = payload
        elif isinstance(payload, BaseModel):
            current_input = payload.model_dump_json(indent=2)
        elif isinstance(payload, dict):
            import json
            current_input = json.dumps(payload, indent=2)
        elif payload is None:
            current_input = None
        else:
            current_input = str(payload)

        # Get context
        if self._context is None:
            # Fallback to regular instruction building if no context
            return current_input or ""

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
                value = getattr(state, placeholder, None)
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
        group_id: Optional[str] = None,
        tracker: Optional[Any] = None,
    ) -> Any:
        """Make ContextAgent callable directly.

        This allows usage like: result = await agent(input_data)

        When called with tracker provided, uses the agent_step function for full
        tracking/tracing. Otherwise, uses ContextRunner.

        Note: When calling directly without tracker, input validation
        is relaxed to allow string inputs even if agent has a defined input_model.

        Args:
            payload: Input data for the agent
            group_id: Optional group ID for tracking. Must be provided explicitly when needed.
            tracker: Optional RuntimeTracker for execution with tracking

        Returns:
            Parsed output if in pipeline context, otherwise RunResult
        """
        # Build instructions with automatic context injection if enabled
        if self.auto_inject_context and self._context is not None:
            instructions = self.build_contextual_instructions(payload)
        else:
            # Build prompt without validation
            if isinstance(payload, str):
                instructions = payload
            elif isinstance(payload, BaseModel):
                instructions = payload.model_dump_json(indent=2)
            elif isinstance(payload, dict):
                import json
                instructions = json.dumps(payload, indent=2)
            elif payload is None and isinstance(self.instructions, str):
                instructions = self.instructions
            else:
                instructions = str(payload)

        # If tracker is provided, use agent_step function for full tracking
        if tracker:
            from agentz.runner.executor import agent_step
            
            result = await agent_step(
                tracker=tracker,
                agent=self,
                instructions=instructions,
                group_id=group_id,
            )
            # Extract final output for cleaner API
            output = result.final_output if hasattr(result, 'final_output') else result

            # Automatic iteration tracking based on role
            if self._role and self._context:
                from agentz.runner.utils import serialize_output, record_structured_payload

                state = getattr(self._context, "state", None)
                try:
                    iteration = self._context.current_iteration

                    # Special handling for "observe" role - set iteration.observation
                    if self._role == "observe":
                        # Extract observations field if output is a BaseModel with that field
                        if isinstance(output, BaseModel) and hasattr(output, 'observations'):
                            iteration.observation = output.observations
                        else:
                            serialized = serialize_output(output)
                            iteration.observation = serialized

                    # Record structured payload for all roles
                    record_structured_payload(state, output, context_label=self._role)
                except Exception:
                    # Silently skip if context/iteration not available
                    pass

                if state is not None and self._role == "writer":
                    try:
                        if output is not None:
                            state.final_report = serialize_output(output)
                        elif state.final_report is None:
                            state.final_report = ""
                    except Exception:
                        # If serialization fails, fall back to string coercion
                        if output is not None:
                            state.final_report = str(output)
                        elif state.final_report is None:
                            state.final_report = ""

            return output

        # Otherwise, use ContextRunner to execute the agent
        from agentz.runner import ContextRunner

        result = await ContextRunner.run(
            starting_agent=self,
            input=instructions,
        )

        return result

    async def parse_output(self, run_result: RunResult) -> RunResult:
        """Apply legacy string parser only when no structured output is configured."""
        if self.output_parser and self.output_model is None:
            # import ipdb
            # ipdb.set_trace()
            run_result.final_output = self.output_parser(run_result.final_output)
        return run_result
