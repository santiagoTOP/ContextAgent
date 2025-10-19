import asyncio
import functools
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

from loguru import logger
from rich.console import Console

from agents.tracing.create import function_span
from agentz.utils.config import BaseConfig, resolve_config
from agentz.runner import (
    AgentExecutor,
    RuntimeTracker,
    execute_tools,
)
from agentz.artifacts import RunReporter
from agentz.utils import Printer, get_experiment_timestamp



class BasePipeline:
    """Base class for all pipelines with common configuration and setup."""

    # Constants for iteration group IDs
    ITERATION_GROUP_PREFIX = "iter"
    FINAL_GROUP_ID = "iter-final"

    def __init__(self, config: Union[str, Path, Mapping[str, Any], BaseConfig]):
        """Initialize the pipeline using a single configuration input.

        Args:
            spec: Configuration specification:
                - str/Path: Load YAML/JSON file
                - dict with 'config_path': Load file, then deep-merge dict on top (dict wins)
                - dict without 'config_path': Use as-is
                - BaseConfig: Use as-is
            strict: Whether to strictly validate configuration (default: True).

        Examples:
            # Load from file
            BasePipeline("pipelines/configs/data_science.yaml")

            # Dict without config_path
            BasePipeline({"provider": "openai", "data": {"path": "data.csv"}})

            # Dict that patches a file (use 'config_path')
            BasePipeline({
                "config_path": "pipelines/configs/data_science.yaml",
                "data": {"path": "data/banana_quality.csv"},
                "user_prompt": "Custom prompt..."
            })

            # BaseConfig object
            BasePipeline(BaseConfig(provider="openai", data={"path": "data.csv"}))
        """
        self.console = Console()
        self._printer: Optional[Printer] = None
        self.reporter: Optional[RunReporter] = None

        # Resolve configuration using the new unified API
        self.config = resolve_config(config)

        # Generic pipeline settings
        self.experiment_id = get_experiment_timestamp()

        pipeline_settings = self.config.pipeline
        default_slug = self.__class__.__name__.replace("Pipeline", "").lower()
        self.pipeline_slug = (
            pipeline_settings.get("slug")
            or pipeline_settings.get("name")
            or default_slug
        )
        self.workflow_name = (
            pipeline_settings.get("workflow_name")
            or pipeline_settings.get("name")
        )
        if not self.workflow_name:
            # Default pattern: use class name + experiment_id
            pipeline_name = self.__class__.__name__.replace("Pipeline", "").lower()
            self.workflow_name = f"{pipeline_name}_{self.experiment_id}"

        self.verbose = pipeline_settings.get("verbose", True)
        self.max_iterations = pipeline_settings.get("max_iterations", 5)
        self.max_time_minutes = pipeline_settings.get("max_time_minutes", 10)

        # Research workflow name (optional, for pipelines with research components)
        self.research_workflow_name = pipeline_settings.get(
            "research_workflow_name",
            f"researcher_{self.experiment_id}",
        )

        # Iterative pipeline state
        self.iteration = 0
        self.start_time: Optional[float] = None
        self.should_continue = True
        self.constraint_reason = ""
        self._current_group_id: Optional[str] = None

        # Setup tracing configuration and logging
        self._setup_tracing()

        # Initialize runtime tracker and executor
        self._runtime_tracker: Optional[RuntimeTracker] = None
        self._executor: Optional[AgentExecutor] = None

    # ============================================
    # Core Properties
    # ============================================

    @property
    def enable_tracing(self) -> bool:
        """Get tracing enabled flag from config."""
        return self.config.pipeline.get("enable_tracing", True)

    @property
    def trace_sensitive(self) -> bool:
        """Get trace sensitive data flag from config."""
        return self.config.pipeline.get("trace_include_sensitive_data", False)

    @property
    def state(self) -> Optional[Any]:
        """Get pipeline state if available."""
        if hasattr(self, 'context') and hasattr(self.context, 'state'):
            return self.context.state
        return None

    @property
    def printer(self) -> Optional[Printer]:
        return self._printer

    @property
    def runtime_tracker(self) -> RuntimeTracker:
        """Get or create the runtime tracker."""
        if self._runtime_tracker is None:
            self._runtime_tracker = RuntimeTracker(
                printer=self.printer,
                enable_tracing=self.enable_tracing,
                trace_sensitive=self.trace_sensitive,
                iteration=self.iteration,
                experiment_id=self.experiment_id,
                reporter=self.reporter,
            )
        else:
            # Update iteration in existing tracker
            self._runtime_tracker.iteration = self.iteration
            self._runtime_tracker.printer = self.printer
            self._runtime_tracker.reporter = self.reporter
        return self._runtime_tracker

    @property
    def executor(self) -> AgentExecutor:
        """Get or create the agent executor."""
        # Refresh runtime tracker so iteration/printer stay in sync across loops
        tracker = self.runtime_tracker

        if self._executor is None:
            self._executor = AgentExecutor(tracker)
        else:
            # Executor holds a reference to the tracker; update it in case it changed
            self._executor.tracker = tracker
        return self._executor

    # ============================================
    # Printer & Reporter Management
    # ============================================

    def start_printer(self) -> Printer:
        if self._printer is None:
            self._printer = Printer(self.console)
        return self._printer

    def stop_printer(self) -> None:
        """Stop the live printer and finalize reporter if active."""
        if self._printer is not None:
            self._printer.end()
            self._printer = None
        if self.reporter is not None:
            self.reporter.finalize()
            self.reporter.print_terminal_report()

    def start_group(
        self,
        group_id: str,
        *,
        title: Optional[str] = None,
        border_style: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> None:
        """Start a printer group and notify the reporter."""
        if self.reporter:
            self.reporter.record_group_start(
                group_id=group_id,
                title=title,
                border_style=border_style,
                iteration=iteration,
            )
        if self.printer:
            self.printer.start_group(
                group_id,
                title=title,
                border_style=border_style,
            )

    def end_group(
        self,
        group_id: str,
        *,
        is_done: bool = True,
        title: Optional[str] = None,
    ) -> None:
        """Mark a printer group complete and notify the reporter."""
        if self.reporter:
            self.reporter.record_group_end(
                group_id=group_id,
                is_done=is_done,
                title=title,
            )
        if self.printer:
            self.printer.end_group(
                group_id,
                is_done=is_done,
                title=title,
            )

    # ============================================
    # Initialization & Setup
    # ============================================

    def _initialize_run(
        self,
        additional_logging: Optional[Callable] = None,
        enable_reporter: bool = True,
        outputs_dir: Optional[Union[str, Path]] = None,
        enable_printer: bool = True,
        workflow_name: Optional[str] = None,
        trace_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a pipeline run with logging, printer, and tracing.

        Args:
            additional_logging: Optional callable for pipeline-specific logging
            enable_reporter: Whether to create/start the RunReporter
            outputs_dir: Override outputs directory (None uses config value)
            enable_printer: Whether to start the Printer
            workflow_name: Override workflow name (None uses self.workflow_name)
            trace_metadata: Additional metadata to merge into trace context

        Returns:
            Trace context manager for the workflow
        """
        # Basic logging
        logger.info(
            f"Running {self.__class__.__name__} with experiment_id: {self.experiment_id}"
        )

        # Pipeline-specific logging
        if additional_logging:
            additional_logging()

        # Use workflow_name override if provided, otherwise use instance workflow_name
        effective_workflow_name = workflow_name or self.workflow_name

        # Conditionally create and start reporter
        if enable_reporter:
            # Use outputs_dir override if provided, otherwise use config value
            effective_outputs_dir = Path(outputs_dir) if outputs_dir else Path(self.config.pipeline.get("outputs_dir", "outputs"))

            if self.reporter is None:
                self.reporter = RunReporter(
                    base_dir=effective_outputs_dir,
                    pipeline_slug=self.pipeline_slug,
                    workflow_name=effective_workflow_name,
                    experiment_id=self.experiment_id,
                    console=self.console,
                )
            self.reporter.start(self.config)

        # Conditionally start printer and update workflow
        if enable_printer:
            self.start_printer()
            if self.printer:
                self.printer.update_item(
                    "workflow",
                    f"Workflow: {effective_workflow_name}",
                    is_done=True,
                    hide_checkmark=True,
                )

        # Create trace context with merged metadata
        base_trace_metadata = {
            "experiment_id": self.experiment_id,
            "includes_sensitive_data": "true" if self.trace_sensitive else "false",
        }

        # Merge custom trace_metadata if provided
        if trace_metadata:
            base_trace_metadata.update(trace_metadata)

        return self.trace_context(effective_workflow_name, metadata=base_trace_metadata)

    def _setup_tracing(self) -> None:
        """Setup tracing configuration with user-friendly output.

        Subclasses can override this method to add pipeline-specific information.
        """
        if self.enable_tracing:
            pipeline_name = self.__class__.__name__.replace("Pipeline", "")
            self.console.print(f"🍌 Starting {pipeline_name} Pipeline with Tracing")
            self.console.print(f"🔧 Provider: {self.config.provider}")
            self.console.print(f"🤖 Model: {self.config.llm.model_name}")
            self.console.print("🔍 Tracing: Enabled")
            self.console.print(
                f"🔒 Sensitive Data in Traces: {'Yes' if self.trace_sensitive else 'No'}"
            )
            self.console.print(f"🏷️ Workflow: {self.workflow_name}")
        else:
            pipeline_name = self.__class__.__name__.replace("Pipeline", "")
            self.console.print(f"🍌 Starting {pipeline_name} Pipeline")
            self.console.print(f"🔧 Provider: {self.config.provider}")
            self.console.print(f"🤖 Model: {self.config.llm.model_name}")

    def trace_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a trace context - delegates to RuntimeTracker."""
        return self.runtime_tracker.trace_context(name, metadata=metadata)

    def span_context(self, span_factory, **kwargs):
        """Create a span context - delegates to RuntimeTracker."""
        return self.runtime_tracker.span_context(span_factory, **kwargs)

    async def agent_step(self, *args, **kwargs) -> Any:
        """Run an agent with span tracking and optional output parsing.

        Delegates to AgentExecutor.agent_step(). See AgentExecutor.agent_step() for full documentation.
        """
        return await self.executor.agent_step(*args, **kwargs)

    def update_printer(self, *args, **kwargs) -> None:
        """Update printer status if printer is active.

        Delegates to RuntimeTracker.update_printer(). See RuntimeTracker.update_printer() for full documentation.
        """
        self.runtime_tracker.update_printer(*args, **kwargs)

    # ============================================
    # Context Managers & Utilities
    # ============================================

    @contextmanager
    def run_context(
        self,
        additional_logging: Optional[Callable] = None,
        # Timer control
        start_timer: bool = True,
        # Reporter control
        enable_reporter: bool = True,
        outputs_dir: Optional[Union[str, Path]] = None,
        # Printer control
        enable_printer: bool = True,
        # Tracing control
        workflow_name: Optional[str] = None,
        trace_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Context manager for run lifecycle handling.

        Manages trace context initialization, printer lifecycle, and cleanup.
        Provides fine-grained control over pipeline components.

        Args:
            additional_logging: Optional callable for pipeline-specific logging
            start_timer: Whether to start the constraint checking timer (default: True)
            enable_reporter: Whether to create/start the RunReporter (default: True)
            outputs_dir: Override outputs directory (default: None, uses config value)
            enable_printer: Whether to start the live status Printer (default: True)
            workflow_name: Override workflow name for this run (default: None, uses self.workflow_name)
            trace_metadata: Additional metadata to merge into trace context (default: None)

        Yields:
            Trace context for the workflow
        """
        # Track which resources existed before initialization
        had_reporter = self.reporter is not None
        had_printer = self.printer is not None

        # Conditionally start pipeline timer for constraint checking
        if start_timer:
            self.start_time = time.time()

        trace_ctx = self._initialize_run(
            additional_logging=additional_logging,
            enable_reporter=enable_reporter,
            outputs_dir=outputs_dir,
            enable_printer=enable_printer,
            workflow_name=workflow_name,
            trace_metadata=trace_metadata,
        )

        # Track what was actually created (not pre-existing)
        created_reporter = enable_reporter and not had_reporter and self.reporter is not None
        created_printer = enable_printer and not had_printer and self.printer is not None

        try:
            with trace_ctx:
                yield trace_ctx
        finally:
            # Only cleanup resources that were created by this context
            # Note: stop_printer() handles both printer and reporter cleanup
            if created_printer or created_reporter:
                self.stop_printer()

    async def run_span_step(self, *args, **kwargs) -> Any:
        """Execute a step with span context and printer updates.

        Delegates to AgentExecutor.run_span_step(). See AgentExecutor.run_span_step() for full documentation.
        """
        return await self.executor.run_span_step(*args, **kwargs)

    # ============================================
    # Iteration & Group Management
    # ============================================

    def begin_iteration(
        self,
        title: Optional[str] = None,
        border_style: str = "white"
    ) -> Any:
        """Begin a new iteration with its associated group.

        Combines context.begin_iteration() + start_group() into a single call.
        Automatically manages the group_id internally.

        Args:
            title: Optional custom title (default: "Iteration {index}")
            border_style: Border style for the group (default: "white")

        Returns:
            The iteration record
        """
        iteration, group_id = self.context.begin_iteration()
        self.iteration = iteration.index
        self._current_group_id = group_id

        display_title = title or f"Iteration {iteration.index}"
        self.start_group(
            group_id,
            title=display_title,
            border_style=border_style,
            iteration=iteration.index,
        )

        return iteration

    def end_iteration(self, is_done: bool = True) -> None:
        """End the current iteration and its associated group.

        Combines context.mark_iteration_complete() + end_group() into a single call.
        Automatically uses the internally managed group_id.

        Args:
            is_done: Whether the iteration completed successfully (default: True)
        """
        self.context.mark_iteration_complete()
        self.end_group(self._current_group_id, is_done=is_done)
        self._current_group_id = None

    def begin_final_report(
        self,
        title: str = "Final Report",
        border_style: str = "white"
    ) -> None:
        """Begin the final report phase with its associated group.

        Combines context.begin_final_report() + start_group() into a single call.
        Automatically manages the group_id internally.

        Args:
            title: Title for the final report group (default: "Final Report")
            border_style: Border style for the group (default: "white")
        """
        _, group_id = self.context.begin_final_report()
        self._current_group_id = group_id
        self.start_group(group_id, title=title, border_style=border_style)

    def end_final_report(self, is_done: bool = True) -> None:
        """End the final report phase and its associated group.

        Combines context.mark_final_complete() + end_group() into a single call.
        Automatically uses the internally managed group_id.

        Args:
            is_done: Whether the final report completed successfully (default: True)
        """
        self.context.mark_final_complete()
        self.end_group(self._current_group_id, is_done=is_done)
        self._current_group_id = None

    def prepare_query(
        self,
        content: str,
        step_key: str = "prepare_query",
        span_name: str = "prepare_research_query",
        start_msg: str = "Preparing research query...",
        done_msg: str = "Research query prepared"
    ) -> str:
        """Prepare query/content with span context and printer updates.

        Args:
            content: The query/content to prepare
            step_key: Printer status key
            span_name: Name for the span
            start_msg: Start message for printer
            done_msg: Completion message for printer

        Returns:
            The prepared content
        """
        self.update_printer(step_key, start_msg)

        with self.span_context(function_span, name=span_name) as span:
            logger.debug(f"Prepared {span_name}: {content}")

            if span and hasattr(span, "set_output"):
                span.set_output({"output_preview": content[:200]})

        self.update_printer(step_key, done_msg, is_done=True)
        return content

    def _log_message(self, message: str) -> None:
        """Log a message using the configured logger."""
        logger.info(message)

    # ============================================
    # Execution Entry Points
    # ============================================

    def run_sync(self, *args, **kwargs):
        """Synchronous wrapper for the async run method."""
        return asyncio.run(self.run(*args, **kwargs))

    async def run(self, query: Any = None) -> Any:
        """Execute the pipeline - must be implemented by subclasses.

        Each pipeline implements its own complete execution logic.
        Use the utility methods and context managers provided by BasePipeline.

        Args:
            query: Optional query input (can be None for pipelines without input)

        Returns:
            Final result (pipeline-specific)

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError("Subclasses must implement run()")


    # ============================================
    # Integration with Runner Module
    # ============================================

    async def _execute_tools(
        self,
        route_plan: Any,
        tool_agents: Dict[str, Any],
        group_id: Optional[str] = None,
    ) -> None:
        """Execute tool agents based on routing plan.

        Delegates to runner.patterns.execute_tools.

        Args:
            route_plan: The routing plan (can be AgentSelectionPlan or other)
            tool_agents: Dict mapping agent names to agent instances
            group_id: Optional group ID for printer updates. If None, uses pipeline's current group.
        """
        effective_group_id = group_id if group_id is not None else self._current_group_id
        await execute_tools(
            route_plan=route_plan,
            tool_agents=tool_agents,
            group_id=effective_group_id,
            context=self.context,
            agent_step_fn=self.agent_step,
            update_printer_fn=self.update_printer,
        )


def autotracing(
    additional_logging: Optional[Callable] = None,
    start_timer: bool = True,
    enable_reporter: bool = True,
    outputs_dir: Optional[Union[str, Path]] = None,
    enable_printer: bool = True,
    workflow_name: Optional[str] = None,
    trace_metadata: Optional[Dict[str, Any]] = None,
):
    """Decorator factory that wraps async methods with run_context lifecycle management.

    This decorator provides automatic initialization and cleanup of pipeline resources
    (reporter, printer, tracing) without requiring explicit `with self.run_context():` usage.

    Args:
        additional_logging: Optional callable for pipeline-specific logging
        start_timer: Whether to start the constraint checking timer (default: True)
        enable_reporter: Whether to create/start the RunReporter (default: True)
        outputs_dir: Override outputs directory (default: None, uses config value)
        enable_printer: Whether to start the live status Printer (default: True)
        workflow_name: Override workflow name for this run (default: None, uses self.workflow_name)
        trace_metadata: Additional metadata to merge into trace context (default: None)

    Returns:
        Decorator that wraps the method with run_context lifecycle

    Usage:
        @autotracing()
        async def run(self, query: Any = None) -> Any:
            # Pipeline logic here - no 'with' statement needed
            pass

        @autotracing(enable_printer=False, start_timer=False)
        async def run_silent(self, query: Any = None) -> Any:
            # Runs without printer or timer
            pass

    Note:
        The existing `run_context()` context manager remains available for advanced use cases
        where explicit control over the context lifecycle is needed.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            with self.run_context(
                additional_logging=additional_logging,
                start_timer=start_timer,
                enable_reporter=enable_reporter,
                outputs_dir=outputs_dir,
                enable_printer=enable_printer,
                workflow_name=workflow_name,
                trace_metadata=trace_metadata,
            ):
                return await func(self, *args, **kwargs)
        return wrapper
    return decorator