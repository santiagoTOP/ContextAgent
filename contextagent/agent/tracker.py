"""Runtime state tracking for agent execution operations."""

from contextlib import nullcontext, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional

from agents.tracing.create import trace
from contextagent.utils import Printer
from contextagent.context.data_store import DataStore
from contextagent.artifacts import RunReporter

# Context variable to store the current runtime tracker
# This allows tools to access the tracker without explicit parameter passing
_current_runtime_tracker: ContextVar[Optional['RuntimeTracker']] = ContextVar(
    'current_runtime_tracker',
    default=None
)


@dataclass
class AgentStepHandle:
    """Internal handle for coordinating tracker-managed agent step state."""

    step_id: Optional[str]
    span_factory: Any
    span_name: str
    span_kwargs: Dict[str, Any] = field(default_factory=dict)
    printer_key: Optional[str] = None
    full_printer_key: Optional[str] = None
    printer_title: Optional[str] = None
    printer_border_style: Optional[str] = None
    iteration_idx: int = 0
    start_time: float = 0.0
    span: Any = None
    agent_name: str = ""


def _derive_agent_metadata(
    agent: Any,
    span_name: Optional[str],
    printer_key: Optional[str],
    printer_title: Optional[str],
) -> tuple[str, str, Optional[str], Optional[str]]:
    """Resolve agent-related metadata with sensible defaults."""
    agent_name = getattr(agent, "name", getattr(agent, "__class__", type("obj", (), {})).__name__)

    resolved_span_name = span_name or str(agent_name)
    resolved_printer_key = printer_key or str(agent_name)

    if printer_title:
        resolved_printer_title = printer_title
    elif printer_key:
        resolved_printer_title = printer_key
    else:
        resolved_printer_title = str(agent_name).capitalize()

    return str(agent_name), resolved_span_name, resolved_printer_key, resolved_printer_title


class RuntimeTracker:
    """Manages runtime state and tracking for agent execution.

    This class encapsulates the runtime infrastructure needed for agent execution including:
    - Ownership of Printer and Reporter instances
    - Tracing configuration and context creation
    - Iteration tracking
    - Pipeline-scoped data store for sharing objects between agents

    RuntimeTracker is the single source of truth for all runtime infrastructure.
    """

    def __init__(
        self,
        console: Any,
        context: Optional[Any] = None,
        enable_tracing: bool = True,
        trace_sensitive: bool = False,
        experiment_id: Optional[str] = None,
    ):
        """Initialize runtime tracker.

        Args:
            console: Console instance for creating printer
            context: Optional context reference for accessing current iteration
            enable_tracing: Whether tracing is enabled
            trace_sensitive: Whether to include sensitive data in traces
            experiment_id: Optional experiment ID for data store tracking
        """
        # Store dependencies for creating components
        self.console = console
        self.context = context
        self.enable_tracing = enable_tracing
        self.trace_sensitive = trace_sensitive
        self.experiment_id = experiment_id

        # Components owned by tracker (created on-demand)
        self._printer: Optional[Printer] = None
        self._reporter: Optional[RunReporter] = None
        self.data_store = DataStore(experiment_id=experiment_id)

    @property
    def printer(self) -> Optional[Printer]:
        """Get the printer instance."""
        return self._printer

    @property
    def reporter(self) -> Optional[RunReporter]:
        """Get the reporter instance."""
        return self._reporter

    @property
    def current_iteration_index(self) -> int:
        """Get current iteration index from context (always fresh).

        Returns:
            Current iteration index, or 0 if no iteration is active
        """
        if self.context and hasattr(self.context, 'state'):
            try:
                return self.context.state.current_iteration.index
            except (ValueError, AttributeError):
                pass
        return 0

    def start_agent_step(
        self,
        *,
        agent: Any,
        span_name: Optional[str],
        span_factory,
        span_kwargs: Optional[Dict[str, Any]] = None,
        printer_key: Optional[str] = None,
        printer_title: Optional[str] = None,
        printer_border_style: Optional[str] = None,
    ) -> AgentStepHandle:
        """Initialize tracker artifacts for an agent step."""
        agent_name, resolved_span_name, resolved_printer_key, resolved_printer_title = _derive_agent_metadata(
            agent,
            span_name,
            printer_key,
            printer_title,
        )

        iteration_idx = self.current_iteration_index
        step_id: Optional[str] = None

        if self._reporter:
            step_id = f"{iteration_idx}-{resolved_span_name}-{time.time_ns()}"
            self._reporter.record_agent_step_start(
                step_id=step_id,
                agent_name=str(agent_name),
                span_name=resolved_span_name,
                iteration=iteration_idx,
                group_id=f"iter-{iteration_idx}" if iteration_idx > 0 else None,
                printer_title=resolved_printer_title,
            )

        full_printer_key: Optional[str] = None
        if resolved_printer_key:
            full_printer_key = f"iter:{iteration_idx}:{resolved_printer_key}"
            self.update_printer(
                full_printer_key,
                "Working...",
                title=resolved_printer_title,
                border_style=printer_border_style,
            )

        return AgentStepHandle(
            step_id=step_id,
            span_factory=span_factory,
            span_name=resolved_span_name,
            span_kwargs=dict(span_kwargs or {}),
            printer_key=resolved_printer_key,
            full_printer_key=full_printer_key,
            printer_title=resolved_printer_title,
            printer_border_style=printer_border_style,
            iteration_idx=iteration_idx,
            start_time=time.perf_counter(),
            agent_name=str(agent_name),
        )

    def finish_agent_step(
        self,
        handle: AgentStepHandle,
        *,
        status: str,
        error: Optional[str] = None,
        success_message: str = "Completed",
        failure_message: str = "Failed",
    ) -> None:
        """Finalize tracker state for an agent step."""
        if handle.full_printer_key:
            message = success_message if status == "success" else failure_message
            self.update_printer(
                handle.full_printer_key,
                message,
                is_done=True,
                title=handle.printer_title,
                border_style=handle.printer_border_style,
            )

        if self._reporter and handle.step_id is not None:
            self._reporter.record_agent_step_end(
                step_id=handle.step_id,
                status=status,
                duration_seconds=time.perf_counter() - handle.start_time,
                error=error,
            )

    @contextmanager
    def span_scope(self, handle: AgentStepHandle):
        """Context manager for span lifecycle tied to an agent step handle."""
        kwargs = dict(handle.span_kwargs)
        kwargs.setdefault("name", handle.span_name)
        with self.span_context(handle.span_factory, **kwargs) as span:
            handle.span = span
            yield span

    def log_agent_panel(self, handle: AgentStepHandle, content: str) -> None:
        """Render a standalone panel for the agent output if configured."""
        if not content or not content.strip():
            return
        title = handle.printer_title or handle.printer_key
        if not title:
            return

        self.log_panel(
            title,
            content,
            border_style=handle.printer_border_style,
            iteration=handle.iteration_idx,
        )

    def preview_output(self, handle: AgentStepHandle, preview: str) -> None:
        """Attach an output preview to the active span."""
        if handle.span and hasattr(handle.span, "set_output"):
            handle.span.set_output({"output_preview": preview})

    def start_printer(self) -> Printer:
        """Create and return printer if not exists."""
        if self._printer is None:
            self._printer = Printer(self.console)
        return self._printer

    def stop_printer(self) -> None:
        """Stop printer and finalize reporter."""
        if self._printer is not None:
            self._printer.end()
            self._printer = None
        if self._reporter is not None:
            self._reporter.finalize()
            self._reporter.print_terminal_report()

    def initialize_reporter(
        self,
        base_dir: Any,
        pipeline_slug: str,
        workflow_name: str,
        experiment_id: str,
        config: Any,
    ) -> RunReporter:
        """Create and start reporter."""
        if self._reporter is None:
            self._reporter = RunReporter(
                base_dir=base_dir,
                pipeline_slug=pipeline_slug,
                workflow_name=workflow_name,
                experiment_id=experiment_id,
                console=self.console,
            )
        self._reporter.start(config)
        return self._reporter

    def start_group(
        self,
        group_id: str,
        *,
        title: Optional[str] = None,
        border_style: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> None:
        """Start a printer group and notify the reporter."""
        if self._reporter:
            self._reporter.record_group_start(
                group_id=group_id,
                title=title,
                border_style=border_style,
                iteration=iteration,
            )
        if self._printer:
            self._printer.start_group(
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
        """End a printer group and notify the reporter."""
        if self._reporter:
            self._reporter.record_group_end(
                group_id=group_id,
                is_done=is_done,
                title=title,
            )
        if self._printer:
            self._printer.end_group(
                group_id,
                is_done=is_done,
                title=title,
            )

    def trace_context(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a trace context manager.

        Args:
            name: Name for the trace
            metadata: Optional metadata to attach to trace

        Returns:
            Trace context manager if tracing enabled, otherwise nullcontext
        """
        if self.enable_tracing:
            return trace(name, metadata=metadata)
        return nullcontext()

    def span_context(self, span_factory, **kwargs):
        """Create a span context manager.

        Args:
            span_factory: Factory function for creating spans (agent_span or function_span)
            **kwargs: Arguments to pass to span factory

        Returns:
            Span context manager if tracing enabled, otherwise nullcontext
        """
        if self.enable_tracing:
            return span_factory(**kwargs)
        return nullcontext()

    def update_printer(
        self,
        key: str,
        message: str,
        is_done: bool = False,
        hide_checkmark: bool = False,
        title: Optional[str] = None,
        border_style: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> None:
        """Update printer status if printer is active.

        Args:
            key: Status key to update
            message: Status message
            is_done: Whether the task is complete
            hide_checkmark: Whether to hide the checkmark when done
            title: Optional panel title
            border_style: Optional border color
            group_id: Optional group to nest this item in
        """
        # Auto-derive group_id from current iteration if not explicitly provided
        if group_id is None:
            iteration_idx = self.current_iteration_index
            if iteration_idx > 0:
                group_id = f"iter-{iteration_idx}"

        if self.reporter:
            self.reporter.record_status_update(
                item_id=key,
                content=message,
                is_done=is_done,
                title=title,
                border_style=border_style,
                group_id=group_id,
            )
        if self.printer:
            self.printer.update_item(
                key,
                message,
                is_done=is_done,
                hide_checkmark=hide_checkmark,
                title=title,
                border_style=border_style,
                group_id=group_id
            )

    def log_panel(
        self,
        title: str,
        content: str,
        *,
        border_style: Optional[str] = None,
        iteration: Optional[int] = None,
        group_id: Optional[str] = None,
    ) -> None:
        """Proxy helper for rendering standalone panels via the printer."""
        # Auto-derive group_id from iteration if not provided
        if group_id is None and iteration is None:
            iteration = self.current_iteration_index if self.current_iteration_index > 0 else None

        if group_id is None and iteration is not None:
            group_id = f"iter-{iteration}"

        if self.reporter:
            self.reporter.record_panel(
                title=title,
                content=content,
                border_style=border_style,
                iteration=iteration,
                group_id=group_id,
            )
        if self.printer:
            self.printer.log_panel(
                title,
                content,
                border_style=border_style,
                iteration=iteration,
                group_id=group_id,
            )

    @contextmanager
    def activate(self):
        """Context manager to set this tracker as the current runtime tracker.

        This allows tools to access the tracker via get_current_tracker().

        Example:
            with tracker.activate():
                # Tools can now access this tracker
                result = await agent.run(...)
        """
        token = _current_runtime_tracker.set(self)
        try:
            yield self
        finally:
            _current_runtime_tracker.reset(token)


def get_current_tracker() -> Optional[RuntimeTracker]:
    """Get the current runtime tracker (if any).

    Returns:
        The current RuntimeTracker or None if not in a runtime context
    """
    return _current_runtime_tracker.get()


def get_current_data_store() -> Optional[DataStore]:
    """Get the data store from the current runtime tracker (if any).

    This is a convenience function for tools that need to access the data store.

    Returns:
        The current DataStore or None if not in a runtime context
    """
    tracker = get_current_tracker()
    return tracker.data_store if tracker else None
