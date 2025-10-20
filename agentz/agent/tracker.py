"""Runtime state tracking for agent execution operations."""

from contextlib import nullcontext, contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional

from agents.tracing.create import trace
from agentz.utils import Printer
from agentz.context.data_store import DataStore
from agentz.artifacts import RunReporter

# Context variable to store the current runtime tracker
# This allows tools to access the tracker without explicit parameter passing
_current_runtime_tracker: ContextVar[Optional['RuntimeTracker']] = ContextVar(
    'current_runtime_tracker',
    default=None
)


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
        enable_tracing: bool = True,
        trace_sensitive: bool = False,
        iteration: int = 0,
        experiment_id: Optional[str] = None,
    ):
        """Initialize runtime tracker.

        Args:
            console: Console instance for creating printer
            enable_tracing: Whether tracing is enabled
            trace_sensitive: Whether to include sensitive data in traces
            iteration: Current iteration number (for iterative workflows)
            experiment_id: Optional experiment ID for data store tracking
        """
        # Store dependencies for creating components
        self.console = console
        self.enable_tracing = enable_tracing
        self.trace_sensitive = trace_sensitive
        self.iteration = iteration
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
