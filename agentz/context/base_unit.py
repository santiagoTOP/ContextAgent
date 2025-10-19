from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple, List, Callable

def identity_wrapper(value: Any) -> Any:
    return value

class BaseContextContent(ABC): 
    content: Any = None
    preprocess_wrapper: Callable[[Any], Any] = identity_wrapper
    postprocess_wrapper: Callable[[Any], Any] = identity_wrapper

class BaseContextUnit(ABC):
    """Abstract base class for context components with lightweight storage.

    Subclasses can persist intermediate state (e.g., per-agent notes or metrics)
    without needing a full data store. Hooks are provided so implementations can
    validate or transform values before they are written.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, BaseContextContent] = {}

    def register_key(self, key: str) -> None:
        """Register a key under ``key``."""
        self._storage[key] = BaseContextContent()

    def store(self, key: str, value: Any|BaseContextContent) -> None:
        """Insert or replace a value under ``key``."""
        if isinstance(value, BaseContextContent):
            self._storage[key] = value
        else:
            self._storage[key] = BaseContextContent(content=value)

    def retrieve(self, key: str, default: Optional[BaseContextContent] = None) -> BaseContextContent:
        """Fetch a value stored under ``key``."""
        if key not in self._storage:
            if default is None:
                raise KeyError(f"Key '{key}' not found in context unit")
            return default
        return self._storage[key]

    def update(self, key: str, value: Any) -> Any:
        """Update an existing entry and return the normalized value."""
        if key not in self._storage:
            raise KeyError(f"Key '{key}' not found in context unit")
        normalized = self._normalize_value(key, value)
        self._storage[key] = normalized
        return normalized
    
    def entries(self) -> List[str]:
        """Return all stored entries."""
        return self._storage.keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Iterate over stored key/value pairs."""
        return self._storage.items()

    def clear(self) -> None:
        """Remove all stored entries."""
        self._storage.clear()

    # @abstractmethod
    def _normalize_value(self, key: str, value: Any) -> Any:
        """Hook for subclasses to validate or transform stored values."""
        return value
