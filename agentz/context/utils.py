from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Optional, Tuple, List, Callable

def identity_wrapper(value: Any) -> Any:
    return value