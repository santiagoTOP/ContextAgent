from pydantic import BaseModel
from typing import Any, Callable
from agentz.context.utils import identity_wrapper

class BaseContextModule(BaseModel):
    """Base class for all context modules."""

    def get_with_wrapper(self, key: str, wrapper: Callable[[Any], Any] = identity_wrapper) -> Any:
        return wrapper(getattr(self, key))