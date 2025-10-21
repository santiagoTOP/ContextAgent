"""
Miscellaneous helper utilities.
"""

import datetime
import json
from typing import Any, Optional

from pydantic import BaseModel


def get_experiment_timestamp() -> str:
    """Get timestamp for experiment naming."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def extract_final_output(result: Any) -> Any:
    """Extract final output from agent execution results.

    Handles various result types by checking for final_output attribute.

    Args:
        result: Agent execution result (may have final_output attribute)

    Returns:
        The final output if available, otherwise the result itself
    """
    return getattr(result, "final_output", result)


def serialize_content(value: Any) -> str:
    """Convert any value to a string representation suitable for display.

    Handles various types including BaseModel, dict, and primitives.

    Args:
        value: The value to serialize

    Returns:
        String representation of the value
    """
    if hasattr(value, 'output'):
        return str(value.output)
    elif isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    elif isinstance(value, dict):
        return json.dumps(value, indent=2)
    else:
        return str(value)


def parse_to_model(
    raw_output: Any,
    model_class: type[BaseModel],
    span: Optional[Any] = None
) -> BaseModel:
    """Parse raw output into a specified pydantic model.

    Handles various input types: BaseModel, dict, list, str, bytes.
    Optionally sets the output on a tracing span.

    Args:
        raw_output: The raw output to parse
        model_class: The pydantic model class to parse into
        span: Optional tracing span to set output on

    Returns:
        Parsed model instance
    """
    if isinstance(raw_output, model_class):
        output = raw_output
    elif isinstance(raw_output, BaseModel):
        output = model_class.model_validate(raw_output.model_dump())
    elif isinstance(raw_output, (dict, list)):
        output = model_class.model_validate(raw_output)
    elif isinstance(raw_output, (str, bytes, bytearray)):
        output = model_class.model_validate_json(raw_output)
    else:
        output = model_class.model_validate(raw_output)

    if span and hasattr(span, "set_output"):
        span.set_output(output.model_dump())

    return output
