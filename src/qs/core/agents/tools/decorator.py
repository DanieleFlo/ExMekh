from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

type ToolFunction = Callable[..., Any]

_ENTRYPOINT_ATTRIBUTE: Final[str] = "__qs_deep_tool__"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str | None
    description: str | None


def deep_tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[ToolFunction], ToolFunction]:
    _validate_definition(name, description)

    definition = ToolDefinition(
        name=name,
        description=description.strip() if description is not None else None,
    )

    def decorate(function: ToolFunction) -> ToolFunction:
        if not inspect.isfunction(function):
            raise TypeError("@deep_tool can decorate only functions")

        if get_tool_definition(function) is not None:
            raise ValueError(
                f"Function '{function.__name__}' is already decorated with @deep_tool"
            )

        setattr(function, _ENTRYPOINT_ATTRIBUTE, definition)
        return function

    return decorate


def _validate_definition(
    name: str | None,
    description: str | None,
) -> None:
    if name is not None and not isinstance(name, str):
        raise TypeError("Tool name must be a string or None")

    if description is not None and not isinstance(description, str):
        raise TypeError("Tool description must be a string or None")

    if description is not None and not description.strip():
        raise ValueError("Tool description cannot be blank")


def get_tool_definition(
    function: ToolFunction,
) -> ToolDefinition | None:
    definition = getattr(function, _ENTRYPOINT_ATTRIBUTE, None)

    if isinstance(definition, ToolDefinition):
        return definition

    return None
