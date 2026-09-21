from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

type ExitToolFunction = Callable[..., Any]

_EXIT_TOOL_ATTRIBUTE: Final[str] = "__qs_exit_tool__"


@dataclass(frozen=True, slots=True)
class ExitToolDefinition:
    instructions_path: Path
    name: str | None
    description: str | None


def exit_tool(
    *,
    instructions_path: str | Path,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[ExitToolFunction], ExitToolFunction]:
    if not isinstance(instructions_path, (str, Path)):
        raise TypeError("instructions_path must be a string or Path")
    if isinstance(instructions_path, str) and not instructions_path.strip():
        raise ValueError("instructions_path cannot be blank")
    _validate_definition(name, description)
    definition = ExitToolDefinition(
        instructions_path=Path(instructions_path),
        name=name,
        description=description.strip() if description is not None else None,
    )

    def decorate(function: ExitToolFunction) -> ExitToolFunction:
        if not inspect.isfunction(function):
            raise TypeError("@exit_tool can decorate only functions")
        if get_exit_tool_definition(function) is not None:
            raise ValueError(
                f"Function '{function.__name__}' is already decorated with @exit_tool"
            )

        setattr(function, _EXIT_TOOL_ATTRIBUTE, definition)
        return function

    return decorate


def _validate_definition(name: str | None, description: str | None) -> None:
    if name is not None and not isinstance(name, str):
        raise TypeError("Tool name must be a string or None")
    if description is not None and not isinstance(description, str):
        raise TypeError("Tool description must be a string or None")
    if description is not None and not description.strip():
        raise ValueError("Tool description cannot be blank")


def get_exit_tool_definition(function: Any) -> ExitToolDefinition | None:
    definition = getattr(function, _EXIT_TOOL_ATTRIBUTE, None)
    return definition if isinstance(definition, ExitToolDefinition) else None
