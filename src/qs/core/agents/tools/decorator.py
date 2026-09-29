from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial, wraps
from pathlib import Path
from typing import Any, Final

from qs.config.capabilities import CAPABILITY_FORMAT
from qs.core.models.tool import Tool
from qs.interfaces.tool_hook import AfterTool, BeforeTool, validate_tool_hooks

type ToolFunction = Callable[..., Any]

_ENTRYPOINT_ATTRIBUTE: Final[str] = CAPABILITY_FORMAT.deep_tool_attribute


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str | None
    description: str | None
    before: BeforeTool | None = None
    after: AfterTool | None = None
    background_enabled: bool = True


def deep_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    before: BeforeTool | None = None,
    after: AfterTool | None = None,
    background_enabled: bool = True,
) -> Callable[[ToolFunction], ToolFunction]:
    _validate_definition(name, description)
    validate_tool_hooks(before, after)
    if not isinstance(background_enabled, bool):
        raise TypeError("background_enabled must be a bool")

    definition = ToolDefinition(
        name=name,
        description=description.strip() if description is not None else None,
        before=before,
        after=after,
        background_enabled=background_enabled,
    )

    def decorate(function: ToolFunction) -> ToolFunction:
        if not inspect.isfunction(function):
            raise TypeError("@deep_tool can decorate only functions")

        if get_tool_definition(function) is not None:
            raise ValueError(
                f"Function '{function.__name__}' is already decorated with @deep_tool"
            )

        _validate_work_root_parameter(function)

        @wraps(function)
        def decorated(work_root: Path, *args: Any, **kwargs: Any) -> Any:
            return function(work_root, *args, **kwargs)

        setattr(decorated, _ENTRYPOINT_ATTRIBUTE, definition)
        return decorated

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


def _validate_work_root_parameter(function: ToolFunction) -> None:
    parameters = tuple(inspect.signature(function).parameters.values())

    if (
        not parameters
        or parameters[0].name != "work_root"
        or parameters[0].kind
        not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ):
        raise TypeError(
            "@deep_tool entrypoint must define work_root as its first "
            "positional parameter"
        )


def get_tool_definition(
    function: ToolFunction,
) -> ToolDefinition | None:
    definition = getattr(function, _ENTRYPOINT_ATTRIBUTE, None)

    if isinstance(definition, ToolDefinition):
        return definition

    return None


def bind_deep_tool(
    function: ToolFunction,
    work_root: Path,
    *runtime_arguments: Any,
    name: str | None = None,
    cache_result: bool = True,
) -> Tool:
    """Bind runtime-only arguments while retaining decorator metadata."""
    definition = get_tool_definition(function)
    if definition is None or definition.description is None:
        raise ValueError("Tool function must have @deep_tool with a description")
    bound_function = partial(function, work_root, *runtime_arguments)
    bound_function.__doc__ = ""
    return Tool(
        bound_function,
        name=name or definition.name or function.__name__,
        description=definition.description,
        cache_result=cache_result,
        before=definition.before,
        after=definition.after,
        background_enabled=definition.background_enabled,
    )
