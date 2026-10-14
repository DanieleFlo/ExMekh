from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Final

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.models.interactive import (
    InteractiveDefinition,
    validate_interactive_tool,
)
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool, validate_tool_hooks

type ToolFunction = Callable[..., Any]

_ENTRYPOINT_ATTRIBUTE: Final[str] = CAPABILITY_FORMAT.deep_tool_attribute


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str | None
    description: str | None
    before: BeforeTool | None = None
    after: AfterTool | None = None
    background_enabled: bool = True
    argument_guidance: str = ""
    interactive: InteractiveDefinition | None = None


def deep_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    before: BeforeTool | None = None,
    after: AfterTool | None = None,
    background_enabled: bool = True,
    argument_guidance: str = "",
    interactive: InteractiveDefinition | None = None,
) -> Callable[[ToolFunction], ToolFunction]:
    _validate_definition(name, description)
    validate_tool_hooks(before, after)
    if not isinstance(background_enabled, bool):
        raise TypeError(AGENT_TEXT.background_enabled_must_be_a_bool)
    if not isinstance(argument_guidance, str):
        raise TypeError(AGENT_TEXT.argument_guidance_must_be_a_string)
    validate_interactive_tool(interactive, background_enabled=background_enabled)

    definition = ToolDefinition(
        name=name,
        description=description.strip() if description is not None else None,
        before=before,
        after=after,
        background_enabled=background_enabled,
        argument_guidance=argument_guidance,
        interactive=interactive,
    )

    def decorate(function: ToolFunction) -> ToolFunction:
        if not inspect.isfunction(function):
            raise TypeError(AGENT_TEXT.deep_tool_can_decorate_only_functions)

        if get_tool_definition(function) is not None:
            raise ValueError(
                AGENT_TEXT.function_is_already_decorated_with_deep_tool.format(
                    function_name=function.__name__
                )
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
        raise TypeError(AGENT_TEXT.tool_name_must_be_a_string_or_none)

    if description is not None and not isinstance(description, str):
        raise TypeError(AGENT_TEXT.tool_description_must_be_a_string_or_none)

    if description is not None and not description.strip():
        raise ValueError(AGENT_TEXT.tool_description_cannot_be_blank)


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
        raise TypeError(AGENT_TEXT.deep_tool_entrypoint_must_define_work_root_as_its)


def get_tool_definition(
    function: ToolFunction,
) -> ToolDefinition | None:
    definition = getattr(function, _ENTRYPOINT_ATTRIBUTE, None)

    if isinstance(definition, ToolDefinition):
        return definition

    return None
