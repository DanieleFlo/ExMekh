from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool, validate_tool_hooks

type ExitToolFunction = Callable[..., Any]

_EXIT_TOOL_ATTRIBUTE: Final[str] = CAPABILITY_FORMAT.exit_tool_attribute


@dataclass(frozen=True, slots=True)
class ExitToolDefinition:
    instructions_path: Path
    name: str | None
    description: str | None
    before: BeforeTool | None = None
    after: AfterTool | None = None


def exit_tool(
    *,
    instructions_path: str | Path,
    name: str | None = None,
    description: str | None = None,
    before: BeforeTool | None = None,
    after: AfterTool | None = None,
) -> Callable[[ExitToolFunction], ExitToolFunction]:
    if not isinstance(instructions_path, (str, Path)):
        raise TypeError(AGENT_TEXT.instructions_path_must_be_a_string_or_path)
    if isinstance(instructions_path, str) and not instructions_path.strip():
        raise ValueError(AGENT_TEXT.instructions_path_cannot_be_blank)
    _validate_definition(name, description)
    validate_tool_hooks(before, after)
    definition = ExitToolDefinition(
        instructions_path=Path(instructions_path),
        name=name,
        description=description.strip() if description is not None else None,
        before=before,
        after=after,
    )

    def decorate(function: ExitToolFunction) -> ExitToolFunction:
        if not inspect.isfunction(function):
            raise TypeError(AGENT_TEXT.exit_tool_can_decorate_only_functions)
        if get_exit_tool_definition(function) is not None:
            raise ValueError(
                AGENT_TEXT.function_is_already_decorated_with_exit_tool.format(
                    function_name=function.__name__
                )
            )

        setattr(function, _EXIT_TOOL_ATTRIBUTE, definition)
        return function

    return decorate


def _validate_definition(name: str | None, description: str | None) -> None:
    if name is not None and not isinstance(name, str):
        raise TypeError(AGENT_TEXT.tool_name_must_be_a_string_or_none)
    if description is not None and not isinstance(description, str):
        raise TypeError(AGENT_TEXT.tool_description_must_be_a_string_or_none)
    if description is not None and not description.strip():
        raise ValueError(AGENT_TEXT.tool_description_cannot_be_blank)


def get_exit_tool_definition(function: Any) -> ExitToolDefinition | None:
    definition = getattr(function, _EXIT_TOOL_ATTRIBUTE, None)
    return definition if isinstance(definition, ExitToolDefinition) else None
