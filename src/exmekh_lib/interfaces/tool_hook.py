from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from exmekh_lib.interfaces.message import Message


@dataclass(frozen=True, slots=True)
class ToolContext:
    arguments: dict[str, Any]
    work_root: Path
    history: tuple[Message[object], ...]


@dataclass(frozen=True, slots=True)
class ToolArguments:
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResponse:
    result: Any


class BeforeTool(ABC):
    @abstractmethod
    def handle(self, context: ToolContext) -> ToolArguments | ToolResponse | None:
        """Change arguments, answer in place of the tool, or continue unchanged."""


class AfterTool(ABC):
    @abstractmethod
    def handle(self, context: ToolContext, result: Any) -> Any:
        """Return the result that should be exposed to the agent."""


def validate_tool_hooks(before: BeforeTool | None, after: AfterTool | None) -> None:
    if before is not None and not isinstance(before, BeforeTool):
        raise TypeError("before must be a BeforeTool instance")
    if after is not None and not isinstance(after, AfterTool):
        raise TypeError("after must be an AfterTool instance")
    for hook, argument_count in ((before, 1), (after, 2)):
        if hook is None:
            continue
        method = hook.handle
        if (
            inspect.iscoroutinefunction(method)
            or inspect.isgeneratorfunction(method)
            or inspect.isasyncgenfunction(method)
        ):
            raise TypeError("Tool hook handle must be synchronous")
        parameters = tuple(inspect.signature(method).parameters.values())
        if len(parameters) != argument_count or any(
            parameter.kind
            not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for parameter in parameters
        ):
            raise TypeError(
                f"Tool hook handle must accept {argument_count} argument(s)"
            )
