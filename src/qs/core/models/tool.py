from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_type_hints

from qs.config.runtime import RUNTIME
from qs.config.validation import VALIDATION_PATTERNS
from qs.core.models.assistant_message import ToolCall
from qs.core.models.tool_hooks import ToolHooks
from qs.interfaces.tool_hook import (
    AfterTool,
    BeforeTool,
    ToolArguments,
    ToolContext,
    ToolResponse,
    validate_tool_hooks,
)


@dataclass(frozen=True, slots=True)
class PreformattedToolResult:
    """Tool content that must be returned without an additional XML header."""

    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("Preformatted tool content must be a string")


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    mode: Literal["wait", "background"] = "wait"
    wait_seconds: float = RUNTIME.default_tool_wait_seconds

    def __post_init__(self) -> None:
        if self.mode not in ("wait", "background"):
            raise ValueError("execution_mode must be 'wait' or 'background'")
        self.validate_wait_seconds(self.wait_seconds)

    @classmethod
    def from_arguments(cls, arguments: dict[str, Any]) -> ExecutionOptions:
        mode = arguments.get("execution_mode", "wait")
        wait_seconds = arguments.get("wait_seconds", RUNTIME.default_tool_wait_seconds)
        mode = "wait" if mode is None else mode
        wait_seconds = (
            RUNTIME.default_tool_wait_seconds if wait_seconds is None else wait_seconds
        )
        return cls(mode=mode, wait_seconds=wait_seconds)

    @staticmethod
    def validate_wait_seconds(wait_seconds: float) -> float:
        if isinstance(wait_seconds, bool) or not isinstance(wait_seconds, (int, float)):
            raise TypeError("wait_seconds must be a number")
        if not 0 < wait_seconds <= RUNTIME.max_tool_wait_seconds:
            raise ValueError(
                f"wait_seconds must be between 0 and {RUNTIME.max_tool_wait_seconds}"
            )
        return float(wait_seconds)


@dataclass(slots=True)
class ToolExecutionOperation:
    """One tool invocation; specialized tools may supply cancellation and output reads."""

    run: Callable[[], Any]
    cancel: Callable[[], bool] | None = None
    read_partial: Callable[[int], tuple[str, int]] | None = None


class Tool:
    def __init__(
        self,
        tool: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        cache_result: bool = True,
        background_enabled: bool = True,
        before: BeforeTool | None = None,
        after: AfterTool | None = None,
    ) -> None:
        self.tool = tool
        self._tool_type: Literal["function"] = "function"
        self._check_callable_tool()
        self.name = self._resolve_tool_name(tool, name)
        self.description = (
            (inspect.getdoc(tool) or "") if description is None else description
        )
        if not isinstance(cache_result, bool):
            raise TypeError("cache_result must be a bool")
        self.cache_result = cache_result
        if not isinstance(background_enabled, bool):
            raise TypeError("background_enabled must be a bool")
        self.background_enabled = background_enabled
        if (
            background_enabled
            and RUNTIME.execution_argument_names
            & inspect.signature(tool).parameters.keys()
        ):
            raise ValueError("Tool parameters cannot use reserved execution names")
        validate_tool_hooks(before, after)
        self.before = before
        self.after = after
        self._check_inputs()

    def _resolve_tool_name(self, tool: Callable[..., Any], name: str | None) -> str:
        resolved_name = getattr(tool, "__name__", "") if name is None else name
        if not self.is_valid_tool_name(resolved_name):
            raise ValueError(
                "Tool name must contain 1 to 64 letters, digits, underscores, or hyphens"
            )
        return resolved_name

    def _check_callable_tool(self) -> None:
        if not callable(self.tool):
            raise TypeError("Tool must be callable")

    def _check_inputs(self):
        if not isinstance(self.description, str):
            raise TypeError("Tool description must be a string")

    def is_valid_tool_name(self, name: str) -> bool:
        _TOOL_NAME_PATTERN = re.compile(VALIDATION_PATTERNS.tool_name)
        return bool(_TOOL_NAME_PATTERN.fullmatch(name))

    def execute(self, **arguments: Any) -> Any:
        return self.tool(**arguments)

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate the final arguments before invoking the tool."""
        inspect.signature(self.tool).bind(**arguments)

    def is_execution_argument(self, name: str) -> bool:
        return self.background_enabled and name in RUNTIME.execution_argument_names

    def normalize_call_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Null execution options use defaults; null business arguments remain intact."""
        return {
            name: value
            for name, value in arguments.items()
            if value is not None or not self.is_execution_argument(name)
        }

    def get_work_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            name: value
            for name, value in arguments.items()
            if not self.is_execution_argument(name)
        }

    def execute_with_hooks(
        self,
        arguments: dict[str, Any],
        *,
        work_root: Path,
        history: tuple[dict[str, str], ...],
        call_id: str = "",
    ) -> Any:
        return self.create_execution_with_hooks(
            arguments, call_id=call_id, work_root=work_root, history=history
        ).run()

    def _finalize_result(self, context: ToolContext, result: Any) -> Any:
        return self.after.handle(context, result) if self.after is not None else result

    def _prepare_context(
        self,
        arguments: dict[str, Any],
        work_root: Path,
        history: tuple[dict[str, str], ...],
    ) -> tuple[ToolContext, ToolResponse | None]:
        context = ToolContext(dict(arguments), work_root, history)
        decision = self.before.handle(context) if self.before is not None else None
        if isinstance(decision, ToolArguments):
            context = ToolContext(dict(decision.arguments), work_root, history)
            return context, None
        if decision is not None and not isinstance(decision, ToolResponse):
            raise TypeError(
                "BeforeTool.handle must return ToolArguments, ToolResponse, or None"
            )
        return context, decision

    def create_execution(
        self,
        arguments: dict[str, Any],
        *,
        call_id: str,
        work_root: Path,
        history: tuple[dict[str, str], ...],
    ) -> ToolExecutionOperation:
        """Create an execution handle; subclasses can add cancellation or progress."""
        return ToolExecutionOperation(run=lambda: self.execute(**arguments))

    def create_execution_with_hooks(
        self,
        arguments: dict[str, Any],
        *,
        call_id: str,
        work_root: Path,
        history: tuple[dict[str, str], ...],
    ) -> ToolExecutionOperation:
        """Apply hooks once, including for tools with custom execution handles."""

        def run() -> Any:
            context, decision = self._prepare_context(arguments, work_root, history)
            if isinstance(decision, ToolResponse):
                result = decision.result
            else:
                self.validate_arguments(context.arguments)
                work = self.create_execution(
                    context.arguments,
                    call_id=call_id,
                    work_root=work_root,
                    history=history,
                )
                operation.cancel = work.cancel
                # AfterTool filters complete results; raw progress must not bypass it.
                if self.after is None:
                    operation.read_partial = work.read_partial
                result = work.run()
            return self._finalize_result(context, result)

        operation = ToolExecutionOperation(run)
        return operation

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_signature(self) -> inspect.Signature:
        signature = inspect.signature(self.tool)
        if not self.background_enabled:
            return signature
        parameters = [
            inspect.Parameter(
                "execution_mode",
                inspect.Parameter.KEYWORD_ONLY,
                default="wait",
                annotation=Literal["wait", "background"],
            ),
            inspect.Parameter(
                "wait_seconds",
                inspect.Parameter.KEYWORD_ONLY,
                default=RUNTIME.default_tool_wait_seconds,
                annotation=float,
            ),
        ]
        original = list(signature.parameters.values())
        insert_at = next(
            (
                index
                for index, value in enumerate(original)
                if value.kind == inspect.Parameter.VAR_KEYWORD
            ),
            len(original),
        )
        return signature.replace(
            parameters=[*original[:insert_at], *parameters, *original[insert_at:]]
        )

    def get_type_hints(self) -> dict[str, Any]:
        callable_tool = self.tool.func if isinstance(self.tool, partial) else self.tool
        hints = get_type_hints(callable_tool, include_extras=True)
        if self.background_enabled:
            hints["execution_mode"] = Annotated[
                Literal["wait", "background"],
                "Wait up to wait_seconds, or return immediately while the tool continues.",
            ]
            hints["wait_seconds"] = Annotated[
                float,
                "Maximum initial wait in seconds; the tool continues after this time.",
            ]
        return hints

    def with_description(self, description: str) -> Self:
        return self._copy(name=self.name, description=description)

    def with_name(self, name: str) -> Self:
        """Copy a tool under another registered name, preserving its behavior."""
        return self._copy(name=name, description=self.description)

    def with_hooks(self, hooks: ToolHooks) -> Self:
        """Copy a tool with selected hooks replaced, preserving its behavior."""
        if not isinstance(hooks, ToolHooks):
            raise TypeError("hooks must be a ToolHooks instance")
        copied = copy(self)
        copied.before, copied.after = hooks.resolve(self.before, self.after)
        return copied

    def _copy(self, *, name: str, description: str) -> Self:
        copied = copy(self)
        copied.name = self._resolve_tool_name(self.tool, name)
        copied.description = description
        copied._check_inputs()
        return copied

    def without_background_execution(self) -> Self:
        """Keep structural tool overrides synchronous without mutating the source."""
        copied = copy(self)
        copied.background_enabled = False
        return copied

    def with_description_suffix(self, suffix: str) -> Self:
        description = " ".join(part for part in (self.description, suffix) if part)
        return self.with_description(description)

    def __str__(self) -> str:
        return f"Name: {self.name}\nType: Tool\nDescription: {self.description[:50]}..."

    def __repr__(self) -> str:
        return "\n---\n" + str(self)


class CalledTool:
    def __init__(self, tool: Tool, tool_call: ToolCall) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("tool must be a Tool instance")
        if not isinstance(tool_call, ToolCall):
            raise TypeError("tool_call must be a ToolCall instance")

        self._tool = tool
        self._tool_call = tool_call
        self._validate_matching()

    @classmethod
    def from_tool_call(cls, tool: Tool, tool_call: ToolCall) -> CalledTool:
        return cls(tool, tool_call)

    @property
    def name(self) -> str:
        return self._tool.get_name()

    @property
    def call_id(self) -> str:
        return self._tool_call.get_id()

    @property
    def arguments(self) -> dict[str, Any]:
        return self._tool_call.get_arguments()

    @property
    def tool_call(self) -> ToolCall:
        return self._tool_call

    @property
    def cache_result(self) -> bool:
        return self._tool.cache_result

    @property
    def background_enabled(self) -> bool:
        return self._tool.background_enabled

    @property
    def execution_options(self) -> ExecutionOptions:
        return ExecutionOptions.from_arguments(self.arguments)

    @property
    def work_arguments(self) -> dict[str, Any]:
        return self._tool.get_work_arguments(self.arguments)

    def execute(
        self,
        *,
        work_root: Path | None = None,
        history: tuple[dict[str, str], ...] = (),
    ) -> Any:
        return self._tool.execute_with_hooks(
            self.work_arguments,
            call_id=self.call_id,
            work_root=work_root if work_root is not None else Path.cwd(),
            history=history,
        )

    def create_execution(
        self,
        *,
        work_root: Path,
        history: tuple[dict[str, str], ...],
    ) -> ToolExecutionOperation:
        return self._tool.create_execution_with_hooks(
            self.work_arguments,
            call_id=self.call_id,
            work_root=work_root,
            history=history,
        )

    def _validate_matching(self) -> None:
        call_name = self._tool_call.get_name()
        if call_name != self.name:
            raise ValueError(
                f"Tool call name '{call_name}' does not match tool name '{self.name}'"
            )
