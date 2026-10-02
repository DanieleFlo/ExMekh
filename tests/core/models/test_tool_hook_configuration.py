from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.interfaces.tool_hook import (
    AfterTool,
    BeforeTool,
    ToolArguments,
    ToolContext,
)


class StripText(BeforeTool):
    def handle(self, context: ToolContext) -> ToolArguments:
        return ToolArguments({"text": context.arguments["text"].strip()})


class ReplaceText(BeforeTool):
    def handle(self, context: ToolContext) -> ToolArguments:
        return ToolArguments({"text": "replacement"})


class Suffix(AfterTool):
    def __init__(self, suffix: str) -> None:
        self.suffix = suffix

    def handle(self, context: ToolContext, result: Any) -> str:
        return f"{result}{self.suffix}"


@pytest.mark.parametrize(
    ("hooks", "expected"),
    [
        (ToolHooks(), "input!"),
        (ToolHooks(after=None), "input"),
        (ToolHooks(before=None), " input !"),
        (ToolHooks(before=None, after=None), " input "),
        (ToolHooks(after=Suffix("?")), "input?"),
        (ToolHooks(before=ReplaceText()), "replacement!"),
    ],
)
def test_hook_copies_preserve_replace_and_remove_hooks(
    tmp_path: Path, hooks: ToolHooks, expected: str
) -> None:
    tool = Tool(
        lambda text: text,
        name="echo",
        description="Echo text",
        before=StripText(),
        after=Suffix("!"),
        cache_result=False,
        background_enabled=False,
    )

    configured = tool.with_hooks(hooks)

    assert configured is not tool
    assert configured.get_name() == tool.get_name()
    assert configured.get_description() == tool.get_description()
    assert configured.cache_result is False
    assert configured.background_enabled is False
    assert configured.get_signature() == tool.get_signature()
    assert (
        configured.execute_with_hooks(
            {"text": " input "}, work_root=tmp_path, history=()
        )
        == expected
    )
    assert (
        tool.execute_with_hooks({"text": " input "}, work_root=tmp_path, history=())
        == "input!"
    )


def test_hook_configuration_is_immutable() -> None:
    hooks = ToolHooks()
    with pytest.raises(FrozenInstanceError):
        hooks.after = None


@pytest.mark.parametrize("field", ["before", "after"])
def test_hook_configuration_rejects_invalid_instances(field: str) -> None:
    with pytest.raises(TypeError, match=f"{field} must be a"):
        ToolHooks(**{field: object()})


def test_hook_configuration_rejects_async_handlers() -> None:
    class AsyncAfter(AfterTool):
        async def handle(self, context: ToolContext, result: Any) -> Any:
            return result

    with pytest.raises(TypeError, match="must be synchronous"):
        ToolHooks(after=AsyncAfter())


def test_hook_configuration_rejects_wrong_handler_signatures() -> None:
    class WrongBefore(BeforeTool):
        def handle(self) -> None:
            pass

    with pytest.raises(TypeError, match="must accept 1 argument"):
        ToolHooks(before=WrongBefore())


def test_tool_rejects_invalid_hook_configuration() -> None:
    with pytest.raises(TypeError, match="hooks must be a ToolHooks instance"):
        Tool(lambda: "result", name="work").with_hooks(object())
