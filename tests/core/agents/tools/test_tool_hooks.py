from pathlib import Path
from typing import Any

import pytest

from qs.core.agents.exit_tool.data import ExitToolData
from qs.core.agents.exit_tool.decorator import exit_tool, get_exit_tool_definition
from qs.core.agents.exit_tool.exit_tool import ExitTool
from qs.core.agents.tools.decorator import deep_tool, get_tool_definition
from qs.core.agents.tools.deep_tool import DeepTool
from qs.core.models.assistant_message import FunctionCall, ToolCall
from qs.core.models.tool import CalledTool, Tool
from qs.interfaces.tool_hook import (
    AfterTool,
    BeforeTool,
    ToolArguments,
    ToolContext,
    ToolResponse,
)


class ChangeArguments(BeforeTool):
    def handle(self, context: ToolContext) -> ToolArguments:
        assert context.arguments == {"value": "original"}
        assert context.history == ({"role": "user", "content": "call it"},)
        return ToolArguments({"value": "changed"})


class ChangeResult(AfterTool):
    def handle(self, context: ToolContext, result: Any) -> str:
        assert context.arguments == {"value": "changed"}
        return f"{result}!"


def test_hooks_change_arguments_and_result(tmp_path: Path) -> None:
    calls: list[str] = []
    before = ChangeArguments()
    after = ChangeResult()

    @deep_tool(description="Test", before=before, after=after)
    def entrypoint(work_root: Path, value: str) -> str:
        assert work_root == tmp_path
        calls.append(value)
        return value

    definition = get_tool_definition(entrypoint)
    assert definition is not None
    assert definition.before is before
    assert definition.after is after
    tool = DeepTool(entrypoint, tmp_path)
    result = tool.execute_with_hooks(
        {"value": "original"},
        work_root=tmp_path,
        history=({"role": "user", "content": "call it"},),
    )
    assert result == "changed!"
    assert calls == ["changed"]
    assert tuple(tool.get_signature().parameters) == (
        "value",
        "execution_mode",
        "wait_seconds",
    )


def test_before_can_answer_without_calling_tool(tmp_path: Path) -> None:
    class AnswerEarly(BeforeTool):
        def handle(self, context: ToolContext) -> ToolResponse:
            assert context.work_root == tmp_path
            return ToolResponse(None)

    class FormatAnswer(AfterTool):
        def handle(self, context: ToolContext, result: Any) -> str:
            assert result is None
            return "handled"

    def unreachable(value: str) -> str:
        pytest.fail("tool should not run")

    tool = Tool(unreachable, before=AnswerEarly(), after=FormatAnswer())
    assert (
        tool.execute_with_hooks({"value": "x"}, work_root=tmp_path, history=())
        == "handled"
    )


def test_called_tool_applies_hooks_without_explicit_context() -> None:
    class PrefixResult(AfterTool):
        def handle(self, context: ToolContext, result: Any) -> str:
            assert context.work_root == Path.cwd()
            return f"hooked: {result}"

    tool = Tool(lambda value: value, name="original", after=PrefixResult())
    renamed = tool.with_name("renamed")
    called = CalledTool(
        renamed, ToolCall("call-1", FunctionCall("renamed", {"value": "ok"}))
    )

    assert renamed.after is tool.after
    assert called.execute() == "hooked: ok"


def test_invalid_final_arguments_are_rejected_before_tool_runs(tmp_path: Path) -> None:
    calls: list[str] = []

    def entrypoint(value: str) -> str:
        calls.append(value)
        return value

    tool = Tool(entrypoint)
    with pytest.raises(TypeError, match="missing a required argument"):
        tool.execute_with_hooks({}, work_root=tmp_path, history=())
    assert calls == []


def test_rejects_invalid_hooks_at_registration() -> None:
    with pytest.raises(TypeError, match="before must be a BeforeTool"):
        deep_tool(before=object())

    with pytest.raises(TypeError, match="after must be an AfterTool"):
        deep_tool(after=object())

    class WrongSignature(BeforeTool):
        def handle(self) -> None:
            pass

    with pytest.raises(TypeError, match="must accept 1 argument"):
        deep_tool(before=WrongSignature())

    class AsyncAfter(AfterTool):
        async def handle(self, context: ToolContext, result: Any) -> Any:
            return result

    with pytest.raises(TypeError, match="must be synchronous"):
        deep_tool(after=AsyncAfter())


@pytest.mark.parametrize("managed", [False, True])
def test_invalid_before_hook_response_prevents_execution(tmp_path, managed) -> None:
    class InvalidResponse(BeforeTool):
        def handle(self, context: ToolContext):
            return "unsupported response"

    def work() -> None:
        pytest.fail("An invalid hook response must stop the tool")

    tool = Tool(work, before=InvalidResponse())
    with pytest.raises(TypeError, match="BeforeTool.handle must return"):
        if managed:
            operation = tool.create_execution_with_hooks(
                {}, call_id="call", work_root=tmp_path, history=()
            )
            operation.run()
        else:
            tool.execute_with_hooks({}, work_root=tmp_path, history=())


def test_managed_before_hook_can_answer_without_creating_work(tmp_path) -> None:
    after_results = []

    class AnswerEarly(BeforeTool):
        def handle(self, context: ToolContext) -> ToolResponse:
            return ToolResponse("shortcut")

    class FormatAnswer(AfterTool):
        def handle(self, context: ToolContext, result: Any) -> str:
            after_results.append(result)
            return f"public: {result}"

    def work() -> None:
        pytest.fail("The before hook already answered")

    operation = Tool(
        work, before=AnswerEarly(), after=FormatAnswer()
    ).create_execution_with_hooks({}, call_id="call", work_root=tmp_path, history=())

    assert operation.run() == "public: shortcut"
    assert after_results == ["shortcut"]
    assert operation.cancel is None and operation.read_partial is None


def test_exit_hooks_can_fix_arguments_and_change_output(tmp_path: Path) -> None:
    class FixArguments(BeforeTool):
        def handle(self, context: ToolContext) -> ToolArguments:
            return ToolArguments({"text": "finished", "items": []})

    class ChangeOutput(AfterTool):
        def handle(self, context: ToolContext, result: Any) -> dict[str, Any]:
            return {**result, "text": "changed"}

    @exit_tool(
        instructions_path="instructions.md", before=FixArguments(), after=ChangeOutput()
    )
    def finish(text: str, items: list[str]) -> ExitToolData:
        return ExitToolData(text, items)

    definition = get_exit_tool_definition(finish)
    assert definition is not None
    tool = ExitTool(
        finish,
        name="finish",
        description="Finish",
        instructions_path=tmp_path / "instructions.md",
        instructions="Finish the task.",
        folder_path=tmp_path,
        before=definition.before,
        after=definition.after,
    )
    assert tool.execute_with_hooks(
        {"text": "", "items": []}, work_root=tmp_path, history=()
    ) == {"text": "changed", "items": []}
