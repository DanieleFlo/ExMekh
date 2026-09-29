from pathlib import Path
from typing import cast

import pytest

from qs.core.agents.exit_tool.data import ExitToolData
from qs.core.agents.exit_tool.exit_tool import CalledExitTool, ExitTool
from qs.core.models.assistant_message import FunctionCall, ToolCall
from qs.interfaces.tool_hook import AfterTool, ToolContext


def _function(text: str, items: list[str]) -> ExitToolData:
    return ExitToolData(text=text, items=items)


def _exit_tool(tmp_path: Path) -> ExitTool:
    return ExitTool(
        _function,
        name="exit_agent",
        description="Exit",
        instructions_path=tmp_path / "instructions.md",
        instructions="Finish now",
        folder_path=tmp_path,
    )


@pytest.mark.parametrize(
    ("text", "items", "error_type", "message"),
    [
        (cast(str, None), ["item"], TypeError, "text must be a string"),
        ("   ", ["item"], ValueError, "text cannot be blank"),
        ("done", cast(list[str], "item"), TypeError, "items must be a list"),
        ("done", [cast(str, 1)], TypeError, "items must contain only strings"),
        ("done", ["   "], ValueError, "items cannot contain blank strings"),
    ],
)
def test_exit_tool_data_rejects_invalid_values(
    text: str,
    items: list[str],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        ExitToolData(text=text, items=items)


def test_exit_tool_data_strips_and_dumps_values() -> None:
    data = ExitToolData(text=" done ", items=[" first ", "second"])

    assert data.dump() == {"text": "done", "items": ["first", "second"]}
    assert ExitToolData(text="done", items=[]).dump() == {
        "text": "done",
        "items": [],
    }


def test_exit_tool_executes_and_validates_return_type(tmp_path: Path) -> None:
    tool = _exit_tool(tmp_path)

    assert (
        tool.render_prompt_section()
        == '<exit-tool name="exit_agent">Finish now</exit-tool>'
    )
    assert tool.execute(" done ", [" item "]) == {
        "text": "done",
        "items": ["item"],
    }

    invalid = ExitTool(
        lambda: "wrong",
        name="invalid",
        description="Invalid",
        instructions_path=tmp_path / "instructions.md",
        instructions="Finish now",
        folder_path=tmp_path,
    )
    with pytest.raises(TypeError, match="must return ExitToolData"):
        invalid.execute()


@pytest.mark.parametrize("managed", [False, True])
def test_exit_tool_rejects_a_non_dictionary_after_hook_result(
    tmp_path: Path, managed: bool
) -> None:
    class InvalidOutput(AfterTool):
        def handle(self, context: ToolContext, result):
            return "invalid exit output"

    tool = _exit_tool(tmp_path)
    tool.after = InvalidOutput()

    with pytest.raises(TypeError, match="Exit tool result must be a dictionary"):
        arguments = {"text": "done", "items": []}
        if managed:
            tool.create_execution_with_hooks(
                arguments, call_id="exit", work_root=tmp_path, history=()
            ).run()
        else:
            tool.execute_with_hooks(arguments, work_root=tmp_path, history=())


@pytest.mark.parametrize("managed", [False, True])
def test_exit_tool_normalizes_the_after_hook_result_once(
    tmp_path: Path, managed: bool
) -> None:
    results = []

    class ChangeOutput(AfterTool):
        def handle(self, context: ToolContext, result):
            results.append(result)
            return {"text": " changed ", "items": [" item "]}

    tool = _exit_tool(tmp_path)
    tool.after = ChangeOutput()
    arguments = {"text": "done", "items": []}

    if managed:
        result = tool.create_execution_with_hooks(
            arguments, call_id="exit", work_root=tmp_path, history=()
        ).run()
    else:
        result = tool.execute_with_hooks(arguments, work_root=tmp_path, history=())

    assert result == {"text": "changed", "items": ["item"]}
    assert results == [{"text": "done", "items": []}]


@pytest.mark.parametrize("managed", [False, True])
def test_exit_tool_rejects_invalid_fields_from_after_hooks(
    tmp_path: Path, managed: bool
) -> None:
    class InvalidOutput(AfterTool):
        def handle(self, context: ToolContext, result):
            return {"text": "done", "items": "invalid"}

    tool = _exit_tool(tmp_path)
    tool.after = InvalidOutput()

    with pytest.raises(TypeError, match="items must be a list"):
        arguments = {"text": "done", "items": []}
        if managed:
            tool.create_execution_with_hooks(
                arguments, call_id="exit", work_root=tmp_path, history=()
            ).run()
        else:
            tool.execute_with_hooks(arguments, work_root=tmp_path, history=())


def test_called_exit_tool_validates_types_and_exposes_call_id(tmp_path: Path) -> None:
    tool = _exit_tool(tmp_path)
    tool_call = ToolCall(
        "call-1",
        FunctionCall("exit_agent", {"text": "done", "items": ["a"]}),
    )

    called = CalledExitTool(tool, tool_call)

    assert called.name == "exit_agent"
    assert called.call_id == "call-1"
    assert called.tool_call is tool_call
    assert called.arguments == {"text": "done", "items": ["a"]}
    called.validate_arguments()
    assert called.execute() == {"text": "done", "items": ["a"]}

    invalid_arguments = CalledExitTool(
        tool,
        ToolCall("call-2", FunctionCall("exit_agent", {"text": "done", "items": 1})),
    )
    with pytest.raises(TypeError, match="items must be a list"):
        invalid_arguments.validate_arguments()

    with pytest.raises(TypeError, match="exit_tool must be an ExitTool"):
        CalledExitTool(cast(ExitTool, object()), tool_call)
    with pytest.raises(TypeError, match="tool_call must be a ToolCall"):
        CalledExitTool(tool, cast(ToolCall, object()))


@pytest.mark.parametrize(
    ("path", "instructions", "error", "message"),
    [
        ("instructions.md", "Finish", TypeError, "instructions_path must be a Path"),
        (Path("instructions.md"), None, TypeError, "instructions must be a string"),
        (Path("instructions.md"), " \t\n", ValueError, "cannot be blank"),
    ],
)
def test_rejects_invalid_instruction_inputs(path, instructions, error, message):
    with pytest.raises(error, match=message):
        ExitTool(
            _function,
            name="exit_agent",
            description="Exit",
            folder_path=Path("."),
            instructions_path=path,
            instructions=instructions,
        )
