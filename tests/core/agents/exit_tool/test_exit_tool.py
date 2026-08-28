from pathlib import Path
from typing import cast

import pytest

from qs.core.agents.exit_tool.data import ExitToolData
from qs.core.agents.exit_tool.exit_tool import CalledExitTool, ExitTool
from qs.core.models.assistant_message import FunctionCall, ToolCall


def _function(text: str, items: list[str]) -> ExitToolData:
    return ExitToolData(text=text, items=items)


def _exit_tool(tmp_path: Path) -> ExitTool:
    return ExitTool(
        _function,
        name="exit_agent",
        description="Exit",
        exit_skill_name="finish",
        folder_path=tmp_path,
    )


@pytest.mark.parametrize(
    ("text", "items", "error_type", "message"),
    [
        (cast(str, None), ["item"], TypeError, "text must be a string"),
        ("   ", ["item"], ValueError, "text cannot be blank"),
        ("done", cast(list[str], "item"), TypeError, "items must be a list"),
        ("done", [], ValueError, "items cannot be empty"),
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


def test_exit_tool_executes_and_validates_return_type(tmp_path: Path) -> None:
    tool = _exit_tool(tmp_path)

    assert tool.get_exit_skill_name() == "finish"
    assert tool.execute(" done ", [" item "]) == {
        "text": "done",
        "items": ["item"],
    }

    invalid = ExitTool(
        lambda: "wrong",
        name="invalid",
        description="Invalid",
        exit_skill_name="finish",
        folder_path=tmp_path,
    )
    with pytest.raises(TypeError, match="must return ExitToolData"):
        invalid.execute()


def test_called_exit_tool_validates_types_and_exposes_call_id(tmp_path: Path) -> None:
    tool = _exit_tool(tmp_path)
    tool_call = ToolCall(
        "call-1",
        FunctionCall("exit_agent", {"text": "done", "items": ["a"]}),
    )

    called = CalledExitTool(tool, tool_call)

    assert called.name == "exit_agent"
    assert called.call_id == "call-1"
    assert called.execute() == {"text": "done", "items": ["a"]}

    with pytest.raises(TypeError, match="exit_tool must be an ExitTool"):
        CalledExitTool(cast(ExitTool, object()), tool_call)
    with pytest.raises(TypeError, match="tool_call must be a ToolCall"):
        CalledExitTool(tool, cast(ToolCall, object()))
