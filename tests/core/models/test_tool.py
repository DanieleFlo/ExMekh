from typing import cast

import pytest

from qs.core.models.assistant_message import (
    FunctionCall,
    ToolCall,
)
from qs.core.models.tool import CalledTool, Tool


def documented_tool(query: str) -> str:
    """Search the documentation."""
    return query


def test_tool_uses_callable_metadata_by_default() -> None:
    tool = Tool(documented_tool)

    assert tool.tool is documented_tool
    assert tool.name == "documented_tool"
    assert tool.description == "Search the documentation."


def test_tool_accepts_explicit_metadata() -> None:
    tool = Tool(documented_tool, name="search-docs", description="Search docs")

    assert tool.name == "search-docs"
    assert tool.description == "Search docs"


def test_tool_creates_copy_with_description() -> None:
    tool = Tool(documented_tool, name="search-docs", description="Search docs")

    described_tool = tool.with_description("Available in skill 'docs'.")

    assert described_tool is not tool
    assert described_tool.tool is documented_tool
    assert described_tool.name == "search-docs"
    assert described_tool.description == "Available in skill 'docs'."
    assert tool.description == "Search docs"


def test_tool_creates_copy_with_description_suffix() -> None:
    tool = Tool(documented_tool, name="search-docs", description="Search docs.")

    described_tool = tool.with_description_suffix("Available in every skill.")

    assert described_tool.description == "Search docs. Available in every skill."


@pytest.mark.parametrize("name", ["", "contains spaces", "x" * 65])
def test_tool_rejects_invalid_openai_name(name: str) -> None:
    with pytest.raises(ValueError, match="Tool name"):
        Tool(documented_tool, name=name)


def test_tool_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="callable"):
        Tool("not-callable")


def test_tool_rejects_non_string_description() -> None:
    with pytest.raises(TypeError, match="description must be a string"):
        Tool(documented_tool, description=cast(str, object()))


def test_called_tool_executes_arguments_and_exposes_response_metadata() -> None:
    called_tool = CalledTool(
        Tool(documented_tool, name="search"),
        ToolCall(
            id="call-123",
            function=FunctionCall(
                name="search",
                arguments={"query": "clean code"},
            ),
        ),
    )

    assert called_tool.execute() == "clean code"
    assert called_tool.name == "search"
    assert called_tool.call_id == "call-123"


def test_called_tool_rejects_a_mismatched_name() -> None:
    with pytest.raises(ValueError, match="does not match"):
        CalledTool(
            Tool(documented_tool, name="search"),
            ToolCall(
                id="call-123",
                function=FunctionCall(name="other", arguments={}),
            ),
        )


def test_called_tool_rejects_invalid_types() -> None:
    tool = Tool(documented_tool)
    tool_call = ToolCall(
        id="call-123",
        function=FunctionCall(name="documented_tool", arguments={}),
    )

    with pytest.raises(TypeError, match="tool must be a Tool"):
        CalledTool(cast(Tool, object()), tool_call)
    with pytest.raises(TypeError, match="tool_call must be a ToolCall"):
        CalledTool.from_tool_call(tool, cast(ToolCall, object()))


def test_called_tool_factory_returns_valid_wrapper() -> None:
    tool = Tool(documented_tool)
    tool_call = ToolCall(
        id="call-123",
        function=FunctionCall(
            name="documented_tool",
            arguments={"query": "docs"},
        ),
    )

    assert CalledTool.from_tool_call(tool, tool_call).execute() == "docs"
