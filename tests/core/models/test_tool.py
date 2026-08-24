import pytest

from qs.core.models.tool import Tool


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


@pytest.mark.parametrize("name", ["", "contains spaces", "x" * 65])
def test_tool_rejects_invalid_openai_name(name: str) -> None:
    with pytest.raises(ValueError, match="Tool name"):
        Tool(documented_tool, name=name)


def test_tool_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="callable"):
        Tool("not-callable")
