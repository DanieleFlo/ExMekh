import pytest

from qs.core.models.tool import Tool
from qs.core.models.tools import Tools


def first_tool() -> None:
    pass


def second_tool() -> None:
    pass


def test_tools_registers_and_returns_tools_in_order() -> None:
    first = Tool(first_tool)
    second = Tool(second_tool)
    tools = Tools([first, second])

    assert tools.list_tools() == ["first_tool", "second_tool"]
    assert tools.get_tools() == [first, second]
    assert tools.get_tool("second_tool") is second
    assert tools.get_tool("missing") is None


def test_tools_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        Tools([Tool(first_tool), Tool(second_tool, name="first_tool")])


def test_tools_rejects_invalid_collection() -> None:
    with pytest.raises(TypeError, match="list of Tool"):
        Tools([object()])


def test_tool_registry_uses_the_public_tool_name():
    class NamedTool(Tool):
        def get_name(self):
            return "public-name"

    tool = NamedTool(first_tool)
    tools = Tools([tool])

    assert tools.list_tools() == ["public-name"]
    assert tools.get_tool("public-name") is tool
    assert tools.get_tool("first_tool") is None
    with pytest.raises(ValueError, match="unique"):
        Tools([tool, NamedTool(second_tool)])
