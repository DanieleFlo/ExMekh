from unittest.mock import MagicMock
from xml.etree import ElementTree as ET

import pytest

from qs.core.agents.exit_tool.exit_tool import ExitTool
from qs.core.agents.node.agentic_scope import AgenticScope
from qs.core.agents.node.tool_call_executor import ToolCallExecutor
from qs.core.models.assistant_message import FunctionCall, ToolCall
from qs.core.models.client import Client
from qs.core.models.tool import CalledTool, Tool
from qs.core.tool_calls.execution_manager import ToolExecutionManager


@pytest.fixture
def make_executor():
    def build(manager=None):
        client = MagicMock(spec=Client)
        client.generate_id.return_value = "output"
        client.dump_history.return_value = []
        executor = ToolCallExecutor(
            client=client,
            exit_tool=MagicMock(spec=ExitTool),
            initial_depth=0,
            agentic_scope=MagicMock(spec=AgenticScope),
            exit_content_resolver=MagicMock(),
            execution_manager=manager,
        )
        return executor

    return build


def unavailable_annotation():
    raise RuntimeError("Thinking annotation unavailable")


@pytest.mark.parametrize("managed", [False, True])
def test_annotation_failure_preserves_the_tool_result(make_executor, managed):
    executor = make_executor(ToolExecutionManager() if managed else None)
    called = CalledTool(
        Tool(lambda: "done", name="work", background_enabled=managed),
        ToolCall("call", FunctionCall("work", {})),
    )

    result = executor.execute(
        called, thinking_annotation_provider=unavailable_annotation
    )

    assert result.raw_result == "done"
    assert result.response.get_tool_call_id() == "call"
    if managed:
        header, body = result.response.get_message().split("</header>\n", 1)
        root = ET.fromstring(header + "</header>")
        assert root.findtext("id") == result.output_id == "output"
        assert root.findtext("response-type") == "final"
        assert root.find("thinking") is None
        assert body == "done"
    else:
        assert result.response.get_message() == "done"
        assert result.output_id is None


def test_header_fallback_preserves_explicit_id_without_optional_response_type(
    make_executor,
):
    executor = make_executor()

    content, output_id = executor._format_output(
        "work",
        "# tool-authored output",
        source="Tool",
        output_id="canonical",
        thinking_annotation_provider=unavailable_annotation,
    )

    header, body = content.split("</header>\n", 1)
    root = ET.fromstring(header + "</header>")
    assert root.findtext("id") == output_id == "canonical"
    assert root.find("response-type") is None
    assert body == "# tool-authored output"
