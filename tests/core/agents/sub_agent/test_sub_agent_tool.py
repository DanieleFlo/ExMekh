import base64
from pathlib import Path
from unittest.mock import Mock
from xml.etree.ElementTree import fromstring

import pytest

from exmekh_lib.capabilities.tools.call_sub_agent.sub_agent_tool import SubAgentTool
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.openai_chat_completion_annotation_schema_builder import (
    OpenAIChatCompletionAnnotationSchemaBuilder,
)
from exmekh_lib.core.providers.openai_function_tool_definition_mapper import (
    OpenAIFunctionToolDefinitionMapper,
)
from exmekh_lib.core.providers.openai_response_tool_mapper import (
    OpenAIResponseToolMapper,
)


def _child(tmp_path: Path, name: str, response: str = "done") -> Node:
    child = Node([tmp_path / name / "capabilities"], name_hds=name)
    child.run = Mock(  # type: ignore[method-assign]
        return_value=ProviderResponse(AssistantMessage(response))
    )
    return child


def test_exposes_child_enum_and_non_empty_text_schema(tmp_path: Path) -> None:
    tool = SubAgentTool([_child(tmp_path, "writer"), _child(tmp_path, "reviewer")])
    mapper = OpenAIFunctionToolDefinitionMapper(
        OpenAIChatCompletionAnnotationSchemaBuilder()
    )

    definition = mapper.map(tool, strict=True)

    assert definition.get_name() == "call_sub_agent"
    assert tool.child_names == ("writer", "reviewer")
    assert definition.get_parameters() == {
        "type": "object",
        "properties": {
            "sub_agent": {
                "type": "string",
                "enum": ["writer", "reviewer"],
                "description": (
                    "Name of the direct sub-agent that should receive the message. "
                    "Available: writer, reviewer."
                ),
            },
            "text": {
                "type": "string",
                "description": (
                    "Non-empty user message to send to the selected sub-agent."
                ),
                "minLength": 1,
                "pattern": r"\S",
            },
        },
        "additionalProperties": False,
        "required": ["sub_agent", "text"],
    }
    assert "writer, reviewer" in definition.get_description()

    response_tool = OpenAIResponseToolMapper.map_tools(Tools([tool]))[0]
    assert response_tool["description"] == definition.get_description()
    assert response_tool["parameters"] == definition.get_parameters()

    tool.set_child_descriptions(
        {"writer": "Writes <drafts> & edits", "reviewer": "Reviews drafts"}
    )
    xml = tool.get_description().split("\n", 1)[1]
    assert [
        (agent.findtext("name"), agent.findtext("description"))
        for agent in fromstring(xml)
    ] == [
        ("writer", "Writes <drafts> & edits"),
        ("reviewer", "Reviews drafts"),
    ]


def test_delegates_exact_text_and_returns_final_response(tmp_path: Path) -> None:
    writer = _child(tmp_path, "writer", "child result")
    tool = SubAgentTool([writer])

    result = tool.execute(sub_agent="writer", text="write this")

    assert result == "child result"
    writer.run.assert_called_once_with("write this")  # type: ignore[attr-defined]


@pytest.mark.parametrize("arguments", [{}, {"sub_agent": "missing", "text": "hi"}])
def test_returns_available_children_for_invalid_selector(
    tmp_path: Path,
    arguments: dict[str, str],
) -> None:
    writer = _child(tmp_path, "writer")
    reviewer = _child(tmp_path, "reviewer")
    tool = SubAgentTool([writer, reviewer])

    assert tool.execute(**arguments) == (
        "Error: sub_agent must be one of: writer, reviewer."
        ' Use call_sub_agent(sub_agent="<one of: writer, reviewer>", text="<non-empty task>").'
    )
    writer.run.assert_not_called()  # type: ignore[attr-defined]
    reviewer.run.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.parametrize("text", [None, "", "   ", 1])
def test_returns_error_for_invalid_text(tmp_path: Path, text: object) -> None:
    tool = SubAgentTool([_child(tmp_path, "writer")])

    assert tool.execute(sub_agent="writer", text=text) == (
        "Error: text must be a non-empty string."
        ' Use call_sub_agent(sub_agent="<one of: writer>", text="<non-empty task>").'
    )


def test_returns_error_for_unexpected_arguments(tmp_path: Path) -> None:
    tool = SubAgentTool([_child(tmp_path, "writer")])

    assert tool.execute(sub_agent="writer", text="hi", extra=True) == (
        "Error: call_sub_agent accepts only the arguments sub_agent and text. "
        "Unexpected: extra."
        ' Use call_sub_agent(sub_agent="<one of: writer>", text="<non-empty task>").'
    )


def test_unexpected_argument_error_uses_registered_tool_name(tmp_path: Path) -> None:
    tool = SubAgentTool([_child(tmp_path, "writer")]).with_name("delegate_child")
    result = tool.execute(sub_agent="writer", text="hi", extra=True)
    assert "delegate_child accepts" in result
    assert "call_sub_agent" not in result


def test_requires_at_least_one_child() -> None:
    with pytest.raises(ValueError, match="requires at least one child"):
        SubAgentTool([])


def test_rejects_duplicate_child_hds_names(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError,
        match="Direct child HDS names must be unique: writer",
    ):
        SubAgentTool([_child(tmp_path, "writer"), _child(tmp_path, "writer")])


def test_raises_when_child_has_no_final_text(tmp_path: Path) -> None:
    child = _child(tmp_path, "writer")
    child.run.return_value = ProviderResponse(AssistantMessage(None))  # type: ignore[attr-defined]
    tool = SubAgentTool([child])

    with pytest.raises(RuntimeError, match="returned no final response text"):
        tool.execute(sub_agent="writer", text="hi")


def test_propagates_child_execution_errors(tmp_path: Path) -> None:
    child = _child(tmp_path, "writer")
    child.run.side_effect = RuntimeError("provider failed")  # type: ignore[attr-defined]
    tool = SubAgentTool([child])

    with pytest.raises(RuntimeError, match="provider failed"):
        tool.execute(sub_agent="writer", text="hi")


def test_can_be_registered_as_a_standard_tool(tmp_path: Path) -> None:
    tool = SubAgentTool([_child(tmp_path, "writer")])

    assert Tools([tool]).list_tools() == ["call_sub_agent"]


@pytest.mark.parametrize(
    "description", ["line\rreturn", "control\x00character", "surrogate\ud800"]
)
def test_child_descriptions_follow_shared_xml_encoding(tmp_path, description):
    tool = SubAgentTool([_child(tmp_path, "writer")])
    tool.set_child_descriptions({"writer": description})
    root = fromstring(tool.get_description().split("\n", 1)[1])
    element = root.find("sub_agent/description")
    assert element is not None
    actual = element.text
    if element.get("encoding") == "base64":
        actual = base64.b64decode(actual).decode("utf-8", "surrogatepass")
    assert actual == description
