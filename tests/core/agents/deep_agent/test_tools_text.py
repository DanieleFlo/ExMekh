import json
from pathlib import Path
from unittest.mock import patch

import httpx2
import pytest
from openai import OpenAI

from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.node import DefaultCapabilities, Node
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.providers.local import LocalProvider
from tests.core.agents.deep_agent.test_deep_agent import (
    StubProvider,
    _StubTokenizer,
    _write_capabilities,
)
from tests.core.agents.node.test_node import _write_nested_capabilities


@pytest.fixture(params=("chat_completions", "responses"))
def local_provider(request):
    bodies: list[str] = []

    def handle(http_request: httpx2.Request) -> httpx2.Response:
        if http_request.method == "GET":
            return httpx2.Response(200, json={"data": [{"id": "model"}]})
        bodies.append(http_request.content.decode("utf-8"))
        if request.param == "chat_completions":
            response = {
                "id": "response-1",
                "created": 1,
                "model": "model",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            }
        else:
            response = {
                "id": "response-1",
                "created_at": 1,
                "model": "model",
                "object": "response",
                "status": "completed",
                "error": None,
                "output": [
                    {
                        "id": "message-1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "done", "annotations": []}
                        ],
                    }
                ],
            }
        return httpx2.Response(200, json=response)

    with (
        OpenAI(
            api_key="test",
            base_url="http://provider.test/v1",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handle)),
        ) as client,
        patch.object(LocalProvider, "_build_client", return_value=client),
    ):
        provider = LocalProvider("model", api=request.param, tokenizer=_StubTokenizer())
        yield provider, bodies


def request_tools_text(body: str) -> str:
    start = body.index('"tools":') + len('"tools":')
    _, end = json.JSONDecoder().raw_decode(body, start)
    return body[start:end]


@pytest.mark.parametrize("persistent_history", (True, False))
def test_tools_text_matches_http_payload_and_includes_contextual_and_runtime_tools(
    tmp_path: Path, local_provider, persistent_history: bool
) -> None:
    provider, bodies = local_provider
    capabilities_path = _write_nested_capabilities(tmp_path / "root")
    skill_path = capabilities_path / "deep-skill" / "mock-skill.md"
    skill_path.write_text(
        skill_path.read_text().replace(
            "description: Mock skill", 'description: Skill città <&> "ricerca" 🛠️'
        ),
        encoding="utf-8",
    )
    child = Node([_write_capabilities(tmp_path / "child", "child")], name_hds="child")
    root = Node(
        [capabilities_path],
        name_hds="main",
        children=[child],
        persistent_history=persistent_history,
    )
    agent = DeepAgent(provider=provider, node=root, work_root=tmp_path)

    text = agent.get_tools_text()
    assert agent.get_tools_text() == text
    assert bodies == []
    definitions = json.loads(text)
    functions = [item.get("function", item) for item in definitions]
    by_name = {item["name"]: item for item in functions}
    assert list(by_name) == [
        "mock-skill",
        "nested-skill",
        "mock-sub-tool",
        "mock-tool",
        "nested-sub-tool",
        "orphan-tool",
        "shared-tool",
        "exit_agent",
        "call_sub_agent",
        "get_execution_history",
        "search_tool_outputs",
        "read_tool_output",
        "control_tool_execution",
        "bash",
    ]
    assert by_name["mock-skill"]["description"] == 'Skill città <&> "ricerca" 🛠️'
    assert by_name["nested-skill"]["description"] == "Available in skill 'mock-skill'."
    assert by_name["mock-sub-tool"]["description"] == "Available in skill 'mock-skill'."
    assert "description" not in by_name["orphan-tool"]
    assert (
        "<description>child</description>" in by_name["call_sub_agent"]["description"]
    )
    assert by_name["call_sub_agent"]["parameters"]["properties"]["sub_agent"][
        "enum"
    ] == ["child"]
    assert "work_root" not in by_name["mock-tool"]["parameters"]["properties"]
    for name in ("mock-sub-tool", "nested-sub-tool", "shared-tool"):
        parameters = by_name[name]["parameters"]["properties"]
        assert "description" not in parameters["execution_mode"]
        assert "description" not in parameters["wait_seconds"]
    for name in ("mock-tool", "bash"):
        parameters = by_name[name]["parameters"]["properties"]
        assert "description" in parameters["execution_mode"]
        assert "description" in parameters["wait_seconds"]
    assert "città" in text
    assert "🛠️" in text
    assert "Mock skill body" not in text
    assert root.get_tools_text() == text
    assert "call_sub_agent" not in child.get_tools_text()

    for _ in range(2):
        assert agent.run("hello").get_message() == "done"
        assert request_tools_text(bodies[-1]) == text
        assert agent.get_tools_text() == text
    assert len(bodies) == 2
    first_messages = json.loads(bodies[0])
    messages = first_messages.get("messages", first_messages.get("input"))
    assert [message["role"] for message in messages] == ["system", "user"]


def test_tools_text_respects_excluded_runtime_tools(tmp_path: Path, local_provider):
    provider, bodies = local_provider
    root = Node(
        [_write_capabilities(tmp_path, "root")],
        name_hds="root",
        default_capabilities=DefaultCapabilities(harness=False),
    )
    agent = DeepAgent(provider=provider, node=root, work_root=tmp_path)

    text = agent.get_tools_text()
    assert len(json.loads(text)) == 1
    assert bodies == []
    agent.run("hello")
    assert request_tools_text(bodies[0]) == text


@pytest.mark.parametrize("tools", (None, Tools([])))
def test_tools_text_is_empty_when_request_omits_tools(local_provider, tools):
    provider, bodies = local_provider

    assert provider.get_tools_text(tools) == ""
    assert bodies == []


def test_tools_text_requires_initialized_node(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="root")

    with pytest.raises(RuntimeError, match="initialized before accessing tools"):
        node.get_tools_text()


def test_tools_text_reports_unsupported_provider(tmp_path: Path) -> None:
    provider = StubProvider()
    root = Node([_write_capabilities(tmp_path, "root")], name_hds="root")
    agent = DeepAgent(provider=provider, node=root, work_root=tmp_path)

    with pytest.raises(NotImplementedError, match="tools request serialization"):
        agent.get_tools_text()
    assert provider.last_request is None
