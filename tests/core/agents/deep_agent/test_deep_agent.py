from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest
from pydantic import BaseModel

from qs.core.agents.deep_agent.deep_agent import DeepAgent
from qs.core.agents.node import Node
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.interfaces.provider import Provider


class StubProvider(Provider):
    def __init__(self) -> None:
        self.last_request: ProviderRequest | None = None

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.last_request = request
        return ProviderResponse(AssistantMessage("done"))

    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        del request
        raise NotImplementedError


def _write_capabilities(parent: Path, hds_name: str) -> Path:
    capabilities_path = parent / "capabilities"
    hds_path = capabilities_path / "hds"
    exit_skill_path = capabilities_path / "exit-skill" / "finish.md"
    exit_tool_path = capabilities_path / "exit-tool" / "finish"
    hds_path.mkdir(parents=True)
    exit_skill_path.parent.mkdir(parents=True)
    exit_tool_path.mkdir(parents=True)
    (hds_path / f"{hds_name}.md").write_text(
        f"---\ndescription: {hds_name}\nexit-skill: finish\n---\n"
        f"{hds_name} instructions"
    )
    exit_skill_path.write_text("---\ndescription: Finish\n---\nFinish the agent")
    (exit_tool_path / "main.py").write_text(
        "from qs.core.agents.exit_tool.data import ExitToolData\n"
        "from qs.core.agents.exit_tool.decorator import exit_tool\n\n"
        "@exit_tool(name='exit_agent', description='Exit agent')\n"
        "def exit_agent(text: str, items: list[str]) -> ExitToolData:\n"
        "    return ExitToolData(text=text, items=items)\n"
    )
    return capabilities_path


def test_loads_and_validates_every_node_depth_first(tmp_path: Path) -> None:
    grandchild = Node(
        [_write_capabilities(tmp_path / "grandchild", "grandchild")],
        name_hds="grandchild",
    )
    child = Node(
        [_write_capabilities(tmp_path / "child", "child")],
        name_hds="child",
        children=[grandchild],
    )
    root = Node(
        [_write_capabilities(tmp_path / "root", "root")],
        name_hds="root",
        children=[child],
    )

    agent = DeepAgent(provider=StubProvider(), node=root)

    assert agent.root_node is root
    assert isinstance(agent.capabilities_by_node, MappingProxyType)
    assert tuple(agent.capabilities_by_node) == (root, child, grandchild)
    assert tuple(
        capabilities.hds[0].name for capabilities in agent.capabilities_by_node.values()
    ) == ("root", "child", "grandchild")


def test_rejects_invalid_hds_in_a_descendant(tmp_path: Path) -> None:
    child = Node(
        [_write_capabilities(tmp_path / "child", "actual")],
        name_hds="missing",
    )
    root = Node(
        [_write_capabilities(tmp_path / "root", "root")],
        name_hds="root",
        children=[child],
    )

    with pytest.raises(ValueError, match="HDS 'missing' not found"):
        DeepAgent(provider=StubProvider(), node=root)


def test_shared_descendant_is_loaded_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = Node([tmp_path / "shared" / "capabilities"], name_hds="shared")
    first = Node(
        [tmp_path / "first" / "capabilities"],
        name_hds="first",
        children=[shared],
    )
    second = Node(
        [tmp_path / "second" / "capabilities"],
        name_hds="second",
        children=[shared],
    )
    root = Node(
        [tmp_path / "root" / "capabilities"],
        name_hds="root",
        children=[first, second],
    )
    loaded_nodes: list[Node] = []
    original_load_capabilities = Node.load_capabilities

    def load_capabilities(node: Node):
        loaded_nodes.append(node)
        valid_path = _write_capabilities(tmp_path / node.name_hds, node.name_hds)
        return original_load_capabilities(Node([valid_path], node.name_hds))

    monkeypatch.setattr(Node, "load_capabilities", load_capabilities)

    DeepAgent(provider=StubProvider(), node=root)

    assert loaded_nodes == [root, first, shared, second]


def test_rejects_cycle_in_node_hierarchy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Node([tmp_path / "capabilities"], name_hds="root")
    root._children = (root,)
    original_load_capabilities = Node.load_capabilities

    def load_capabilities(node: Node):
        valid_path = _write_capabilities(tmp_path / "valid", node.name_hds)
        return original_load_capabilities(Node([valid_path], node.name_hds))

    monkeypatch.setattr(Node, "load_capabilities", load_capabilities)

    with pytest.raises(ValueError, match="hierarchy cannot contain cycles"):
        DeepAgent(provider=StubProvider(), node=root)


def test_rejects_non_node_root() -> None:
    with pytest.raises(TypeError, match="node must be a Node"):
        DeepAgent(provider=StubProvider(), node=cast(Node, object()))


def test_rejects_non_provider_before_loading_node(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="root")

    with pytest.raises(TypeError, match="provider must be a Provider"):
        DeepAgent(provider=cast(Provider, object()), node=node)


def test_uses_root_hds_as_system_instructions(tmp_path: Path) -> None:
    provider = StubProvider()
    root = Node(
        [_write_capabilities(tmp_path, "root")],
        name_hds="root",
    )
    agent = DeepAgent(provider=provider, node=root)

    response = agent.run("hello")

    assert response.get_message() == "done"
    assert response.get_history_tree() == []
    assert provider.last_request is not None
    assert provider.last_request.get_messages()[0].message_dump() == {
        "role": "system",
        "content": "root instructions",
    }
    tools = provider.last_request.get_tools()
    assert tools is not None
    assert tools.list_tools() == ["finish", "exit_agent"]
