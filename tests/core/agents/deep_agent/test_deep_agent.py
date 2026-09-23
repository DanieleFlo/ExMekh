import os
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest
from pydantic import BaseModel

from qs.core.agents.deep_agent.deep_agent import DeepAgent
from qs.core.agents.node import Node
from qs.core.cache.cache_manager import CacheManager
from qs.core.cache.session_cache import SessionCache
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.interfaces.provider import Provider
from qs.interfaces.tokenizer import Tokenizer


class _StubTokenizer(Tokenizer):
    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

    def initialize(self) -> None:
        pass

    def generate_id(self) -> str:
        return "stub-id"

    def get_session(self) -> SessionCache:
        return CacheManager.get_session(
            "stub-deep-agent", "default", Path(".qs/tokenizers")
        )


class StubProvider(Provider):
    def __init__(self) -> None:
        super().__init__(_StubTokenizer())
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
    exit_tool_path = capabilities_path / "exit-tool" / "finish"
    hds_path.mkdir(parents=True)
    exit_tool_path.mkdir(parents=True)
    (hds_path / f"{hds_name}.md").write_text(
        f"---\ndescription: {hds_name}\nexit-tool: exit_agent\n---\n"
        f"{hds_name} instructions"
    )
    (exit_tool_path / "instructions.md").write_text("Finish the agent")
    (exit_tool_path / "main.py").write_text(
        "from qs.core.agents.exit_tool.data import ExitToolData\n"
        "from qs.core.agents.exit_tool.decorator import exit_tool\n\n"
        "@exit_tool(instructions_path='instructions.md', name='exit_agent', description='Exit agent')\n"
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

    agent = DeepAgent(provider=StubProvider(), node=root, work_root=tmp_path)

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
        DeepAgent(provider=StubProvider(), node=root, work_root=tmp_path)


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

    def load_capabilities(node: Node, *, work_root: Path):
        loaded_nodes.append(node)
        assert work_root == tmp_path.resolve()
        valid_path = _write_capabilities(tmp_path / node.name_hds, node.name_hds)
        return original_load_capabilities(
            Node([valid_path], node.name_hds),
            work_root=work_root,
        )

    monkeypatch.setattr(Node, "load_capabilities", load_capabilities)

    DeepAgent(provider=StubProvider(), node=root, work_root=tmp_path)

    assert loaded_nodes == [root, first, shared, second]


def test_rejects_cycle_in_node_hierarchy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Node([tmp_path / "capabilities"], name_hds="root")
    root._children = (root,)
    original_load_capabilities = Node.load_capabilities

    def load_capabilities(node: Node, *, work_root: Path):
        valid_path = _write_capabilities(tmp_path / "valid", node.name_hds)
        return original_load_capabilities(
            Node([valid_path], node.name_hds),
            work_root=work_root,
        )

    monkeypatch.setattr(Node, "load_capabilities", load_capabilities)

    with pytest.raises(ValueError, match="hierarchy cannot contain cycles"):
        DeepAgent(provider=StubProvider(), node=root, work_root=tmp_path)


def test_rejects_non_node_root(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="node must be a Node"):
        DeepAgent(
            provider=StubProvider(),
            node=cast(Node, object()),
            work_root=tmp_path,
        )


def test_rejects_non_provider_before_loading_node(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="root")

    with pytest.raises(TypeError, match="provider must be a Provider"):
        DeepAgent(
            provider=cast(Provider, object()),
            node=node,
            work_root=tmp_path,
        )


def test_rejects_non_callable_stream_handler(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="root")

    with pytest.raises(TypeError, match="on_stream must be a callable"):
        DeepAgent(
            provider=StubProvider(),
            node=node,
            work_root=tmp_path,
            on_stream=cast(object, "invalid"),
        )


def test_rejects_non_path_work_root(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="root")

    with pytest.raises(TypeError, match="work_root must be a Path"):
        DeepAgent(
            provider=StubProvider(),
            node=node,
            work_root=cast(Path, str(tmp_path)),
        )


def test_rejects_missing_work_root(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="root")

    with pytest.raises(FileNotFoundError, match="work_root does not exist"):
        DeepAgent(
            provider=StubProvider(),
            node=node,
            work_root=tmp_path / "missing",
        )


def test_rejects_work_root_that_is_not_a_directory(tmp_path: Path) -> None:
    work_root = tmp_path / "file.txt"
    work_root.write_text("content")
    node = Node([tmp_path / "capabilities"], name_hds="root")

    with pytest.raises(ValueError, match="work_root must be a directory"):
        DeepAgent(provider=StubProvider(), node=node, work_root=work_root)


def test_rejects_work_root_without_required_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="root")
    checked_modes: list[int] = []

    def deny_access(path: Path, mode: int) -> bool:
        assert path == tmp_path
        checked_modes.append(mode)
        return False

    monkeypatch.setattr(os, "access", deny_access)

    with pytest.raises(
        PermissionError,
        match="work_root must be readable, writable, and traversable",
    ):
        DeepAgent(provider=StubProvider(), node=node, work_root=tmp_path)

    assert checked_modes == [os.R_OK | os.W_OK | os.X_OK]


def test_uses_root_hds_as_system_instructions(tmp_path: Path) -> None:
    provider = StubProvider()
    root = Node(
        [_write_capabilities(tmp_path, "root")],
        name_hds="root",
    )
    agent = DeepAgent(provider=provider, node=root, work_root=tmp_path)

    response = agent.run("hello")

    assert response.get_message() == "done"
    assert response.get_history_tree() == []
    assert provider.last_request is not None
    system_message = provider.last_request.get_messages()[0].message_dump()
    assert system_message["role"] == "system"
    assert system_message["content"].startswith("root instructions\n\n")
    assert "<harness-output-guidance>" in system_message["content"]
    tools = provider.last_request.get_tools()
    assert tools is not None
    assert tools.list_tools() == [
        "exit_agent",
        "get_execution_history",
        "search_saved_tool_outputs",
        "read_saved_tool_output",
    ]


def test_rejects_duplicate_agent_context_id_in_hierarchy(tmp_path: Path) -> None:
    capabilities = _write_capabilities(tmp_path, "shared_hds")
    child = Node([capabilities], name_hds="shared_hds")
    root = Node([capabilities], name_hds="shared_hds", children=[child])

    with pytest.raises(
        ValueError,
        match="Distinct nodes in hierarchy must have unique agent_context_id: shared_hds",
    ):
        DeepAgent(provider=StubProvider(), node=root, work_root=tmp_path)


def test_allows_distinct_agent_context_id_overrides_for_same_hds(
    tmp_path: Path,
) -> None:
    capabilities = _write_capabilities(tmp_path, "shared_hds")
    child = Node([capabilities], name_hds="shared_hds", agent_context_id="child_ctx")
    root = Node(
        [capabilities],
        name_hds="shared_hds",
        agent_context_id="root_ctx",
        children=[child],
    )

    agent = DeepAgent(provider=StubProvider(), node=root, work_root=tmp_path)
    assert agent.root_node is root
    assert root.agent_context_id == "root_ctx"
    assert child.agent_context_id == "child_ctx"
