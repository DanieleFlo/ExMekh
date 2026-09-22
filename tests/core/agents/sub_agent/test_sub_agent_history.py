from __future__ import annotations

from pathlib import Path

from qs.core.agents.deep_agent.deep_agent import DeepAgent
from qs.core.agents.node import Node
from qs.core.cache.cache_manager import CacheManager
from qs.core.cache.session_cache import SessionCache
from qs.core.models.assistant_message import AssistantMessage, FunctionCall, ToolCall
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.interfaces.provider import Provider
from qs.interfaces.tokenizer import Tokenizer


class MockTokenizer(Tokenizer):
    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = cache_dir or Path(".qs/tokenizers")

    def initialize(self) -> None:
        pass

    def generate_id(self) -> str:
        return "mock-id"

    def get_session(self) -> SessionCache:
        return CacheManager.get_session("mock-node", "default", self._cache_dir)


class SequenceProvider(Provider):
    def __init__(
        self,
        responses: list[AssistantMessage],
        cache_dir: Path | None = None,
    ) -> None:
        super().__init__(MockTokenizer(cache_dir))
        self._responses = list(responses)
        self.requests: list[ProviderRequest] = []

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("No more mocked responses available")
        return ProviderResponse(self._responses.pop(0))

    def call_structured(self, request):
        raise NotImplementedError


def _setup_capabilities(root: Path) -> Path:
    capabilities_path = root / "capabilities"
    hds_path = capabilities_path / "hds"
    exit_tool_path = capabilities_path / "exit-tool" / "finish"
    hds_path.mkdir(parents=True)
    exit_tool_path.mkdir(parents=True)

    (hds_path / "parent.md").write_text(
        "---\n"
        "description: Parent agent\n"
        "exit-tool: exit_agent\n"
        "agent-context-id: parent-ctx\n"
        "---\n"
        "Parent instructions"
    )
    (hds_path / "child.md").write_text(
        "---\n"
        "description: Child agent\n"
        "exit-tool: exit_agent\n"
        "agent-context-id: child-ctx\n"
        "---\n"
        "Child instructions"
    )

    (exit_tool_path / "instructions.md").write_text("Finish")
    (exit_tool_path / "main.py").write_text(
        "from qs.core.agents.exit_tool.data import ExitToolData\n"
        "from qs.core.agents.exit_tool.decorator import exit_tool\n\n"
        "@exit_tool(instructions_path='instructions.md', name='exit_agent', description='Exit')\n"
        "def exit_agent(text: str, items: list[str]) -> ExitToolData:\n"
        "    return ExitToolData(text=text, items=items)\n"
    )
    return capabilities_path


def _tool_call(call_id: str, sub_agent: str, text: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=FunctionCall(
            name="call_sub_agent",
            arguments={"sub_agent": sub_agent, "text": text},
        ),
    )


def test_sub_agent_with_persistent_history_retains_code_in_message_chain(
    tmp_path: Path,
) -> None:
    capabilities_path = _setup_capabilities(tmp_path)
    code = "SECRET-CODE-8821"

    provider = SequenceProvider(
        [
            # Turn 1: parent calls sub_agent with secret code
            AssistantMessage(
                None, tool_calls=[_tool_call("c1", "child", f"Store: {code}")]
            ),
            # Child responds in Turn 1
            AssistantMessage("CHILD: Stored"),
            # Parent finishes Turn 1
            AssistantMessage("PARENT: Done storing"),
            # Turn 2: parent calls sub_agent to retrieve code
            AssistantMessage(
                None, tool_calls=[_tool_call("c2", "child", "Retrieve code")]
            ),
            # Child responds in Turn 2
            AssistantMessage(f"CHILD: Found {code}"),
            # Parent finishes Turn 2
            AssistantMessage(f"PARENT: Received {code}"),
        ],
        cache_dir=tmp_path / "cache",
    )

    child_node = Node(
        [capabilities_path],
        name_hds="child",
        persistent_history=True,
    )
    parent_node = Node(
        [capabilities_path],
        name_hds="parent",
        children=[child_node],
        persistent_history=True,
    )
    agent = DeepAgent(provider=provider, node=parent_node, work_root=tmp_path)

    # Turn 1
    resp1 = agent.run("Please store this secret code in the child agent")
    assert resp1.get_message() == "PARENT: Done storing"

    # Turn 2
    resp2 = agent.run("Please ask the child agent for the stored code")
    assert resp2.get_message() == f"PARENT: Received {code}"

    # Verify the child's message history in the second call includes the first turn
    child_req2 = provider.requests[4]
    child_messages = [msg.get_message() for msg in child_req2.get_messages()]
    assert any(code in msg for msg in child_messages if isinstance(msg, str))


def test_sub_agent_without_persistent_history_forgets_code_across_runs(
    tmp_path: Path,
) -> None:
    capabilities_path = _setup_capabilities(tmp_path)
    code = "EPHEMERAL-CODE-3310"

    provider = SequenceProvider(
        [
            # Turn 1: parent calls sub_agent with code
            AssistantMessage(
                None, tool_calls=[_tool_call("c1", "child", f"Store: {code}")]
            ),
            # Child responds
            AssistantMessage("CHILD: Stored"),
            # Parent finishes Turn 1
            AssistantMessage("PARENT: Done storing"),
            # Turn 2: parent calls sub_agent to ask for code
            AssistantMessage(
                None, tool_calls=[_tool_call("c2", "child", "Recall code")]
            ),
            # Child responds without code
            AssistantMessage("CHILD: No code in memory"),
            # Parent finishes Turn 2
            AssistantMessage("PARENT: Child reports no code"),
        ],
        cache_dir=tmp_path / "cache",
    )

    child_node = Node(
        [capabilities_path],
        name_hds="child",
        persistent_history=False,
    )
    parent_node = Node(
        [capabilities_path],
        name_hds="parent",
        children=[child_node],
        persistent_history=True,
    )
    agent = DeepAgent(provider=provider, node=parent_node, work_root=tmp_path)

    # Turn 1
    resp1 = agent.run("Store code")
    assert resp1.get_message() == "PARENT: Done storing"

    # Turn 2
    resp2 = agent.run("Ask child for code")
    assert resp2.get_message() == "PARENT: Child reports no code"

    # Verify the child's message history in the second call does NOT contain the code
    child_req2 = provider.requests[4]
    child_messages = [msg.get_message() for msg in child_req2.get_messages()]
    assert not any(code in msg for msg in child_messages if isinstance(msg, str))
    # Child saw only system prompt and the recall message
    assert len(child_messages) == 2


def test_sub_agent_direct_persistent_history_retains_context(tmp_path: Path) -> None:
    capabilities_path = _setup_capabilities(tmp_path)
    token = "DIRECT-TOKEN-4455"

    provider = SequenceProvider(
        [
            AssistantMessage(f"ACK: Saved {token}"),
            AssistantMessage(f"Recalled: {token}"),
        ],
        cache_dir=tmp_path / "cache",
    )

    node = Node([capabilities_path], name_hds="child", persistent_history=True)
    node.initialize(provider, work_root=tmp_path)

    res1 = node.run(f"Store {token}")
    assert token in (res1.get_message() or "")

    res2 = node.run("What token was stored?")
    assert token in (res2.get_message() or "")

    req2_messages = [msg.get_message() for msg in provider.requests[1].get_messages()]
    assert any(token in msg for msg in req2_messages if isinstance(msg, str))
    assert len(req2_messages) == 4  # system, user1, assistant1, user2


def test_sub_agent_direct_non_persistent_history_resets_context(tmp_path: Path) -> None:
    capabilities_path = _setup_capabilities(tmp_path)
    token = "DIRECT-TOKEN-9911"

    provider = SequenceProvider(
        [
            AssistantMessage(f"ACK: Saved {token}"),
            AssistantMessage("No token known"),
        ],
        cache_dir=tmp_path / "cache",
    )

    node = Node([capabilities_path], name_hds="child", persistent_history=False)
    node.initialize(provider, work_root=tmp_path)

    res1 = node.run(f"Store {token}")
    assert token in (res1.get_message() or "")

    res2 = node.run("What token was stored?")
    assert res2.get_message() == "No token known"

    req2_messages = [msg.get_message() for msg in provider.requests[1].get_messages()]
    assert not any(token in msg for msg in req2_messages if isinstance(msg, str))
    assert len(req2_messages) == 2  # system, user2
