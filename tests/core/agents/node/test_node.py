from collections.abc import Sequence
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from xml.etree import ElementTree as ET

import pytest
from pydantic import BaseModel

from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.deep_agent.deep_agent import DeepAgent
from qs.core.agents.deep_agent.stream_event import (
    DeepAgentEvent,
    DeepAgentExecutionEvent,
    DeepAgentStreamEvent,
)
from qs.core.agents.deep_skill.deep_skill import CalledDeepSkill
from qs.core.agents.exit_tool.exit_tool import ExitTool
from qs.core.agents.hds.hds import HDS
from qs.core.agents.node import DefaultCapabilities, Node
from qs.core.agents.node.agentic_scope import AgenticScope
from qs.core.agents.node.node_runtime import NodeRuntime
from qs.core.agents.node.node_runtime_config import NodeRuntimeConfig
from qs.core.agents.node.tool_call_executor import ToolCallExecutor
from qs.core.agents.node.tool_call_resolver import ToolCallResolver
from qs.core.agents.sub_agent import SubAgentTool
from qs.core.cache.cache_manager import CacheManager
from qs.core.cache.node_cache import NodeCache
from qs.core.cache.session_cache import SessionCache
from qs.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from qs.core.models.client import Client
from qs.core.models.model_thinking import ModelThinking
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.tool import CalledTool, PreformattedToolResult, Tool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import ErrorUserMessage, UserMessage
from qs.core.models.xml_header import XmlHeader
from qs.core.tokenizers.sequential_tokenizer import SequentialTokenizer
from qs.core.tool_calls.execution_manager import ToolExecutionManager
from qs.core.tool_calls.memory_tools import ExecutionHistory
from qs.interfaces.provider import Provider
from qs.interfaces.tokenizer import Tokenizer
from tests.event_handler import as_agent_handler

_RUNTIME_TOOL_NAMES = [
    "get_execution_history",
    "search_tool_outputs",
    "read_tool_output",
    "control_tool_execution",
]


class _StubTokenizer(Tokenizer):
    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

    def initialize(self) -> None:
        pass

    def generate_id(self) -> str:
        raise RuntimeError("No tokenizer session for stub")

    def get_session(self) -> SessionCache:
        return CacheManager.get_session("stub-node", "default", Path(".qs/tokenizers"))


class SequenceProvider(Provider):
    def __init__(
        self,
        responses: list[AssistantMessage],
        tokenizer: Tokenizer | None = None,
    ) -> None:
        super().__init__(tokenizer or _StubTokenizer())
        self._responses = responses
        self.requests: list[ProviderRequest] = []

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if request.get_streaming():
            request.execute_on_stream_event(
                ProviderStreamEvent(ProviderStreamEventType.START)
            )
            request.execute_on_stream_event(
                ProviderStreamEvent(
                    ProviderStreamEventType.RESPONSE,
                    content=f"chunk-{len(self.requests)}",
                )
            )
            request.execute_on_stream_event(
                ProviderStreamEvent(ProviderStreamEventType.END)
            )
        return ProviderResponse(self._responses.pop(0))

    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        del request
        raise NotImplementedError


def _write_capabilities(
    parent: Path,
    hds_names: tuple[str, ...],
) -> Path:
    capabilities_path = parent / "capabilities"
    hds_path = capabilities_path / "hds"
    exit_tool_path = capabilities_path / "exit-tool" / "finish"
    hds_path.mkdir(parents=True)
    exit_tool_path.mkdir(parents=True)

    for name in hds_names:
        (hds_path / f"{name}.md").write_text(
            f"---\ndescription: {name}\nexit-tool: exit_agent\n---\n{name} body"
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


def _write_runnable_capabilities(parent: Path) -> Path:
    capabilities_path = _write_capabilities(parent, ("main",))
    hds_path = capabilities_path / "hds" / "main.md"
    hds_path.write_text(
        "---\n"
        "description: Main instructions\n"
        "exit-tool: exit_agent\n"
        "sub-skills:\n"
        "  - mock-skill\n"
        "sub-tools:\n"
        "  - mock-tool\n"
        "---\n"
        "Main body"
    )

    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir()
    (deep_skill_path / "mock-skill.md").write_text(
        "---\ndescription: Mock skill\n---\nMock skill body"
    )

    tool_path = capabilities_path / "tools" / "mock-tool"
    tool_path.mkdir(parents=True)
    (tool_path / "main.py").write_text(
        "from qs.core.agents.tools.decorator import deep_tool\n\n"
        "@deep_tool(name='mock-tool', description='Mock tool')\n"
        "def run(work_root, text: str) -> dict[str, str]:\n"
        "    del work_root\n"
        "    return {'echo': text}\n"
    )
    return capabilities_path


def _write_nested_capabilities(parent: Path) -> Path:
    capabilities_path = _write_runnable_capabilities(parent)
    deep_skill_path = capabilities_path / "deep-skill"
    (deep_skill_path / "mock-skill.md").write_text(
        "---\n"
        "description: Mock skill\n"
        "sub-skills:\n"
        "  - nested-skill\n"
        "sub-tools:\n"
        "  - mock-sub-tool\n"
        "  - shared-tool\n"
        "---\n"
        "Mock skill body"
    )
    (deep_skill_path / "nested-skill.md").write_text(
        "---\n"
        "description: Nested skill\n"
        "inherits-parent-tools: true\n"
        "sub-tools:\n"
        "  - nested-sub-tool\n"
        "  - shared-tool\n"
        "---\n"
        "Nested skill body"
    )

    for name, description in (
        ("mock-sub-tool", "Mock sub-tool"),
        ("nested-sub-tool", "Nested sub-tool"),
        ("shared-tool", "Shared tool"),
        ("orphan-tool", "Orphan tool"),
    ):
        tool_path = capabilities_path / "tools" / name
        tool_path.mkdir()
        (tool_path / "main.py").write_text(
            "from qs.core.agents.tools.decorator import deep_tool\n\n"
            f"@deep_tool(name='{name}', description='{description}')\n"
            "def run(work_root, text: str) -> dict[str, str]:\n"
            "    del work_root\n"
            "    return {'echo': text}\n"
        )
    return capabilities_path


def _function_call(
    name: str,
    arguments: dict[str, object],
    *,
    call_id: str,
) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=FunctionCall(name=name, arguments=arguments),
    )


def test_loads_capabilities_and_accepts_an_existing_hds(tmp_path: Path) -> None:
    capabilities_path = _write_capabilities(tmp_path, ("main",))
    node = Node([capabilities_path], name_hds="main")

    capabilities = node.load_capabilities(work_root=tmp_path)

    assert node.name_hds == "main"
    assert node.children == ()
    assert node.persistent_history is True
    assert tuple(hds.name for hds in capabilities.hds) == ("main",)


def test_accepts_hds_from_any_configured_capabilities_path(
    tmp_path: Path,
) -> None:
    first_path = _write_capabilities(tmp_path / "first", ("first",))
    second_path = _write_capabilities(tmp_path / "second", ("second",))
    for path, name in [(first_path, "first"), (second_path, "second")]:
        entrypoint = path / "exit-tool" / "finish" / "main.py"
        entrypoint.write_text(
            entrypoint.read_text().replace("name='exit_agent'", f"name='exit_{name}'")
        )
        hds = path / "hds" / f"{name}.md"
        hds.write_text(
            f"---\ndescription: {name}\nexit-tool: exit_{name}\n---\n{name} body"
        )
    capabilities = Node(
        [first_path, second_path],
        name_hds="second",
    ).load_capabilities(work_root=tmp_path)

    assert tuple(hds.name for hds in capabilities.hds) == ("first", "second")


def test_rejects_unknown_hds_and_lists_available_names(tmp_path: Path) -> None:
    capabilities_path = _write_capabilities(tmp_path, ("first", "second"))
    node = Node([capabilities_path], name_hds="missing")

    with pytest.raises(
        ValueError,
        match="HDS 'missing' not found. Available HDS: first, second",
    ):
        node.load_capabilities(work_root=tmp_path)


def test_rejects_hds_name_when_no_hds_is_available(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.mkdir()
    node = Node([capabilities_path], name_hds="missing")

    with pytest.raises(
        ValueError,
        match="HDS 'missing' not found. Available HDS: none",
    ):
        node.load_capabilities(work_root=tmp_path)


@pytest.mark.parametrize("name_hds", ["", "   "])
def test_rejects_empty_hds_name(tmp_path: Path, name_hds: str) -> None:
    with pytest.raises(ValueError, match="name_hds cannot be empty"):
        Node([tmp_path / "capabilities"], name_hds=name_hds)


def test_rejects_non_string_hds_name(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="name_hds must be a string"):
        Node([tmp_path / "capabilities"], name_hds=cast(str, 1))


def test_normalizes_children_to_an_immutable_tuple(tmp_path: Path) -> None:
    child = Node([tmp_path / "child" / "capabilities"], name_hds="child")

    node = Node(
        [tmp_path / "root" / "capabilities"],
        name_hds="root",
        children=[child],
        persistent_history=False,
    )

    assert node.children == (child,)
    assert node.persistent_history is False


def test_rejects_invalid_children_collection(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="children must be a sequence of Node"):
        Node(
            [tmp_path / "capabilities"],
            name_hds="main",
            children=cast(Sequence[Node], "child"),
        )


def test_rejects_non_node_child(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="children must contain only Node"):
        Node(
            [tmp_path / "capabilities"],
            name_hds="main",
            children=[cast(Node, object())],
        )


def test_rejects_non_boolean_persistent_history(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="persistent_history must be a bool"):
        Node(
            [tmp_path / "capabilities"],
            name_hds="main",
            persistent_history=cast(bool, 1),
        )


def test_depth_defaults_to_zero(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="main")

    assert node.depth == 0


def test_accepts_non_negative_initial_depth(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="main", depth=2)

    assert node.depth == 2


@pytest.mark.parametrize("depth", [True, 1.5, "1"])
def test_rejects_non_integer_depth(tmp_path: Path, depth: object) -> None:
    with pytest.raises(TypeError, match="depth must be an integer"):
        Node(
            [tmp_path / "capabilities"],
            name_hds="main",
            depth=cast(int, depth),
        )


def test_rejects_negative_depth(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="depth cannot be negative"):
        Node([tmp_path / "capabilities"], name_hds="main", depth=-1)


def test_initialize_rejects_non_provider(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="main")

    with pytest.raises(TypeError, match="provider must be a Provider"):
        node.initialize(cast(Provider, object()), work_root=tmp_path)


def test_initialize_rejects_non_callable_stream_handler(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="main")

    with pytest.raises(TypeError, match="DeepAgentEventHandler"):
        node.initialize(
            SequenceProvider([]),
            work_root=tmp_path,
            on_stream=cast(object, "invalid"),
        )


def test_runtime_config_ignores_stream_events_without_a_handler() -> None:
    config = NodeRuntimeConfig(
        provider=cast(Provider, object()),
        capabilities=cast(CapabilitiesData, object()),
        hds=cast(HDS, object()),
        exit_tool=cast(ExitTool, object()),
        persistent_history=True,
        initial_depth=0,
    )
    event = DeepAgentStreamEvent(
        ProviderStreamEvent(ProviderStreamEventType.START),
        depth=0,
        skill_path="/",
    )

    assert config.is_streaming() is False
    assert config.emit_stream(event) is None


def test_runtime_config_resolves_agent_context_id() -> None:
    config_explicit = NodeRuntimeConfig(
        provider=cast(Provider, object()),
        capabilities=cast(CapabilitiesData, object()),
        hds=cast(HDS, object()),
        exit_tool=cast(ExitTool, object()),
        persistent_history=True,
        initial_depth=0,
        agent_context_id="custom-ctx",
    )
    assert config_explicit.resolved_agent_context_id == "custom-ctx"

    hds_with_context = MagicMock()
    hds_with_context.get_agent_context_id.return_value = "hds-ctx"
    config_hds = NodeRuntimeConfig(
        provider=cast(Provider, object()),
        capabilities=cast(CapabilitiesData, object()),
        hds=cast(HDS, hds_with_context),
        exit_tool=cast(ExitTool, object()),
        persistent_history=True,
        initial_depth=0,
    )
    assert config_hds.resolved_agent_context_id == "hds-ctx"

    hds_with_name = MagicMock(spec=["get_name"])
    hds_with_name.get_name.return_value = "hds-name"
    config_name = NodeRuntimeConfig(
        provider=cast(Provider, object()),
        capabilities=cast(CapabilitiesData, object()),
        hds=cast(HDS, hds_with_name),
        exit_tool=cast(ExitTool, object()),
        persistent_history=True,
        initial_depth=0,
    )
    assert config_name.resolved_agent_context_id == "hds-name"

    hds_with_empty_values = MagicMock()
    hds_with_empty_values.get_agent_context_id.return_value = ""
    hds_with_empty_values.get_name.return_value = "   "
    config_empty = NodeRuntimeConfig(
        provider=cast(Provider, object()),
        capabilities=cast(CapabilitiesData, object()),
        hds=cast(HDS, hds_with_empty_values),
        exit_tool=cast(ExitTool, object()),
        persistent_history=True,
        initial_depth=0,
    )
    assert config_empty.resolved_agent_context_id == "default"

    config_default = NodeRuntimeConfig(
        provider=cast(Provider, object()),
        capabilities=cast(CapabilitiesData, object()),
        hds=cast(HDS, object()),
        exit_tool=cast(ExitTool, object()),
        persistent_history=True,
        initial_depth=0,
    )
    assert config_default.resolved_agent_context_id == "default"


def test_runtime_rejects_duplicate_runtime_tool_names(tmp_path: Path) -> None:
    capabilities = Node(
        [_write_capabilities(tmp_path, ("main",))],
        name_hds="main",
    ).load_capabilities(work_root=tmp_path)
    duplicate_tools = (
        Tool(lambda: "first", name="duplicate"),
        Tool(lambda: "second", name="duplicate"),
    )
    config = NodeRuntimeConfig(
        provider=SequenceProvider([]),
        capabilities=capabilities,
        hds=capabilities.get_hds("main"),
        exit_tool=capabilities.get_exit_tool("exit_agent"),
        persistent_history=True,
        initial_depth=0,
        runtime_tools=duplicate_tools,
    )

    with pytest.raises(
        ValueError,
        match="Harness runtime tool names must be unique: duplicate",
    ):
        NodeRuntime(config)


def test_run_requires_an_initialized_node(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="main")

    with pytest.raises(RuntimeError, match="must be initialized before run"):
        node.run("hello")


def test_restore_history_requires_an_initialized_node(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="main")

    with pytest.raises(RuntimeError, match="before restoring history"):
        node.restore_history([UserMessage("earlier")])


def test_restore_history_uses_public_node_api(tmp_path: Path) -> None:
    provider = SequenceProvider([AssistantMessage("new answer")])
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)
    node.restore_history([UserMessage("earlier"), AssistantMessage("earlier answer")])
    restored_state: list[tuple[ExecutionHistory, NodeCache]] = []
    node.restore_history(
        [UserMessage("earlier"), AssistantMessage("earlier answer")],
        lambda history, cache: restored_state.append((history, cache)),
    )

    node.run("next")

    assert len(restored_state) == 1
    assert isinstance(restored_state[0][0], ExecutionHistory)
    assert isinstance(restored_state[0][1], NodeCache)
    assert [message.get_message() for message in provider.requests[0].get_messages()][
        1:
    ] == ["earlier", "earlier answer", "next"]


def test_restore_history_rejects_non_persistent_node(tmp_path: Path) -> None:
    node = Node(
        [_write_runnable_capabilities(tmp_path)],
        name_hds="main",
        persistent_history=False,
    )
    node.initialize(SequenceProvider([]), work_root=tmp_path)

    with pytest.raises(ValueError, match="non-persistent"):
        node.restore_history([UserMessage("earlier")])


def test_sub_agent_metadata_requires_initialized_nodes(tmp_path: Path) -> None:
    child = Node([tmp_path / "child" / "capabilities"], name_hds="child")
    parent = Node(
        [tmp_path / "parent" / "capabilities"],
        name_hds="parent",
        children=[child],
    )

    with pytest.raises(RuntimeError, match="before accessing its HDS"):
        child.get_hds()
    with pytest.raises(RuntimeError, match="before updating sub-agents"):
        parent.update_sub_agent_tool_description()


def test_tool_descriptions_reflect_hds_and_skill_availability(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider([AssistantMessage("final answer")])
    node = Node([_write_nested_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    node.run("start")

    request_tools = provider.requests[0].get_tools()
    assert request_tools is not None
    descriptions = {tool.name: tool.description for tool in request_tools.get_tools()}
    assert descriptions == {
        "mock-skill": "Mock skill",
        "nested-skill": "Available in skill 'mock-skill'.",
        "mock-tool": "Mock tool",
        "mock-sub-tool": "Available in skill 'mock-skill'.",
        "nested-sub-tool": "Available in skill 'nested-skill'.",
        "shared-tool": "Available in skills 'mock-skill', 'nested-skill'.",
        "orphan-tool": "",
        "exit_agent": "Exit agent Available in every skill.",
        "get_execution_history": (
            "Return navigable XML history of capability calls. Omit "
            "start_position for a head-and-tail overview, or use a 1-based "
            "position and limit from 2 to 100 to read a page."
        ),
        "search_tool_outputs": (
            "Search this node's saved and runtime tool outputs, including "
            "background work. Return paginated XML previews with status "
            "and output IDs. All filters are optional."
        ),
        "read_tool_output": (
            "Read the current status and available output of a tool by output ID. "
            "Use an optional cursor to read only new output; omit it to read the "
            "complete output. Includes the final result when available."
        ),
        "control_tool_execution": (
            "Wait for a tool execution by its original tool_call_id, or request "
            "cancellation. Waiting returns as soon as the work finishes or "
            "the wait limit expires; read its output with read_tool_output."
        ),
    }


@pytest.mark.parametrize("invalid_content", [None, "", "   "])
def test_empty_provider_response_adds_error_message_and_retries(
    tmp_path: Path,
    invalid_content: str | None,
) -> None:
    provider = SequenceProvider(
        [AssistantMessage(invalid_content), AssistantMessage("recovered answer")]
    )
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    response = node.run("start")

    assert response.get_message() == "recovered answer"
    assert len(provider.requests) == 2
    error_message = provider.requests[1].get_messages()[-1]
    assert isinstance(error_message, ErrorUserMessage)
    assert error_message.get_message() == Node.EMPTY_RESPONSE_MESSAGE


def test_empty_tool_call_list_is_treated_as_no_tool_calls(tmp_path: Path) -> None:
    provider = SequenceProvider([AssistantMessage("Hello", tool_calls=[])])
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    response = node.run("ciao")

    assert response.get_message() == "Hello"
    assert len(provider.requests) == 1


def test_tool_call_resolver_accepts_no_calls() -> None:
    resolver = ToolCallResolver(cast(CapabilitiesData, MagicMock()))

    assert resolver.resolve_all(None) == []
    assert resolver.resolve_all([]) == []


def test_identify_tool_choice_rejects_unknown_and_unsupported_tools() -> None:
    capabilities = MagicMock()
    resolver = ToolCallResolver(cast(CapabilitiesData, capabilities))
    tool_call = _function_call("missing", {}, call_id="call")
    capabilities.get_all_tools.return_value = ()

    with pytest.raises(ValueError, match="is not a valid deep skill"):
        resolver.resolve(tool_call)

    capabilities.get_all_tools.return_value = (SimpleNamespace(name="missing"),)
    with pytest.raises(TypeError, match="not a valid deep skill"):
        resolver.resolve(tool_call)


def test_executor_rejects_unsupported_mapped_capability() -> None:
    executor = ToolCallExecutor(
        client=MagicMock(spec=Client),
        exit_tool=cast(ExitTool, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
        exit_content_resolver=MagicMock(),
    )

    with pytest.raises(TypeError, match="Node supports only"):
        executor.execute(cast(CalledDeepSkill, object()))


def test_run_tool_delegates_execution_to_called_tool() -> None:
    client = MagicMock(spec=Client)
    called_tool = MagicMock(spec=CalledTool)
    called_tool.execute.return_value = {"echo": "hello"}
    called_tool.call_id = "call-123"
    called_tool.name = "echo"
    executor = ToolCallExecutor(
        client=client,
        exit_tool=cast(ExitTool, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
        exit_content_resolver=MagicMock(),
    )

    result = executor.execute(called_tool)

    called_tool.execute.assert_called_once_with(work_root=Path.cwd(), history=())
    client.add_tool_response.assert_not_called()
    client.get_session.assert_not_called()
    response = result.response
    assert ET.fromstring(response.get_message()).findtext("echo") == "hello"
    assert response.get_tool_call_id() == "call-123"
    assert response.get_name() == "echo"


def test_run_tool_prepends_xml_header_when_client_generates_id() -> None:
    client = MagicMock(spec=Client)
    client.generate_id.return_value = "id-tool"
    called_tool = MagicMock(spec=CalledTool)
    called_tool.execute.return_value = {"echo": "hello"}
    called_tool.call_id = "call-123"
    called_tool.name = "echo"
    executor = ToolCallExecutor(
        client=client,
        exit_tool=cast(ExitTool, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
        exit_content_resolver=MagicMock(),
    )

    result = executor.execute(called_tool)

    called_tool.execute.assert_called_once_with(work_root=Path.cwd(), history=())
    client.add_tool_response.assert_not_called()
    client.get_session.assert_not_called()
    response = result.response
    expected_header = XmlHeader(source="Tool", name="echo", id="id-tool").to_xml()
    assert response.get_message().startswith(expected_header + "\n")
    body = response.get_message().split("</header>\n", 1)[1]
    assert ET.fromstring(body).findtext("echo") == "hello"
    assert response.get_tool_call_id() == "call-123"
    assert response.get_name() == "echo"


def test_tool_failure_still_emits_the_end_state() -> None:
    called_tool = MagicMock(spec=CalledTool)
    called_tool.execute.side_effect = RuntimeError("tool failed")
    called_tool.call_id = "call-123"
    called_tool.name = "echo"
    client = MagicMock(spec=Client)
    executor = ToolCallExecutor(
        client=client,
        exit_tool=cast(ExitTool, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
        exit_content_resolver=MagicMock(),
    )
    runtime = object.__new__(NodeRuntime)
    runtime._execution_manager = ToolExecutionManager()
    runtime._client = client
    runtime._executor = executor
    runtime._emit_tool_state = MagicMock()

    with pytest.raises(RuntimeError, match="tool failed"):
        runtime._execute_tool_call(called_tool)

    assert runtime._emit_tool_state.call_args_list == [
        (("echo", "call-123"), {"running": True}),
        (("echo", "call-123"), {"running": False}),
    ]
    client.add_tool_response.assert_not_called()
    client.get_session.assert_not_called()


def test_skill_increments_depth_adds_exit_body_and_requires_a_tool(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "mock-skill",
                        {"topic": "XML & cache"},
                        call_id="skill-call",
                    )
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": []},
                        call_id="exit-call",
                    )
                ],
            ),
            AssistantMessage("final answer"),
        ]
    )
    node = Node(
        [_write_runnable_capabilities(tmp_path)],
        name_hds="main",
    )
    node.initialize(provider, work_root=tmp_path)

    response = node.run("start")

    assert response.get_message() == "final answer"
    assert node.depth == 0
    assert [request.get_tool_choice() for request in provider.requests] == [
        None,
        "required",
        None,
    ]
    skill_message = cast(ToolMessage, provider.requests[1].get_messages()[-1])
    assert skill_message.get_tool_call_id() == "skill-call"
    assert skill_message.get_message() == (
        "Mock skill body\n\n<catalog></catalog>\n\n"
        '<exit-tool name="exit_agent">Finish the agent</exit-tool>'
    )
    exit_message = cast(ToolMessage, provider.requests[2].get_messages()[-1])
    assert exit_message.get_message() == "done"


def test_skill_increments_depth_prepends_xml_header_on_exit(tmp_path: Path) -> None:
    class ProviderWithId(SequenceProvider):
        def generate_id(self) -> str:
            return "exit-id-456"

    provider = ProviderWithId(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "mock-skill",
                        {"topic": "XML & cache"},
                        call_id="skill-call",
                    )
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": []},
                        call_id="exit-call",
                    )
                ],
            ),
            AssistantMessage("final answer"),
        ]
    )
    node = Node(
        [_write_runnable_capabilities(tmp_path)],
        name_hds="main",
    )
    node.initialize(provider, work_root=tmp_path)

    response = node.run("start")

    assert response.get_message() == "final answer"
    assert node.depth == 0
    exit_message = cast(ToolMessage, provider.requests[2].get_messages()[-1])
    expected_header = XmlHeader(
        source="DeepSkill", name="mock-skill", id="exit-id-456"
    ).to_xml()
    assert exit_message.get_message() == f"{expected_header}\ndone"
    record = node._runtime._tool_cache.get("exit-id-456")
    assert record.name == "mock-skill"
    assert record.source == "DeepSkill"
    assert record.tool_call_id == "skill-call"
    assert record.args == {"topic": "XML & cache"}
    assert record.skill_path == "/mock-skill"
    assert record.result == {"text": "done", "items": []}


def test_exit_items_retry_keeps_skill_open_then_compacts_selected_output(
    tmp_path: Path,
) -> None:
    class ProviderWithId(SequenceProvider):
        def __init__(self, responses: list[AssistantMessage]) -> None:
            super().__init__(responses)
            self._next_id = 0

        def generate_id(self) -> str:
            self._next_id += 1
            return f"out-{self._next_id}"

    provider = ProviderWithId(
        [
            AssistantMessage(
                None, tool_calls=[_function_call("mock-skill", {}, call_id="open")]
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call("mock-sub-tool", {"text": "A < B"}, call_id="work")
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "wrong", "items": ["missing"]},
                        call_id="bad-exit",
                    )
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "Result: {{out-1}}", "items": ["work"]},
                        call_id="good-exit",
                    )
                ],
            ),
            AssistantMessage("final"),
        ]
    )
    node = Node([_write_nested_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    response = node.run("start")

    assert node.depth == 0
    assert response.get_message() == "final"
    assert provider.requests[3].get_tool_choice() == "required"
    retry_history = provider.requests[3].get_messages()
    assert any(
        isinstance(message, ToolMessage)
        and message.get_tool_call_id() == "bad-exit"
        and "Correct the error and call exit_agent again" in message.get_message()
        for message in retry_history
    )
    assert any(
        isinstance(message, ToolMessage) and message.get_tool_call_id() == "work"
        for message in retry_history
    )
    skill_result = response.skill_executions[0].output
    assert skill_result is not None
    assert skill_result.count('<tool-output id="out-1"') == 1
    embedded_output = ET.fromstring(skill_result.split("Result: ", 1)[1])
    assert (
        ET.fromstring(embedded_output.findtext("content")).findtext("echo") == "A < B"
    )
    assert any(
        isinstance(message, ToolMessage)
        and message.get_tool_call_id() == "open"
        and message.get_message() == skill_result
        for message in provider.requests[-1].get_messages()
    )
    records = node._runtime._tool_cache.get_all()
    assert {(record.id, record.tool_call_id) for record in records} == {
        ("out-1", "work"),
        ("out-2", "open"),
    }
    saved_result = next(record for record in records if record.id == "out-2").result
    assert saved_result == {
        "text": skill_result.split("</header>\n", 1)[1],
        "items": ["work"],
    }


def test_malformed_exit_items_return_feedback_without_closing_skill(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                None, tool_calls=[_function_call("mock-skill", {}, call_id="open")]
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": [1]},
                        call_id="invalid",
                    )
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": []},
                        call_id="valid",
                    )
                ],
            ),
            AssistantMessage("final"),
        ]
    )
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    response = node.run("start")

    assert node.depth == 0
    assert response.get_message() == "final"
    assert provider.requests[2].get_tool_choice() == "required"
    assert "Invalid arguments" in provider.requests[2].get_messages()[-1].get_message()
    assert "items=[]" in provider.requests[2].get_messages()[-1].get_message()
    assert all(
        record.tool_call_id != "invalid"
        for record in node._runtime._tool_cache.get_all()
    )


def test_agentic_scope_tracks_nested_skills_and_returns(tmp_path: Path) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-skill", {}, call_id="parent-skill-call")
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("nested-skill", {}, call_id="nested-skill-call")
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "nested done", "items": []},
                        call_id="nested-exit-call",
                    )
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "parent done", "items": []},
                        call_id="parent-exit-call",
                    )
                ],
            ),
            AssistantMessage("final answer"),
        ]
    )
    node = Node([_write_nested_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    node.run("start")

    assert [request.get_extra() for request in provider.requests] == [
        {
            "agentic": {
                "agent_context_id": "main",
                "allowed_tools": ["mock-tool", *_RUNTIME_TOOL_NAMES],
                "allowed_skills": ["mock-skill"],
            }
        },
        {
            "agentic": {
                "agent_context_id": "main",
                "allowed_tools": [
                    "mock-sub-tool",
                    "shared-tool",
                    *_RUNTIME_TOOL_NAMES,
                    "exit_agent",
                ],
                "allowed_skills": ["nested-skill"],
            }
        },
        {
            "agentic": {
                "agent_context_id": "main",
                "allowed_tools": [
                    "nested-sub-tool",
                    "shared-tool",
                    "mock-sub-tool",
                    *_RUNTIME_TOOL_NAMES,
                    "exit_agent",
                ],
                "allowed_skills": [],
            }
        },
        {
            "agentic": {
                "agent_context_id": "main",
                "allowed_tools": [
                    "mock-sub-tool",
                    "shared-tool",
                    *_RUNTIME_TOOL_NAMES,
                    "exit_agent",
                ],
                "allowed_skills": ["nested-skill"],
                "operation": "return",
                "skill_call_id": "nested-skill-call",
            }
        },
        {
            "agentic": {
                "agent_context_id": "main",
                "allowed_tools": ["mock-tool", *_RUNTIME_TOOL_NAMES],
                "allowed_skills": ["mock-skill"],
                "operation": "return",
                "skill_call_id": "parent-skill-call",
            }
        },
    ]


def test_streams_context_across_nested_skills_and_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot_calls = 0
    original_snapshot = ToolCallExecutor.snapshot

    def count_snapshot(executor: ToolCallExecutor):
        nonlocal snapshot_calls
        snapshot_calls += 1
        return original_snapshot(executor)

    monkeypatch.setattr(ToolCallExecutor, "snapshot", count_snapshot)
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-skill", {}, call_id="parent-skill-call")
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("nested-skill", {}, call_id="nested-skill-call")
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "nested-sub-tool",
                        {"text": "work"},
                        call_id="tool-call",
                    )
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "nested done", "items": []},
                        call_id="nested-exit-call",
                    )
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "parent done", "items": []},
                        call_id="parent-exit-call",
                    )
                ],
            ),
            AssistantMessage("final answer"),
        ]
    )
    events: list[DeepAgentEvent] = []
    agent = DeepAgent(
        provider=provider,
        node=Node([_write_nested_capabilities(tmp_path)], name_hds="main"),
        work_root=tmp_path,
        on_stream=as_agent_handler(events.append),
    )

    response = agent.run("start")

    assert response.get_message() == "final answer"
    assert all(request.get_streaming() for request in provider.requests)
    chunks = [
        (event.provider_event.content, event.depth, event.skill_path)
        for event in events
        if isinstance(event, DeepAgentStreamEvent)
        and event.provider_event.type is ProviderStreamEventType.RESPONSE
    ]
    assert chunks == [
        ("chunk-1", 0, "/"),
        ("chunk-2", 1, "/mock-skill"),
        ("chunk-3", 2, "/mock-skill/nested-skill"),
        ("chunk-4", 2, "/mock-skill/nested-skill"),
        ("chunk-5", 1, "/mock-skill"),
        ("chunk-6", 0, "/"),
    ]

    tool_events = [
        event for event in events if isinstance(event, DeepAgentExecutionEvent)
    ]
    assert snapshot_calls == len(tool_events)
    assert [
        (event.tool_name, event.is_tool_running, event.depth, event.skill_path)
        for event in tool_events
    ] == [
        ("mock-skill", True, 0, "/"),
        ("mock-skill", False, 1, "/mock-skill"),
        ("nested-skill", True, 1, "/mock-skill"),
        ("nested-skill", False, 2, "/mock-skill/nested-skill"),
        ("nested-sub-tool", True, 2, "/mock-skill/nested-skill"),
        ("nested-sub-tool", False, 2, "/mock-skill/nested-skill"),
        ("exit_agent", True, 2, "/mock-skill/nested-skill"),
        ("exit_agent", False, 1, "/mock-skill"),
        ("exit_agent", True, 1, "/mock-skill"),
        ("exit_agent", False, 0, "/"),
    ]

    nested_tool_end = tool_events[5]
    assert nested_tool_end.history[-1]["role"] == "tool"
    body = nested_tool_end.history[-1]["content"].split("</header>\n", 1)[1]
    assert ET.fromstring(body).findtext("echo") == "work"
    parent_tree = nested_tool_end.history_tree[0]
    assert parent_tree["skill"] == "mock-skill"
    assert parent_tree["output"] is None
    assert parent_tree["children"][0]["skill"] == "nested-skill"
    assert parent_tree["children"][0]["output"] is None

    parent_exit_end = tool_events[-1]
    assert parent_exit_end.depth == 0
    assert parent_exit_end.skill_path == "/"
    assert parent_exit_end.history_tree[0]["output"] == "parent done"


def test_exit_tool_is_rejected_at_root(tmp_path: Path) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": []},
                        call_id="exit-call",
                    )
                ],
            ),
            AssistantMessage("final answer"),
        ]
    )
    node = Node(
        [_write_runnable_capabilities(tmp_path)],
        name_hds="main",
    )
    node.initialize(provider, work_root=tmp_path)

    with pytest.raises(ValueError, match="'exit_agent' is not allowed at /"):
        node.run("start")
    assert node.depth == 0
    assert len(provider.requests) == 1


def test_removed_exit_skill_cannot_be_called(tmp_path: Path) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                "", tool_calls=[_function_call("finish", {}, call_id="instructions")]
            ),
        ]
    )
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main", depth=1)
    node.initialize(provider, work_root=tmp_path)
    with pytest.raises(ValueError, match="'finish' is not allowed"):
        node.run("start")
    assert node.depth == 1


@pytest.mark.parametrize("result", [None, {}, {"text": 1}, []])
def test_rejects_invalid_exit_tool_result(result: object) -> None:
    with pytest.raises(TypeError, match="must contain a text string"):
        ToolCallExecutor.get_exit_output(result)


def test_stringify_result_preserves_strings() -> None:
    assert ToolCallExecutor.stringify_result("plain result") == "plain result"


def test_nested_skills_require_tools_until_every_level_has_exited(
    tmp_path: Path,
) -> None:
    skill_call = AssistantMessage(
        "",
        tool_calls=[_function_call("mock-skill", {}, call_id="skill-call")],
    )
    exit_call = AssistantMessage(
        "",
        tool_calls=[
            _function_call(
                "exit_agent",
                {"text": "done", "items": []},
                call_id="exit-call",
            )
        ],
    )
    provider = SequenceProvider(
        [
            skill_call,
            AssistantMessage(
                "",
                tool_calls=[_function_call("nested-skill", {}, call_id="nested-call")],
            ),
            exit_call,
            exit_call,
            AssistantMessage("final"),
        ]
    )
    node = Node(
        [_write_nested_capabilities(tmp_path)],
        name_hds="main",
    )
    node.initialize(provider, work_root=tmp_path)

    node.run("start")

    assert node.depth == 0
    assert [request.get_tool_choice() for request in provider.requests] == [
        None,
        "required",
        "required",
        "required",
        None,
    ]


def test_regular_tool_does_not_change_depth_and_serializes_its_result(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "mock-tool",
                        {"text": "hello"},
                        call_id="tool-call",
                    )
                ],
            ),
            AssistantMessage("final answer"),
        ]
    )
    node = Node(
        [_write_runnable_capabilities(tmp_path)],
        name_hds="main",
    )
    node.initialize(provider, work_root=tmp_path)

    node.run("start")

    assert node.depth == 0
    tool_message = cast(ToolMessage, provider.requests[1].get_messages()[-1])
    body = tool_message.get_message().split("</header>\n", 1)[1]
    assert ET.fromstring(body).findtext("echo") == "hello"
    assert "<response-type>final</response-type>" in tool_message.get_message()


def test_positive_initial_depth_requires_a_tool_on_first_request(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider([AssistantMessage("final answer")])
    node = Node(
        [_write_runnable_capabilities(tmp_path)],
        name_hds="main",
        depth=1,
    )
    node.initialize(provider, work_root=tmp_path)

    node.run("start")

    assert provider.requests[0].get_tool_choice() == "required"
    assert node.depth == 1


def test_non_persistent_node_starts_each_run_with_fresh_history(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider([AssistantMessage("first"), AssistantMessage("second")])
    node = Node(
        [_write_runnable_capabilities(tmp_path)],
        name_hds="main",
        persistent_history=False,
    )
    node.initialize(provider, work_root=tmp_path)

    node.run("first request")
    node.run("second request")

    assert [
        tuple(message.get_role_name() for message in request.get_messages())
        for request in provider.requests
    ] == [("system", "user"), ("system", "user")]


def test_executor_returns_raw_result_without_saving() -> None:
    client = MagicMock(spec=Client)
    called_tool = CalledTool(
        Tool(lambda x: {"squared": x * x}, name="square_tool"),
        _function_call("square_tool", {"x": 5}, call_id="call-sq-42"),
    )
    executor = ToolCallExecutor(
        client=client,
        exit_tool=cast(ExitTool, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
        exit_content_resolver=MagicMock(),
    )

    result = executor.execute(called_tool)

    assert result.raw_result == {"squared": 25}
    assert result.save_result is True
    client.get_session.assert_not_called()
    client.add_tool_response.assert_not_called()
    client.replace_history_from.assert_not_called()


def test_background_tool_output_is_readable_after_node_returns(tmp_path: Path) -> None:
    release = Event()
    started = Event()

    def work(text: str) -> str:
        started.set()
        release.wait()
        return f"finished: {text}"

    provider = SequenceProvider(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "mock-tool",
                        {"text": "task", "execution_mode": "background"},
                        call_id="original-call",
                    )
                ],
            ),
            AssistantMessage("continuing"),
        ]
    )
    provider.generate_id = lambda: "output-one"
    defaults = DefaultCapabilities(
        harness=False,
        include={"read_tool_output", "control_tool_execution"},
        overrides={"mock-tool": Tool(work, name="mock-tool")},
    )
    with SessionCache(tmp_path / "cache") as session:
        provider.get_session = lambda: session
        node = Node(
            [_write_runnable_capabilities(tmp_path)],
            name_hds="main",
            default_capabilities=defaults,
        )
        node.initialize(provider, work_root=tmp_path)
        try:
            response = node.run("start")
            assert response.get_message() == "continuing"
            assert started.wait(1)
            reader = node._runtime._memory_tools
            assert (
                "<response-type>running</response-type>"
                in reader.read_tool_output("output-one").content
            )
        finally:
            release.set()

        assert node._runtime._execution_manager.wait("original-call", 1).status == (
            "completed"
        )
        final = reader.read_tool_output("output-one").content
        assert "<response-type>final</response-type>" in final
        assert "finished: task" in final


def test_completed_background_output_is_inserted_before_next_inference(
    tmp_path: Path,
) -> None:
    release = Event()
    finished = Event()

    def work(text: str) -> str:
        release.wait()
        finished.set()
        return f"finished: {text}"

    provider = SequenceProvider(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "mock-tool",
                        {"text": "task", "execution_mode": "background"},
                        call_id="original-call",
                    )
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call("get_execution_history", {}, call_id="history")
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    original_call = provider.call

    def call(request: ProviderRequest) -> ProviderResponse:
        if len(provider.requests) == 1:
            release.set()
            assert finished.wait(1)
        return original_call(request)

    provider.call = call
    provider.generate_id = lambda: "output-one"
    defaults = DefaultCapabilities(
        harness=False,
        include={"get_execution_history", "read_tool_output"},
        overrides={"mock-tool": Tool(work, name="mock-tool")},
    )
    with SessionCache(tmp_path / "cache") as session:
        provider.get_session = lambda: session
        node = Node(
            [_write_runnable_capabilities(tmp_path)],
            name_hds="main",
            default_capabilities=defaults,
        )
        node.initialize(provider, work_root=tmp_path)
        try:
            assert node.run("start").get_message() == "done"
        finally:
            release.set()

    continuation = provider.requests[2].get_messages()
    synthetic_calls = [
        call
        for message in continuation
        if isinstance(message, AssistantMessage)
        for call in message.get_tool_calls() or ()
        if call.get_name() == "read_tool_output"
    ]
    assert len(synthetic_calls) == 1
    assert synthetic_calls[0].get_arguments() == {"output_id": "output-one"}
    assert any(
        isinstance(message, ToolMessage)
        and message.get_tool_call_id() == synthetic_calls[0].get_id()
        and "finished: task" in message.get_message()
        for message in continuation
    )


def test_executor_returns_preformatted_transient_content_without_header() -> None:
    client = MagicMock(spec=Client)
    called_tool = CalledTool(
        Tool(
            lambda: PreformattedToolResult("<execution-history />"),
            name="runtime_history",
            cache_result=False,
        ),
        _function_call("runtime_history", {}, call_id="call-history"),
    )
    executor = ToolCallExecutor(
        client=client,
        exit_tool=cast(ExitTool, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
        exit_content_resolver=MagicMock(),
    )

    result = executor.execute(called_tool)

    assert result.response.get_message() == "<execution-history />"
    assert result.output_id is None
    assert result.save_result is False
    assert result.raw_result == "<execution-history />"
    client.generate_id.assert_not_called()


def test_runtime_memory_history_is_automatic_xml_and_is_not_cached(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call("get_execution_history", {}, call_id="history-call")
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    assert node.run("show history").get_message() == "done"

    response = cast(ToolMessage, provider.requests[-1].get_messages()[-1])
    assert response.get_message() == (
        "<execution-history>\n"
        "  <total>1</total>\n"
        "  <view>overview</view>\n"
        "  <call>\n"
        "    <position>1</position>\n"
        "    <name>get_execution_history</name>\n"
        "    <id>history-call</id>\n"
        "  </call>\n"
        "</execution-history>"
    )
    assert node._runtime._tool_cache.get_all() == []


def test_runtime_memory_history_accepts_position_navigation(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "get_execution_history",
                        {"start_position": 1, "limit": 2},
                        call_id="history-page",
                    )
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    assert node.run("show history page").get_message() == "done"

    response = cast(ToolMessage, provider.requests[-1].get_messages()[-1])
    assert response.get_message() == (
        "<execution-history>\n"
        "  <total>1</total>\n"
        "  <view>page</view>\n"
        "  <start-position>1</start-position>\n"
        "  <end-position>1</end-position>\n"
        "  <call>\n"
        "    <position>1</position>\n"
        "    <name>get_execution_history</name>\n"
        "    <id>history-page</id>\n"
        "  </call>\n"
        "</execution-history>"
    )


@pytest.mark.parametrize(
    ("invalid_arguments", "expected_error"),
    [
        ({"limit": 1}, "limit must be between 2 and 100"),
        ({"limit": 101}, "limit must be between 2 and 100"),
        ({"unexpected": 1}, "unexpected keyword argument"),
    ],
)
def test_invalid_history_arguments_are_returned_to_model_for_correction(
    tmp_path: Path, invalid_arguments: dict[str, int], expected_error: str
) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "get_execution_history",
                        invalid_arguments,
                        call_id="invalid-history",
                    )
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "get_execution_history", {"limit": 2}, call_id="retry"
                    )
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    assert node.run("show history").get_message() == "done"
    failed_call = next(
        message
        for message in provider.requests[1].get_messages()
        if isinstance(message, ToolMessage)
        and message.get_tool_call_id() == "invalid-history"
    )
    assert expected_error in failed_call.get_message()
    recovered_call = next(
        message
        for message in provider.requests[2].get_messages()
        if isinstance(message, ToolMessage) and message.get_tool_call_id() == "retry"
    )
    assert recovered_call.get_message().startswith("<execution-history>")


def test_runtime_rejects_capabilities_using_reserved_memory_tool_names(
    tmp_path: Path,
) -> None:
    capabilities = _write_runnable_capabilities(tmp_path)
    tool_path = capabilities / "tools" / "get_execution_history"
    tool_path.mkdir()
    (tool_path / "main.py").write_text(
        "from qs.core.agents.tools.decorator import deep_tool\n\n"
        "@deep_tool(name='get_execution_history', description='Reserved')\n"
        "def run(work_root) -> str:\n"
        "    return str(work_root)\n"
    )

    with pytest.raises(
        ValueError,
        match="Harness runtime tool names are reserved: get_execution_history",
    ):
        Node([capabilities], name_hds="main").initialize(
            SequenceProvider([]),
            work_root=tmp_path,
        )


def test_node_run_saves_executed_tools_in_cache(tmp_path: Path) -> None:
    class TestTokenizer(SequentialTokenizer):
        def count_tokens(self, text: str) -> int:
            return len(text)

        def truncate_text(
            self, text: str, start_tokens: int, end_tokens: int = 0
        ) -> str:
            return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

        def _load_vocabulary_tokens(self) -> list[str]:
            return ["hello", "cache"]

    CacheManager.reset()
    try:
        tokenizer = TestTokenizer("mock/model", tmp_path / "vocab")
        tokenizer.initialize()

        capabilities_path = _write_runnable_capabilities(tmp_path)
        tool_call = ToolCall(
            id="call_mock_run",
            function=FunctionCall(name="mock-tool", arguments={"text": "hello_cache"}),
        )
        provider = SequenceProvider(
            [
                AssistantMessage(None, tool_calls=[tool_call]),
                AssistantMessage("All done!"),
            ],
            tokenizer=tokenizer,
        )
        node = Node([capabilities_path], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        response = node.run("run mock tool")

        assert response.get_message() == "All done!"
        session = tokenizer.get_session()
        node_cache = session.node(node._runtime._node_id)
        (record,) = node_cache.get_all()
        assert record is not None
        assert record.name == "mock-tool"
        assert record.tool_call_id == "call_mock_run"
        assert node_cache.get(record.id) == record
        tool_message = provider.requests[-1].get_messages()[-1]
        assert f"<id>{record.id}</id>" in tool_message.get_message()
        assert record.result == {"echo": "hello_cache"}
        assert record.args == {"text": "hello_cache"}
    finally:
        CacheManager.reset()


def test_runtime_saves_only_tools_with_their_original_skill_path(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-skill", {}, call_id="open"),
                    _function_call("mock-tool", {"text": "inside"}, call_id="work"),
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": []},
                        call_id="exit",
                    ),
                ],
            ),
            AssistantMessage("final"),
        ]
    )
    session = SessionCache(tmp_path / "cache")
    provider.get_session = lambda: session
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)
    try:
        response = node.run("start")
        records = session.node(node._runtime._node_id).get_all()
        assert {record.tool_call_id for record in records} == {"work", "open"}
        assert {record.tool_call_id: record.skill_path for record in records} == {
            "work": "/",
            "open": "/mock-skill",
        }
        assert {record.tool_call_id: record.result for record in records} == {
            "work": {"echo": "inside"},
            "open": {"text": "done", "items": []},
        }
        messages = provider.requests[-1].get_messages()
        tool_messages = [
            message for message in messages if isinstance(message, ToolMessage)
        ]
        assert len(tool_messages) == 2
        assert tool_messages[1].get_tool_call_id() == "open"
        assert tool_messages[1].get_message() == "done"
        transcript = response.skill_executions[0].history
        assert any(
            isinstance(message, ToolMessage) and message.get_tool_call_id() == "work"
            for message in transcript
        )
    finally:
        session.close()


@pytest.mark.parametrize("skill_first", [False, True])
def test_skill_compaction_preserves_preceding_parallel_tool_responses(
    tmp_path: Path,
    skill_first: bool,
) -> None:
    calls = [
        _function_call("mock-tool", {"text": "before"}, call_id="work"),
        _function_call("mock-skill", {}, call_id="open"),
    ]
    if skill_first:
        calls.reverse()
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=calls,
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "skill done", "items": []},
                        call_id="exit",
                    )
                ],
            ),
            AssistantMessage("final"),
        ]
    )
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    response = node.run("start")

    assert response.get_message() == "final"
    continuation = provider.requests[-1].get_messages()
    tool_messages = [
        message for message in continuation if isinstance(message, ToolMessage)
    ]
    assert [message.get_tool_call_id() for message in tool_messages] == [
        "work",
        "open",
    ]
    body = tool_messages[0].get_message().split("</header>\n", 1)[1]
    assert ET.fromstring(body).findtext("echo") == "before"
    assert tool_messages[1].get_message() == "skill done"
    records = node._runtime._tool_cache.get_all()
    assert {record.tool_call_id: record.skill_path for record in records} == {
        "work": "/",
        "open": "/mock-skill",
    }


@pytest.mark.parametrize("forbidden", ["mock-sub-tool", "nested-skill", "orphan-tool"])
def test_batch_scope_is_validated_before_any_execution(tmp_path, forbidden):
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-skill", {}, call_id="open"),
                    _function_call(forbidden, {}, call_id="forbidden"),
                ],
            )
        ]
    )
    node = Node([_write_nested_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)
    with pytest.raises(ValueError, match="is not allowed at /"):
        node.run("start")
    assert node.depth == 0
    assert node._runtime._executor.skill_executions == ()
    assert node._runtime._tool_cache is None


def test_sibling_skills_are_not_silently_opened_as_nested_skills(tmp_path):
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-skill", {}, call_id="first"),
                    _function_call(
                        "mock-tool", {"text": "do not execute"}, call_id="work"
                    ),
                    _function_call("mock-skill", {}, call_id="second"),
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "first done", "items": []},
                        call_id="exit-first",
                    )
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "second done", "items": []},
                        call_id="exit-second",
                    )
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    node = Node([_write_nested_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)
    response = node.run("start")
    assert response.get_message() == "done"
    assert node.depth == 0
    assert len(response.skill_executions) == 2
    assert all(not skill.children for skill in response.skill_executions)
    assert [skill.opening_tool_call_id for skill in response.skill_executions] == [
        "first",
        "second",
    ]
    scopes = [request.get_extra()["agentic"] for request in provider.requests]
    assert scopes[2]["skill_call_id"] == "first"
    assert scopes[2]["allowed_skills"] == ["nested-skill"]
    assert scopes[3]["skill_call_id"] == "second"
    assert scopes[3]["allowed_skills"] == ["mock-skill"]
    final_outputs = [
        m.get_tool_call_id()
        for m in provider.requests[-1].get_messages()
        if isinstance(m, ToolMessage)
    ]
    assert set(final_outputs) == {"work", "first", "second"}


def test_nested_mixed_batches_keep_scope_paths_and_complete_history(tmp_path):
    provider = SequenceProvider(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call("mock-skill", {}, call_id="parent"),
                    _function_call("mock-tool", {"text": "root"}, call_id="root-work"),
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call("nested-skill", {}, call_id="child"),
                    _function_call(
                        "mock-sub-tool", {"text": "parent"}, call_id="parent-work"
                    ),
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "nested-sub-tool", {"text": "child"}, call_id="child-work"
                    ),
                    _function_call(
                        "exit_agent",
                        {"text": "first child result", "items": []},
                        call_id="child-exit-extra",
                    ),
                    _function_call(
                        "exit_agent",
                        {"text": "child done", "items": []},
                        call_id="child-exit",
                    ),
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "parent done", "items": []},
                        call_id="parent-exit",
                    ),
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    with SessionCache(tmp_path / "cache") as session:
        provider.get_session = lambda: session
        node = Node([_write_nested_capabilities(tmp_path)], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        assert node.run("check capabilities").get_message() == "done"
        assert node.depth == 0
        assert {
            r.tool_call_id: r.skill_path for r in node._runtime._tool_cache.get_all()
        } == {
            "root-work": "/",
            "parent-work": "/mock-skill",
            "child-work": "/mock-skill/nested-skill",
            "child": "/mock-skill/nested-skill",
            "child-exit-extra": "/mock-skill/nested-skill",
            "parent": "/mock-skill",
        }
        for request in provider.requests:
            calls = [
                c.id
                for m in request.get_messages()
                if isinstance(m, AssistantMessage)
                for c in m.get_tool_calls() or []
            ]
            outputs = [
                m.get_tool_call_id()
                for m in request.get_messages()
                if isinstance(m, ToolMessage)
            ]
            assert sorted(calls) == sorted(outputs)
        scopes = [r.get_extra()["agentic"] for r in provider.requests]
        assert scopes[0]["allowed_skills"] == ["mock-skill"]
        assert scopes[1]["allowed_skills"] == ["nested-skill"]
        assert scopes[2]["allowed_skills"] == []
        assert scopes[3]["allowed_skills"] == ["nested-skill"]
        assert scopes[3]["skill_call_id"] == "child"
        assert scopes[4]["allowed_skills"] == ["mock-skill"]
        assert scopes[4]["skill_call_id"] == "parent"
        assert all(
            r.get_tools().list_tools() == provider.requests[0].get_tools().list_tools()
            for r in provider.requests
        )


def test_runtime_isolates_nodes_with_same_hds_and_call_id(tmp_path: Path) -> None:
    session = SessionCache(tmp_path / "cache")
    capabilities = _write_runnable_capabilities(tmp_path)
    nodes = []
    try:
        for text in ("first", "second"):
            provider = SequenceProvider(
                [
                    AssistantMessage(
                        "",
                        tool_calls=[
                            _function_call(
                                "mock-tool", {"text": text}, call_id="same-call"
                            ),
                        ],
                    ),
                    AssistantMessage("done"),
                ]
            )
            provider.get_session = lambda: session
            node = Node([capabilities], name_hds="main")
            node.initialize(provider, work_root=tmp_path)
            node.run("start")
            nodes.append(node)
        assert nodes[0]._runtime._node_id != nodes[1]._runtime._node_id
        assert [
            session.node(node._runtime._node_id).get_all()[0].result for node in nodes
        ] == [{"echo": "first"}, {"echo": "second"}]
    finally:
        session.close()


def test_runtime_saves_none_result(tmp_path: Path) -> None:
    session = SessionCache(tmp_path / "cache")
    client = MagicMock(spec=Client)
    client.get_session.return_value = session
    runtime = object.__new__(NodeRuntime)
    runtime._client = client
    runtime._node_id = "test-node"
    runtime._execution_manager = ToolExecutionManager()
    runtime._tool_cache = session.node("test-node")
    runtime._emit_tool_state = MagicMock()
    runtime._executor = ToolCallExecutor(
        client=client,
        exit_tool=cast(ExitTool, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
        exit_content_resolver=MagicMock(),
    )
    try:
        runtime._execute_tool_call(
            CalledTool(
                Tool(lambda: None, name="empty"),
                _function_call("empty", {}, call_id="empty-call"),
            )
        )
        (record,) = session.node("test-node").get_all()
        assert record.tool_call_id == "empty-call"
        assert record is not None
        assert record.result is None
        assert record.skill_path == "/"
        client.add_tool_response.assert_called_once()
    finally:
        session.close()


def test_runtime_does_not_overwrite_reused_call_ids(tmp_path: Path) -> None:
    session = SessionCache(tmp_path / "cache")
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-tool", {"text": "first"}, call_id="same")
                ],
            ),
            AssistantMessage("done"),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-tool", {"text": "second"}, call_id="same")
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    provider.get_session = lambda: session
    node = Node(
        [_write_runnable_capabilities(tmp_path)],
        name_hds="main",
        persistent_history=False,
    )
    node.initialize(provider, work_root=tmp_path)
    try:
        node.run("first")
        node.run("second")
        records = node._runtime._tool_cache.get_all()
        assert len(records) == 2
        assert {record.tool_call_id for record in records} == {"same"}
        assert {record.result["echo"] for record in records} == {"first", "second"}
        assert [
            (entry.position, entry.call_id)
            for entry in node._runtime._execution_history.entries
        ] == [(1, "same"), (2, "same")]
    finally:
        session.close()


def test_runtime_refreshes_a_closed_cache(tmp_path: Path) -> None:
    session = SessionCache(tmp_path / "first-cache")
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-tool", {"text": "first"}, call_id="first")
                ],
            ),
            AssistantMessage("done"),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-tool", {"text": "second"}, call_id="second")
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    provider.get_session = lambda: session
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)
    try:
        node.run("first")
        old_cache = node._runtime._tool_cache
        session.close()
        session = SessionCache(tmp_path / "second-cache")
        node.run("second")
        assert node._runtime._tool_cache is not old_cache
        (record,) = node._runtime._tool_cache.get_all()
        assert record.tool_call_id == "second"
    finally:
        session.close()


def test_runtime_keeps_history_and_emits_end_when_cache_write_fails(
    tmp_path: Path, monkeypatch
) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-tool", {"text": "work"}, call_id="work")
                ],
            ),
        ]
    )
    events = []
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(
        provider, work_root=tmp_path, on_stream=as_agent_handler(events.append)
    )

    def fail_store(self, record):
        raise OSError("disk full")

    monkeypatch.setattr(NodeCache, "store", fail_store)
    with pytest.raises(OSError, match="disk full"):
        node.run("start")
    history = node._runtime._client.copy_history()
    assert isinstance(history[-1], ToolMessage)
    assert history[-1].get_tool_call_id() == "work"
    tool_events = [
        event for event in events if isinstance(event, DeepAgentExecutionEvent)
    ]
    assert [event.is_tool_running for event in tool_events] == [True, False]
    assert len(provider.requests) == 1


def test_runtime_captures_arguments_before_tool_mutates_them(tmp_path: Path) -> None:
    def mutate(items: list[int]) -> str:
        items.append(2)
        return "done"

    with SessionCache(tmp_path) as session:
        client = MagicMock(spec=Client)
        runtime = object.__new__(NodeRuntime)
        runtime._execution_manager = ToolExecutionManager()
        runtime._client = client
        runtime._tool_cache = session.node("node")
        runtime._emit_tool_state = MagicMock()
        runtime._executor = ToolCallExecutor(
            client=client,
            exit_tool=cast(ExitTool, object()),
            initial_depth=0,
            agentic_scope=MagicMock(spec=AgenticScope),
            exit_content_resolver=MagicMock(),
        )
        items = [1]
        runtime._execute_tool_call(
            CalledTool(
                Tool(mutate),
                _function_call("mutate", {"items": items}, call_id="call"),
            )
        )
        assert items == [1, 2]
        (record,) = runtime._tool_cache.get_all()
        assert record.args == {"items": [1]}


@pytest.mark.parametrize("streaming", [False, True])
def test_runtime_saves_thinking_from_the_exact_tool_calling_turn(tmp_path, streaming):
    class ThinkingProvider(SequenceProvider):
        def call(self, request):

            response = super().call(request)

            thoughts = [
                "  I open and call two tools.\nCity.  ",
                None,
                "Esco dalla skill.",
                "Risposta finale.",
            ]

            return ProviderResponse(
                response, thinking=ModelThinking(thoughts[len(self.requests) - 1])
            )

    provider = ThinkingProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-skill", {}, call_id="skill"),
                    _function_call("mock-tool", {"text": "first"}, call_id="first"),
                    _function_call("mock-tool", {"text": "second"}, call_id="second"),
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call("mock-sub-tool", {"text": "third"}, call_id="third")
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": []},
                        call_id="exit",
                    )
                ],
            ),
            AssistantMessage("final"),
        ]
    )

    with SessionCache(tmp_path / "cache") as session:
        provider.get_session = lambda: session
        provider.generate_id = iter(
            ["result-first", "result-second", "result-third", "result-skill"]
        ).__next__

        node = Node([_write_nested_capabilities(tmp_path)], name_hds="main")

        node.initialize(
            provider,
            work_root=tmp_path,
            on_stream=as_agent_handler((lambda event: None) if streaming else None),
        )

        node.run("start")

        records = node._runtime._tool_cache.get_all()

        assert {record.tool_call_id: record.thinking for record in records} == {
            "first": "  I open and call two tools.\nCity.  ",
            "second": "  I open and call two tools.\nCity.  ",
            "third": None,
            "skill": "  I open and call two tools.\nCity.  ",
        }

        assert {record.tool_call_id: record.skill_path for record in records} == {
            "first": "/",
            "second": "/",
            "third": "/mock-skill",
            "skill": "/mock-skill",
        }

        skill_response = next(
            message
            for message in provider.requests[-1].get_messages()
            if isinstance(message, ToolMessage)
            and message.get_tool_call_id() == "skill"
        )
        header = ET.fromstring(
            skill_response.get_message().split("</header>")[0] + "</header>"
        )
        assert header.findtext("thinking") == "Esco dalla skill."


def test_skill_exit_header_limits_and_escapes_thinking(tmp_path):
    exit_thinking = "Choice & <result> " + "x" * 100

    class ThinkingProvider(SequenceProvider):
        def call(self, request):
            response = super().call(request)
            thinking = exit_thinking if len(self.requests) == 2 else None
            return ProviderResponse(response, thinking=ModelThinking(thinking))

    provider = ThinkingProvider(
        [
            AssistantMessage(
                None, tool_calls=[_function_call("mock-skill", {}, call_id="skill")]
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": []},
                        call_id="exit",
                    )
                ],
            ),
            AssistantMessage("final"),
        ]
    )
    provider.generate_id = lambda: "result-id"
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)
    node.run("start")

    skill_response = next(
        message
        for message in provider.requests[-1].get_messages()
        if isinstance(message, ToolMessage) and message.get_tool_call_id() == "skill"
    )
    header = ET.fromstring(
        skill_response.get_message().split("</header>")[0] + "</header>"
    )
    excerpt = header.findtext("thinking")
    assert excerpt is not None
    assert exit_thinking.startswith(excerpt)
    assert provider.get_tokenizer().count_tokens(excerpt) == 100
    assert "&amp; &lt;result&gt;" in skill_response.get_message()


def _write_alternative_exit_tool(capabilities_path: Path) -> None:
    selected = capabilities_path / "exit-tool" / "finish"
    alternative = capabilities_path / "exit-tool" / "other"
    alternative.mkdir()
    source = (selected / "main.py").read_text()
    (alternative / "main.py").write_text(
        source.replace("name='exit_agent'", "name='other_exit'")
    )
    (alternative / "instructions.md").write_text("Unselected instructions")


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_initial_scope_and_instructions_use_the_selected_exit(tmp_path, depth):
    root = _write_runnable_capabilities(tmp_path)
    _write_alternative_exit_tool(root)
    provider = SequenceProvider([AssistantMessage("done")])
    node = Node([root], name_hds="main", depth=depth)
    data = node.initialize(provider, work_root=tmp_path)
    node.run("start")
    request = provider.requests[0]
    allowed = request.get_extra()["agentic"]["allowed_tools"]
    assert ("exit_agent" in allowed) == (depth > 0)
    assert "other_exit" not in allowed
    assert "finish" not in request.get_tools().list_tools()
    system = request.get_messages()[0].get_message()
    assert ("Finish the agent" in system) == (depth > 0)
    assert "Unselected instructions" not in system
    assert set(data.get_exit_tool("exit_agent").get_signature().parameters) == {
        "text",
        "items",
    }


def test_unselected_exit_tool_cannot_close_a_skill(tmp_path):
    root = _write_runnable_capabilities(tmp_path)
    _write_alternative_exit_tool(root)
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "other_exit",
                        {"text": "done", "items": []},
                        call_id="forbidden-exit",
                    )
                ],
            )
        ]
    )
    node = Node([root], name_hds="main", depth=1)
    node.initialize(provider, work_root=tmp_path)
    with pytest.raises(ValueError, match="'other_exit' is not allowed"):
        node.run("start")
    assert node.depth == 1


def test_exit_from_positive_initial_depth_returns_to_root_without_open_skill(tmp_path):
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": []},
                        call_id="initial-exit",
                    )
                ],
            ),
            AssistantMessage("final"),
        ]
    )
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main", depth=1)
    node.initialize(provider, work_root=tmp_path)
    response = node.run("start")
    assert node.depth == 0
    assert response.skill_executions == ()
    assert provider.requests[0].get_tool_choice() == "required"
    continuation = provider.requests[1]
    assert continuation.get_tool_choice() is None
    assert "exit_agent" not in continuation.get_extra()["agentic"]["allowed_tools"]
    output = continuation.get_messages()[-1]
    assert output.get_tool_call_id() == "initial-exit"
    assert output.get_message() == "done"
    assert response.get_message() == "final"


@pytest.mark.parametrize("skill_count", [2, 3])
def test_sibling_batches_have_complete_history_and_preserve_request_order(
    tmp_path, skill_count
):
    calls = [
        _function_call("mock-skill", {}, call_id=f"skill-{index}")
        for index in range(skill_count)
    ]
    provider = SequenceProvider(
        [
            AssistantMessage("Open siblings", tool_calls=calls),
            *[
                AssistantMessage(
                    None,
                    tool_calls=[
                        _function_call(
                            "exit_agent",
                            {"text": f"result-{index}", "items": []},
                            call_id=f"exit-{index}",
                        )
                    ],
                )
                for index in range(skill_count)
            ],
            AssistantMessage("done"),
        ]
    )
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)
    response = node.run("start")
    assert [execution.output for execution in response.skill_executions] == [
        f"result-{index}" for index in range(skill_count)
    ]
    assert all(not execution.children for execution in response.skill_executions)
    for request in provider.requests:
        messages = request.get_messages()
        requested = [
            call.id
            for message in messages
            if isinstance(message, AssistantMessage)
            for call in message.get_tool_calls() or []
        ]
        answered = [
            message.get_tool_call_id()
            for message in messages
            if isinstance(message, ToolMessage)
        ]
        assert sorted(requested) == sorted(answered)
    assert [call.id for call in calls] == [
        f"skill-{index}" for index in range(skill_count)
    ]


@pytest.mark.parametrize("exit_first", [False, True])
def test_exit_waits_for_other_tools_in_the_same_batch(tmp_path, exit_first):
    calls = [
        _function_call("mock-sub-tool", {"text": "work"}, call_id="work"),
        _function_call("exit_agent", {"text": "done", "items": []}, call_id="exit"),
    ]
    if exit_first:
        calls.reverse()
    provider = SequenceProvider(
        [
            AssistantMessage(
                None, tool_calls=[_function_call("mock-skill", {}, call_id="open")]
            ),
            AssistantMessage(None, tool_calls=calls),
            AssistantMessage("final"),
        ]
    )
    node = Node([_write_nested_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)
    response = node.run("start")
    assert node.depth == 0
    records = node._runtime._tool_cache.get_all()
    assert {record.tool_call_id: record.skill_path for record in records} == {
        "work": "/mock-skill",
        "open": "/mock-skill",
    }
    assert response.skill_executions[0].output == "done"


def test_missing_agentic_scope_fails_before_executing_calls(tmp_path):
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(SequenceProvider([]), work_root=tmp_path)
    runtime = node._runtime
    assert runtime._executor._agentic_scope.node_id == "main"
    runtime._executor.build_agentic_extra = dict
    with pytest.raises(ValueError, match="scope configuration"):
        runtime._prepare_tool_batch(
            [_function_call("mock-tool", {"text": "x"}, call_id="work")]
        )
    assert runtime._tool_cache is None


def test_parent_exit_waits_for_child_and_keeps_original_thinking(tmp_path):
    class ThinkingProvider(SequenceProvider):
        def call(self, request):
            response = super().call(request)
            return ProviderResponse(
                response, thinking=ModelThinking(f"turn-{len(self.requests)}")
            )

    provider = ThinkingProvider(
        [
            AssistantMessage(
                None, tool_calls=[_function_call("mock-skill", {}, call_id="parent")]
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "parent done", "items": []},
                        call_id="parent-exit",
                    ),
                    _function_call("nested-skill", {}, call_id="child"),
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "child done", "items": []},
                        call_id="child-exit",
                    )
                ],
            ),
            AssistantMessage("final"),
        ]
    )
    node = Node([_write_nested_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)
    response = node.run("start")
    assert node.depth == 0
    parent = response.skill_executions[0]
    assert parent.output == "parent done"
    assert parent.children[0].output == "child done"
    records = node._runtime._tool_cache.get_all()
    assert {
        record.tool_call_id: (record.skill_path, record.thinking) for record in records
    } == {
        "child": ("/mock-skill/nested-skill", "turn-2"),
        "parent": ("/mock-skill", "turn-1"),
    }
    child_request = provider.requests[2]
    assert all(
        call.id != "parent-exit"
        for message in child_request.get_messages()
        if isinstance(message, AssistantMessage)
        for call in message.get_tool_calls() or []
    )
    assert provider.requests[-1].get_extra()["agentic"]["skill_call_id"] == "parent"


def test_sub_agent_tool_runs_direct_child_and_caches_final_response(
    tmp_path: Path,
) -> None:
    class ProviderWithIds(SequenceProvider):
        def __init__(self, responses: list[AssistantMessage]) -> None:
            super().__init__(responses)
            self._next_id = 0

        def generate_id(self) -> str:
            self._next_id += 1
            return f"sub-agent-output-{self._next_id}"

    provider = ProviderWithIds(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "call_sub_agent",
                        {"sub_agent": "child", "text": "first request"},
                        call_id="delegate-1",
                    ),
                    _function_call(
                        "call_sub_agent",
                        {"sub_agent": "child", "text": "second request"},
                        call_id="delegate-2",
                    ),
                ],
            ),
            AssistantMessage("first child result"),
            AssistantMessage("second child result"),
            AssistantMessage("parent done"),
        ]
    )
    parent_capabilities = _write_capabilities(tmp_path / "parent", ("parent",))
    child_capabilities = _write_capabilities(tmp_path / "child", ("child",))
    child = Node([child_capabilities], name_hds="child", persistent_history=True)
    parent = Node(
        [parent_capabilities],
        name_hds="parent",
        children=[child],
        persistent_history=False,
    )
    agent = DeepAgent(provider=provider, node=parent, work_root=tmp_path)

    response = agent.run("delegate twice")

    assert response.get_message() == "parent done"
    assert provider.requests[1].get_messages()[-1].get_message() == "first request"
    second_child_messages = provider.requests[2].get_messages()
    assert [message.get_role_name() for message in second_child_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert second_child_messages[-1].get_message() == "second request"

    parent_tools = provider.requests[0].get_tools()
    child_tools = provider.requests[1].get_tools()
    assert parent_tools is not None
    assert child_tools is not None
    assert "call_sub_agent" in parent_tools.list_tools()
    assert "call_sub_agent" not in child_tools.list_tools()

    continuation_messages = provider.requests[3].get_messages()
    tool_messages = [
        message for message in continuation_messages if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    assert tool_messages[0].get_message().endswith("\nfirst child result")
    assert tool_messages[1].get_message().endswith("\nsecond child result")

    records = parent._runtime._tool_cache.get_all()
    assert [record.name for record in records] == [
        "call_sub_agent",
        "call_sub_agent",
    ]
    assert [record.args for record in records] == [
        {"sub_agent": "child", "text": "first request"},
        {"sub_agent": "child", "text": "second request"},
    ]
    assert [record.result for record in records] == [
        "first child result",
        "second child result",
    ]
    assert all(record.skill_path == "/" for record in records)


def test_sub_agent_resets_non_persistent_child_between_calls(tmp_path: Path) -> None:
    class ProviderWithIds(SequenceProvider):
        def __init__(self, responses: list[AssistantMessage]) -> None:
            super().__init__(responses)
            self._next_id = 0

        def generate_id(self) -> str:
            self._next_id += 1
            return f"non-persistent-output-{self._next_id}"

    provider = ProviderWithIds(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "call_sub_agent",
                        {"sub_agent": "child", "text": "first"},
                        call_id="delegate-1",
                    ),
                    _function_call(
                        "call_sub_agent",
                        {"sub_agent": "child", "text": "second"},
                        call_id="delegate-2",
                    ),
                ],
            ),
            AssistantMessage("first result"),
            AssistantMessage("second result"),
            AssistantMessage("done"),
        ]
    )
    child = Node(
        [_write_capabilities(tmp_path / "child", ("child",))],
        name_hds="child",
        persistent_history=False,
    )
    parent = Node(
        [_write_capabilities(tmp_path / "parent", ("parent",))],
        name_hds="parent",
        children=[child],
    )
    DeepAgent(provider=provider, node=parent, work_root=tmp_path)

    assert parent.run("delegate twice").get_message() == "done"
    assert [
        [message.get_role_name() for message in request.get_messages()]
        for request in provider.requests[1:3]
    ] == [["system", "user"], ["system", "user"]]


def test_sub_agent_input_error_is_a_cached_tool_response(tmp_path: Path) -> None:
    class ProviderWithId(SequenceProvider):
        def generate_id(self) -> str:
            return "sub-agent-error"

    provider = ProviderWithId(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "call_sub_agent",
                        {"sub_agent": "missing", "text": "hello"},
                        call_id="invalid-delegation",
                    )
                ],
            ),
            AssistantMessage("recovered"),
        ]
    )
    parent_capabilities = _write_capabilities(tmp_path / "parent", ("parent",))
    child = Node(
        [_write_capabilities(tmp_path / "child", ("child",))],
        name_hds="child",
    )
    parent = Node([parent_capabilities], name_hds="parent", children=[child])
    DeepAgent(provider=provider, node=parent, work_root=tmp_path)

    response = parent.run("delegate")

    assert response.get_message() == "recovered"
    assert len(provider.requests) == 2
    tool_message = cast(ToolMessage, provider.requests[1].get_messages()[-1])
    assert tool_message.get_message().endswith(
        "\nError: sub_agent must be one of: child."
    )
    record = parent._runtime._tool_cache.get("sub-agent-error")
    assert record.result == "Error: sub_agent must be one of: child."


def test_sub_agent_tool_is_available_in_every_skill_scope(tmp_path: Path) -> None:
    class ProviderWithId(SequenceProvider):
        def __init__(self, responses: list[AssistantMessage]) -> None:
            super().__init__(responses)
            self._next_id = 0

        def generate_id(self) -> str:
            self._next_id += 1
            return f"skill-output-{self._next_id}"

    provider = ProviderWithId(
        [
            AssistantMessage(
                None,
                tool_calls=[_function_call("mock-skill", {}, call_id="parent-skill")],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "call_sub_agent",
                        {"sub_agent": "child", "text": "delegated in skill"},
                        call_id="skill-delegation",
                    )
                ],
            ),
            AssistantMessage("child result"),
            AssistantMessage(
                None,
                tool_calls=[_function_call("nested-skill", {}, call_id="nested-skill")],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "nested done", "items": []},
                        call_id="nested-exit",
                    )
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "parent done", "items": []},
                        call_id="parent-exit",
                    )
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    child = Node(
        [_write_capabilities(tmp_path / "child", ("child",))],
        name_hds="child",
    )
    parent = Node(
        [_write_nested_capabilities(tmp_path / "parent")],
        name_hds="main",
        children=[child],
    )
    DeepAgent(provider=provider, node=parent, work_root=tmp_path)

    assert parent.run("start").get_message() == "done"
    parent_requests = [
        request
        for request in provider.requests
        if request.get_messages()[0].get_message().startswith("Main body\n\n")
    ]
    assert all(
        "call_sub_agent" in request.get_extra()["agentic"]["allowed_tools"]
        for request in parent_requests
    )
    child_requests = [
        request
        for request in provider.requests
        if request.get_messages()[0].get_message().startswith("child body\n\n")
    ]
    assert len(child_requests) == 1
    assert (
        "call_sub_agent"
        not in child_requests[0].get_extra()["agentic"]["allowed_tools"]
    )
    delegation = next(
        record
        for record in parent._runtime._tool_cache.get_all()
        if record.name == "call_sub_agent"
    )
    assert delegation.skill_path == "/mock-skill"
    assert delegation.result == "child result"


def test_each_non_leaf_node_gets_its_own_sub_agent_tool(tmp_path: Path) -> None:
    capabilities = _write_capabilities(
        tmp_path,
        ("root", "child", "grandchild"),
    )
    grandchild = Node([capabilities], name_hds="grandchild")
    child = Node([capabilities], name_hds="child", children=[grandchild])
    root = Node([capabilities], name_hds="root", children=[child])
    provider = SequenceProvider([AssistantMessage("done")])

    DeepAgent(provider=provider, node=root, work_root=tmp_path)

    root_tool = root._runtime._client.tools.get_tool("call_sub_agent")
    child_tool = child._runtime._client.tools.get_tool("call_sub_agent")
    assert isinstance(root_tool, SubAgentTool)
    assert isinstance(child_tool, SubAgentTool)
    assert root_tool.child_names == ("child",)
    assert child_tool.child_names == ("grandchild",)
    assert "<name>child</name><description>child</description>" in root_tool.description
    assert (
        "<name>grandchild</name><description>grandchild</description>"
        in child_tool.description
    )
    assert "call_sub_agent" not in grandchild._runtime._client.tools.list_tools()

    assert root.run("list tools").get_message() == "done"
    request = provider.requests[0]
    assert "call_sub_agent" in request.get_extra()["agentic"]["allowed_tools"]
    tools = request.get_tools()
    assert tools is not None
    assert tools.get_tool("call_sub_agent") is root_tool


def test_rejects_duplicate_direct_child_hds_names(tmp_path: Path) -> None:
    capabilities = _write_capabilities(tmp_path, ("root", "child"))
    root = Node(
        [capabilities],
        name_hds="root",
        children=[
            Node([capabilities], name_hds="child"),
            Node([capabilities], name_hds="child"),
        ],
    )

    with pytest.raises(
        ValueError,
        match="Direct child HDS names must be unique: child",
    ):
        DeepAgent(provider=SequenceProvider([]), node=root, work_root=tmp_path)


def test_rejects_capability_using_reserved_sub_agent_tool_name(
    tmp_path: Path,
) -> None:
    capabilities = _write_capabilities(tmp_path / "parent", ("parent",))
    tool_path = capabilities / "tools" / "call_sub_agent"
    tool_path.mkdir(parents=True)
    (tool_path / "main.py").write_text(
        "from qs.core.agents.tools.decorator import deep_tool\n\n"
        "@deep_tool(name='call_sub_agent', description='Reserved')\n"
        "def run(work_root, text: str) -> str:\n"
        "    return text\n"
    )
    child = Node(
        [_write_capabilities(tmp_path / "child", ("child",))],
        name_hds="child",
    )
    root = Node([capabilities], name_hds="parent", children=[child])

    with pytest.raises(
        ValueError,
        match="Harness runtime tool names are reserved: call_sub_agent",
    ):
        DeepAgent(provider=SequenceProvider([]), node=root, work_root=tmp_path)


def test_node_exposes_custom_agent_context_id(tmp_path: Path) -> None:
    capabilities = _write_runnable_capabilities(tmp_path)
    node = Node(
        [capabilities],
        name_hds="main",
        agent_context_id="custom-ctx-id",
    )
    assert node.agent_context_id == "custom-ctx-id"

    provider = SequenceProvider([AssistantMessage("done")])
    node.initialize(provider, work_root=tmp_path)

    assert node.agent_context_id == "custom-ctx-id"
    node.run("hi")
    assert (
        provider.requests[0].get_extra()["agentic"]["agent_context_id"]
        == "custom-ctx-id"
    )


def test_node_exposes_resolved_runtime_context_id(tmp_path: Path) -> None:
    capabilities = _write_runnable_capabilities(tmp_path)
    node = Node([capabilities], name_hds="main")

    assert node.agent_context_id is None

    node.initialize(SequenceProvider([]), work_root=tmp_path)

    assert node.agent_context_id == "main"


def test_node_rejects_invalid_agent_context_id(tmp_path: Path) -> None:
    capabilities = _write_runnable_capabilities(tmp_path)
    with pytest.raises(TypeError, match="agent_context_id must be a string"):
        Node([capabilities], name_hds="main", agent_context_id=cast(str, 123))

    with pytest.raises(ValueError, match="agent_context_id cannot be empty"):
        Node([capabilities], name_hds="main", agent_context_id="   ")


def test_agentic_scope_validates_agent_context_id(tmp_path: Path) -> None:
    capabilities = _write_runnable_capabilities(tmp_path)
    hds = (
        Node([capabilities], name_hds="main")
        .load_capabilities(work_root=tmp_path)
        .get_hds("main")
    )

    scope = AgenticScope(hds=hds, exit_tool_name="exit_agent")
    assert scope.agent_context_id == "main"
    assert scope.node_id == "main"

    with pytest.raises(TypeError, match="agent_context_id must be a string"):
        AgenticScope(
            hds=hds,
            exit_tool_name="exit_agent",
            agent_context_id=cast(str, 42),
        )

    with pytest.raises(ValueError, match="agent_context_id cannot be empty"):
        AgenticScope(hds=hds, exit_tool_name="exit_agent", agent_context_id="")


def test_prepare_tool_batch_rejects_missing_agent_context_id(tmp_path: Path) -> None:
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(SequenceProvider([]), work_root=tmp_path)
    runtime = node._runtime

    runtime._executor.build_agentic_extra = lambda: {
        "agentic": {
            "allowed_tools": ["mock-tool"],
            "allowed_skills": [],
        }
    }
    with pytest.raises(
        ValueError,
        match=(
            r"Agentic scope configuration \(\'agent_context_id\'\) "
            r"must be a non-empty string"
        ),
    ):
        runtime._prepare_tool_batch(
            [_function_call("mock-tool", {"text": "x"}, call_id="work")]
        )
