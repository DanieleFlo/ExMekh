from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

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
from qs.core.agents.exit_skill.exit_skill import ExitSkill
from qs.core.agents.hds.hds import HDS
from qs.core.agents.node import Node
from qs.core.agents.node.agentic_scope import AgenticScope
from qs.core.agents.node.node_runtime import NodeRuntime
from qs.core.agents.node.node_runtime_config import NodeRuntimeConfig
from qs.core.agents.node.tool_call_executor import ToolCallExecutor
from qs.core.agents.node.tool_call_resolver import ToolCallResolver
from qs.core.cache.cache_manager import CacheManager
from qs.core.cache.session_cache import SessionCache
from qs.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from qs.core.models.client import Client
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.tool import CalledTool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import ErrorUserMessage
from qs.core.models.xml_header import XmlHeader
from qs.interfaces.provider import Provider
from qs.interfaces.tokenizer import Tokenizer


class _StubTokenizer(Tokenizer):
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
    exit_skill_path = capabilities_path / "exit-skill" / "finish.md"
    exit_tool_path = capabilities_path / "exit-tool" / "finish"
    hds_path.mkdir(parents=True)
    exit_skill_path.parent.mkdir(parents=True)
    exit_tool_path.mkdir(parents=True)

    for name in hds_names:
        (hds_path / f"{name}.md").write_text(
            f"---\ndescription: {name}\nexit-skill: finish\n---\n{name} body"
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


def _write_runnable_capabilities(parent: Path) -> Path:
    capabilities_path = _write_capabilities(parent, ("main",))
    hds_path = capabilities_path / "hds" / "main.md"
    hds_path.write_text(
        "---\n"
        "description: Main instructions\n"
        "exit-skill: finish\n"
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
    first_exit_skill = first_path / "exit-skill" / "finish.md"
    second_exit_skill = second_path / "exit-skill" / "finish.md"
    first_exit_skill.rename(first_exit_skill.with_name("finish-first.md"))
    second_exit_skill.rename(second_exit_skill.with_name("finish-second.md"))
    (first_path / "exit-tool" / "finish").rename(
        first_path / "exit-tool" / "finish-first"
    )
    (second_path / "exit-tool" / "finish").rename(
        second_path / "exit-tool" / "finish-second"
    )
    (first_path / "hds" / "first.md").write_text(
        "---\ndescription: First\nexit-skill: finish-first\n---\nFirst body"
    )
    (second_path / "hds" / "second.md").write_text(
        "---\ndescription: Second\nexit-skill: finish-second\n---\nSecond body"
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

    with pytest.raises(TypeError, match="on_stream must be a callable"):
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
        exit_skill=cast(ExitSkill, object()),
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


def test_run_requires_an_initialized_node(tmp_path: Path) -> None:
    node = Node([tmp_path / "capabilities"], name_hds="main")

    with pytest.raises(RuntimeError, match="must be initialized before run"):
        node.run("hello")


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
        "finish": "Finish Available in every skill.",
        "exit_agent": "Exit agent Available in every skill.",
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
        exit_skill=cast(ExitSkill, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
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
        exit_skill=cast(ExitSkill, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
    )

    result = executor.execute(called_tool)

    called_tool.execute.assert_called_once_with()
    client.add_tool_response.assert_not_called()
    client.get_session.assert_not_called()
    response = result.response
    assert response.get_message() == '{"echo": "hello"}'
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
        exit_skill=cast(ExitSkill, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
    )

    result = executor.execute(called_tool)

    called_tool.execute.assert_called_once_with()
    client.add_tool_response.assert_not_called()
    client.get_session.assert_not_called()
    response = result.response
    expected_header = XmlHeader(source="Tool", name="echo", id="id-tool").to_xml()
    assert response.get_message() == f'{expected_header}\n{{"echo": "hello"}}'
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
        exit_skill=cast(ExitSkill, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
    )
    runtime = object.__new__(NodeRuntime)
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
                tool_calls=[_function_call("mock-skill", {}, call_id="skill-call")],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": ["completed"]},
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
        '<exit-skill tools="exit_agent">Finish the agent</exit-skill>'
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
                tool_calls=[_function_call("mock-skill", {}, call_id="skill-call")],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": ["completed"]},
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
    assert record.tool_call_id == "exit-call"
    assert record.skill_path == "/mock-skill"
    assert record.result == {"text": "done", "items": ["completed"]}


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
                        {"text": "nested done", "items": ["completed"]},
                        call_id="nested-exit-call",
                    )
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "parent done", "items": ["completed"]},
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
                "allowed_tools": ["mock-tool"],
                "allowed_skills": ["mock-skill"],
            }
        },
        {
            "agentic": {
                "allowed_tools": [
                    "mock-sub-tool",
                    "shared-tool",
                    "finish",
                    "exit_agent",
                ],
                "allowed_skills": ["nested-skill"],
            }
        },
        {
            "agentic": {
                "allowed_tools": [
                    "nested-sub-tool",
                    "shared-tool",
                    "mock-sub-tool",
                    "finish",
                    "exit_agent",
                ],
                "allowed_skills": [],
            }
        },
        {
            "agentic": {
                "allowed_tools": [
                    "mock-sub-tool",
                    "shared-tool",
                    "finish",
                    "exit_agent",
                ],
                "allowed_skills": ["nested-skill"],
                "operation": "return",
                "skill_call_id": "nested-skill-call",
            }
        },
        {
            "agentic": {
                "allowed_tools": ["mock-tool"],
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
                        {"text": "nested done", "items": ["completed"]},
                        call_id="nested-exit-call",
                    )
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "parent done", "items": ["completed"]},
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
        on_stream=events.append,
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
    assert nested_tool_end.history[-1] == {
        "role": "tool",
        "content": '{"echo": "work"}',
    }
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
                        {"text": "done", "items": ["completed"]},
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


def test_exit_skill_adds_its_instructions_without_changing_depth(
    tmp_path: Path,
) -> None:
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[_function_call("finish", {}, call_id="skill-call")],
            ),
            AssistantMessage("final answer"),
        ]
    )
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main", depth=1)
    node.initialize(provider, work_root=tmp_path)

    node.run("start")

    response = cast(ToolMessage, provider.requests[1].get_messages()[-1])
    assert response.get_message() == "Finish the agent"
    assert response.get_tool_call_id() == "skill-call"
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
                {"text": "done", "items": ["completed"]},
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
    assert tool_message.get_message() == '{"echo": "hello"}'


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
    from qs.core.models.tool import Tool

    client = MagicMock(spec=Client)
    called_tool = CalledTool(
        Tool(lambda x: {"squared": x * x}, name="square_tool"),
        _function_call("square_tool", {"x": 5}, call_id="call-sq-42"),
    )
    executor = ToolCallExecutor(
        client=client,
        exit_skill=cast(ExitSkill, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
    )

    result = executor.execute(called_tool)

    assert result.raw_result == {"squared": 25}
    assert result.save_result is True
    client.get_session.assert_not_called()
    client.add_tool_response.assert_not_called()
    client.replace_history_from.assert_not_called()


def test_node_run_saves_executed_tools_in_cache(tmp_path: Path) -> None:
    from qs.core.cache.cache_manager import CacheManager
    from qs.core.tokenizers.sequential_tokenizer import SequentialTokenizer

    class TestTokenizer(SequentialTokenizer):
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
                    _function_call("finish", {}, call_id="instructions"),
                    _function_call(
                        "exit_agent",
                        {"text": "done", "items": ["completed"]},
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
        assert {record.tool_call_id for record in records} == {"work", "exit"}
        assert {record.tool_call_id: record.skill_path for record in records} == {
            "work": "/",
            "exit": "/mock-skill",
        }
        assert {record.tool_call_id: record.result for record in records} == {
            "work": {"echo": "inside"},
            "exit": {"text": "done", "items": ["completed"]},
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
                        {"text": "skill done", "items": ["completed"]},
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
    assert tool_messages[0].get_message() == '{"echo": "before"}'
    assert tool_messages[1].get_message() == "skill done"
    records = node._runtime._tool_cache.get_all()
    assert {record.tool_call_id: record.skill_path for record in records} == {
        "work": "/",
        "exit": "/mock-skill",
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
                        {"text": "second done", "items": ["second"]},
                        call_id="exit-second",
                    )
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "first done", "items": ["first"]},
                        call_id="exit-first",
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
        "second",
        "first",
    ]
    scopes = [request.get_extra()["agentic"] for request in provider.requests]
    assert scopes[2]["skill_call_id"] == "second"
    assert scopes[2]["allowed_skills"] == ["nested-skill"]
    assert scopes[3]["skill_call_id"] == "first"
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
                        {"text": "first child result", "items": ["child"]},
                        call_id="child-exit-extra",
                    ),
                    _function_call(
                        "exit_agent",
                        {"text": "child done", "items": ["child"]},
                        call_id="child-exit",
                    ),
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    _function_call(
                        "exit_agent",
                        {"text": "parent done", "items": ["parent"]},
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
            "child-exit": "/mock-skill/nested-skill",
            "child-exit-extra": "/mock-skill/nested-skill",
            "parent-exit": "/mock-skill",
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
    from qs.core.models.tool import Tool

    session = SessionCache(tmp_path / "cache")
    client = MagicMock(spec=Client)
    client.get_session.return_value = session
    runtime = object.__new__(NodeRuntime)
    runtime._client = client
    runtime._node_id = "test-node"
    runtime._tool_cache = session.node("test-node")
    runtime._emit_tool_state = MagicMock()
    runtime._executor = ToolCallExecutor(
        client=client,
        exit_skill=cast(ExitSkill, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
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
    from qs.core.cache.node_cache import NodeCache

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
    node.initialize(provider, work_root=tmp_path, on_stream=events.append)

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
    from qs.core.models.tool import Tool

    def mutate(items: list[int]) -> str:
        items.append(2)
        return "done"

    with SessionCache(tmp_path) as session:
        client = MagicMock(spec=Client)
        runtime = object.__new__(NodeRuntime)
        runtime._client = client
        runtime._tool_cache = session.node("node")
        runtime._emit_tool_state = MagicMock()
        runtime._executor = ToolCallExecutor(
            client=client,
            exit_skill=cast(ExitSkill, object()),
            initial_depth=0,
            agentic_scope=MagicMock(spec=AgenticScope),
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

    from qs.core.models.model_thinking import ModelThinking

    class ThinkingProvider(SequenceProvider):
        def call(self, request):

            response = super().call(request)

            thoughts = [
                "  Apro e chiamo due tool.\nCittà.  ",
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
                        {"text": "done", "items": ["done"]},
                        call_id="exit",
                    )
                ],
            ),
            AssistantMessage("final"),
        ]
    )

    with SessionCache(tmp_path / "cache") as session:
        provider.get_session = lambda: session

        node = Node([_write_nested_capabilities(tmp_path)], name_hds="main")

        node.initialize(
            provider,
            work_root=tmp_path,
            on_stream=(lambda event: None) if streaming else None,
        )

        node.run("start")

        records = node._runtime._tool_cache.get_all()

        assert {record.tool_call_id: record.thinking for record in records} == {
            "first": "  Apro e chiamo due tool.\nCittà.  ",
            "second": "  Apro e chiamo due tool.\nCittà.  ",
            "third": None,
            "exit": "Esco dalla skill.",
        }

        assert {record.tool_call_id: record.skill_path for record in records} == {
            "first": "/",
            "second": "/",
            "third": "/mock-skill",
            "exit": "/mock-skill",
        }
