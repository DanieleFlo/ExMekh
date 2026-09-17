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
from qs.core.agents.node.runtime import NodeRuntimeConfig
from qs.core.agents.node.tool_call_executor import ToolCallExecutor
from qs.core.agents.node.tool_call_resolver import ToolCallResolver
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


class SequenceProvider(Provider):
    def __init__(self, responses: list[AssistantMessage]) -> None:
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

    assert config.streaming is False
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

    executor.execute(called_tool)

    called_tool.execute.assert_called_once_with()
    response = cast(ToolMessage, client.add_tool_response.call_args.args[0])
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

    executor.execute(called_tool)

    called_tool.execute.assert_called_once_with()
    response = cast(ToolMessage, client.add_tool_response.call_args.args[0])
    expected_header = XmlHeader(source="Tool", name="echo", id="id-tool").to_xml()
    assert response.get_message() == f"{expected_header}\n{{\"echo\": \"hello\"}}"
    assert response.get_tool_call_id() == "call-123"
    assert response.get_name() == "echo"


def test_tool_failure_still_emits_the_end_state() -> None:
    called_tool = MagicMock(spec=CalledTool)
    called_tool.execute.side_effect = RuntimeError("tool failed")
    called_tool.call_id = "call-123"
    called_tool.name = "echo"
    states: list[tuple[str, str, bool]] = []
    executor = ToolCallExecutor(
        client=MagicMock(spec=Client),
        exit_skill=cast(ExitSkill, object()),
        initial_depth=0,
        agentic_scope=MagicMock(spec=AgenticScope),
        on_state=lambda name, call_id, running: states.append((name, call_id, running)),
    )

    with pytest.raises(RuntimeError, match="tool failed"):
        executor.execute(called_tool)

    assert states == [
        ("echo", "call-123", True),
        ("echo", "call-123", False),
    ]


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
        source="DeepSkill", name="exit_agent", id="exit-id-456"
    ).to_xml()
    assert exit_message.get_message() == f"{expected_header}\ndone"


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


def test_exit_skill_keeps_depth_at_zero(tmp_path: Path) -> None:
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

    node.run("start")

    assert node.depth == 0
    assert [request.get_tool_choice() for request in provider.requests] == [None, None]


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
    node = Node([_write_runnable_capabilities(tmp_path)], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    node.run("start")

    response = cast(ToolMessage, provider.requests[1].get_messages()[-1])
    assert response.get_message() == "Finish the agent"
    assert response.get_tool_call_id() == "skill-call"
    assert node.depth == 0


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
        [skill_call, skill_call, exit_call, exit_call, AssistantMessage("final")]
    )
    node = Node(
        [_write_runnable_capabilities(tmp_path)],
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
