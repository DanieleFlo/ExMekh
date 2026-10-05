from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from pydantic import BaseModel

from exmekh_lib.core.agents.capabilities.defaults import BUILTIN_CAPABILITIES_PATH
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.agents.node.thinking_annotation import ThinkingAnnotationResolver
from exmekh_lib.core.cache.session_cache import SessionCache
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.model_thinking import ModelThinking
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.interfaces.provider import Provider
from exmekh_lib.interfaces.tokenizer import Tokenizer
from tests.tool_result_support import parse_tool_result


class _Tokenizer(Tokenizer):
    def __init__(self, session: SessionCache) -> None:
        self.session = session
        self.next_id = 0

    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

    def initialize(self) -> None:
        pass

    def generate_id(self) -> str:
        self.next_id += 1
        return f"output-{self.next_id}"

    def get_session(self) -> SessionCache:
        return self.session


class _Provider(Provider):
    def __init__(
        self,
        session: SessionCache,
        responses: list[tuple[AssistantMessage, str | None]],
        *,
        summary: str | None = "key point",
    ) -> None:
        super().__init__(_Tokenizer(session))
        self.responses = responses
        self.summary = summary
        self.requests: list[ProviderRequest] = []
        self.summary_requests: list[ProviderRequest] = []

    def call(self, request: ProviderRequest) -> ProviderResponse:
        if request.get_reasoning_effort_override() == "none":
            self.summary_requests.append(request)
            if self.summary is None:
                raise RuntimeError("summary unavailable")
            return ProviderResponse(AssistantMessage(self.summary))
        self.requests.append(request)
        message, thinking = self.responses.pop(0)
        return ProviderResponse(
            message, ModelThinking(thinking) if thinking is not None else None
        )

    def call_structured[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        raise NotImplementedError


def _call(name: str, call_id: str, **arguments: object) -> ToolCall:
    return ToolCall(id=call_id, function=FunctionCall(name, arguments))


def _logic_call(call_id: str, name: str, lower_bound: int) -> ToolCall:
    return _call(
        "logic-consistency",
        call_id,
        variables={name: "Int"},
        premises=[
            {
                "op": ">",
                "left": {"var": name},
                "right": {"const": lower_bound},
            }
        ],
    )


def _responses(
    thinking: str | None, *, two_tools: bool = False
) -> list[tuple[AssistantMessage, str | None]]:
    tool_calls = [_logic_call("first", "x", 1)]
    if two_tools:
        tool_calls.append(_logic_call("second", "y", 0))
    return [
        (AssistantMessage(None, tool_calls=[_call("deep-thinking", "open")]), None),
        (AssistantMessage(None, tool_calls=tool_calls), thinking),
        (
            AssistantMessage(
                None, tool_calls=[_call("exit_agent", "exit", text="done", items=[])]
            ),
            "exit trace",
        ),
        (AssistantMessage("complete"), None),
    ]


def _tool_content(request: ProviderRequest, call_id: str) -> str:
    return next(
        message.get_message()
        for message in request.get_messages()
        if isinstance(message, ToolMessage) and message.get_tool_call_id() == call_id
    )


def _header(content: str) -> ET.Element:
    return ET.fromstring(content.split("</header>", 1)[0] + "</header>")


def test_date_time_result_survives_skill_compaction_as_xml(tmp_path: Path) -> None:
    responses = [
        (AssistantMessage(None, tool_calls=[_call("deep-thinking", "open")]), None),
        (
            AssistantMessage(
                None,
                tool_calls=[
                    _call(
                        "date-time",
                        "date-call",
                        operation="difference",
                        value="2026-03-29",
                        other="2026-03-30",
                        timezone="Europe/Rome",
                        unit="hours",
                    )
                ],
            ),
            "Measure elapsed hours through the daylight-saving transition.",
        ),
        (
            AssistantMessage(
                None,
                tool_calls=[
                    _call(
                        "exit_agent",
                        "exit",
                        text="Elapsed time checked.",
                        items=["date-call"],
                    )
                ],
            ),
            None,
        ),
        (AssistantMessage("complete"), None),
    ]
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(session, responses)
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)

        assert node.run("Measure elapsed time.").get_message() == "complete"

        content = _tool_content(provider.requests[2], "date-call")
        result = parse_tool_result(content)
        assert isinstance(result, dict)
        assert result["result"] == 23
        assert result["total_seconds"] == 82_800
        assert _header(content).findtext("type") == "Tool"
        assert _header(content).findtext("name") == "date-time"
        compacted = _tool_content(provider.requests[3], "open")
        retained = ET.fromstring("<tool-output" + compacted.split("<tool-output", 1)[1])
        assert retained.get("name") == "date-time"
        assert parse_tool_result(retained.findtext("content", "")) == result


def test_direct_tool_and_exit_receive_short_thinking(tmp_path: Path) -> None:
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(session, _responses("short reasoning"))
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        assert node.run("investigate").get_message() == "complete"

        tool_header = _header(_tool_content(provider.requests[2], "first"))
        exit_header = _header(_tool_content(provider.requests[3], "open"))
        assert tool_header.findtext("thinking") == "short reasoning"
        assert exit_header.findtext("thinking") == "exit trace"
        assert provider.summary_requests == []


def test_long_thinking_is_summarized_once_for_batch_without_history(
    tmp_path: Path,
) -> None:
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(
            session, _responses("x" * 201, two_tools=True), summary="key point" * 100
        )
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        node.run("investigate")

        for call_id in ("first", "second"):
            assert (
                _header(_tool_content(provider.requests[2], call_id)).findtext(
                    "thinking"
                )
                == "key point" * 100
            )
        assert len(provider.summary_requests) == 1
        summary_request = provider.summary_requests[0]
        assert summary_request.get_tools() is None
        assert len(summary_request.get_messages()) == 2
        assert summary_request.get_reasoning_effort_override() == "none"


def test_same_thinking_reuses_summary_across_batches_in_one_turn(
    tmp_path: Path,
) -> None:
    trace = "x" * 201
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(
            session,
            [
                (
                    AssistantMessage(None, tool_calls=[_call("deep-thinking", "open")]),
                    None,
                ),
                (
                    AssistantMessage(None, tool_calls=[_logic_call("first", "x", 0)]),
                    trace,
                ),
                (
                    AssistantMessage(None, tool_calls=[_logic_call("second", "y", 1)]),
                    trace,
                ),
                (
                    AssistantMessage(
                        None,
                        tool_calls=[_call("exit_agent", "exit", text="done", items=[])],
                    ),
                    None,
                ),
                (AssistantMessage("complete"), None),
            ],
        )
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        node.run("investigate")
        assert len(provider.summary_requests) == 1
        assert (
            _header(_tool_content(provider.requests[2], "first")).findtext("thinking")
            == "key point"
        )
        assert (
            _header(_tool_content(provider.requests[3], "second")).findtext("thinking")
            == "key point"
        )


def test_missing_summary_does_not_block_tool(tmp_path: Path) -> None:
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(session, _responses("x" * 201), summary=None)
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        assert node.run("investigate").get_message() == "complete"
        assert (
            _header(_tool_content(provider.requests[2], "first")).find("thinking")
            is None
        )


def test_token_count_failure_omits_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(session, [])

        def fail_count(_text: str) -> int:
            raise RuntimeError("Tokenizer unavailable")

        monkeypatch.setattr(provider.get_tokenizer(), "count_tokens", fail_count)
        annotations = ThinkingAnnotationResolver(provider)
        assert annotations.resolve("reasoning") is None
        assert annotations.resolve("reasoning") is None
        assert provider.summary_requests == []


def test_exactly_200_tokens_are_kept_without_summary(tmp_path: Path) -> None:
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(session, _responses("x" * 200))
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        node.run("investigate")
        assert (
            _header(_tool_content(provider.requests[2], "first")).findtext("thinking")
            == "x" * 200
        )
        assert provider.summary_requests == []


def test_other_scopes_do_not_annotate_tool_headers(tmp_path: Path) -> None:
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(
            session,
            [
                (
                    AssistantMessage(
                        None, tool_calls=[_call("mock-tool", "root", text="hello")]
                    ),
                    "outside trace",
                ),
                (AssistantMessage("complete"), None),
            ],
        )
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        node.run("hello")
        assert (
            _header(_tool_content(provider.requests[1], "root")).find("thinking")
            is None
        )


def test_existing_saved_output_header_is_annotated_in_skill(tmp_path: Path) -> None:
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(
            session,
            [
                (
                    AssistantMessage(None, tool_calls=[_call("deep-thinking", "open")]),
                    None,
                ),
                (
                    AssistantMessage(None, tool_calls=[_logic_call("work", "x", 1)]),
                    "work trace",
                ),
                (
                    AssistantMessage(
                        None,
                        tool_calls=[
                            _call("read_tool_output", "read", output_id="output-1")
                        ],
                    ),
                    "read trace",
                ),
                (
                    AssistantMessage(
                        None,
                        tool_calls=[_call("exit_agent", "exit", text="done", items=[])],
                    ),
                    None,
                ),
                (AssistantMessage("complete"), None),
            ],
        )
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        node.run("investigate")
        content = _tool_content(provider.requests[3], "read")
        assert _header(content).findtext("thinking") == "read trace"
        body = content.split("</header>\n", 1)[1]
        assert ET.fromstring(body).findtext("status") == "consistent"
        assert ET.fromstring(body).findtext("backend") == "z3"


def test_headerless_tool_result_does_not_trigger_summarizer(tmp_path: Path) -> None:
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(
            session,
            [
                (
                    AssistantMessage(None, tool_calls=[_call("deep-thinking", "open")]),
                    None,
                ),
                (
                    AssistantMessage(
                        None, tool_calls=[_call("get_execution_history", "history")]
                    ),
                    "x" * 201,
                ),
                (
                    AssistantMessage(
                        None,
                        tool_calls=[_call("exit_agent", "exit", text="done", items=[])],
                    ),
                    None,
                ),
                (AssistantMessage("complete"), None),
            ],
        )
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        node.run("investigate")
        assert _tool_content(provider.requests[2], "history").startswith(
            "<execution-history>"
        )
        assert provider.summary_requests == []


def test_invalid_history_arguments_can_be_corrected_inside_skill(
    tmp_path: Path,
) -> None:
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(
            session,
            [
                (
                    AssistantMessage(None, tool_calls=[_call("deep-thinking", "open")]),
                    None,
                ),
                (
                    AssistantMessage(
                        None,
                        tool_calls=[_call("get_execution_history", "invalid", limit=1)],
                    ),
                    "x" * 201,
                ),
                (
                    AssistantMessage(
                        None,
                        tool_calls=[_call("get_execution_history", "retry", limit=2)],
                    ),
                    "valid trace",
                ),
                (
                    AssistantMessage(
                        None,
                        tool_calls=[_call("exit_agent", "exit", text="done", items=[])],
                    ),
                    None,
                ),
                (AssistantMessage("complete"), None),
            ],
        )
        node = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main")
        node.initialize(provider, work_root=tmp_path)

        assert node.run("investigate").get_message() == "complete"
        assert "limit must be between 2 and 100" in _tool_content(
            provider.requests[2], "invalid"
        )
        assert _tool_content(provider.requests[3], "retry").startswith(
            "<execution-history>"
        )
        assert provider.summary_requests == []


def test_nested_skill_calls_keep_their_existing_header_behavior(tmp_path: Path) -> None:
    root = tmp_path / "capabilities"
    (root / "hds").mkdir(parents=True)
    (root / "deep-skill").mkdir()
    (root / "tools" / "echo").mkdir(parents=True)
    (root / "exit-tool" / "finish").mkdir(parents=True)
    (root / "hds" / "main.md").write_text(
        "---\ndescription: main\nexit-tool: exit_agent\nsub-skills:\n"
        "  - deep-thinking\n---\nMain"
    )
    (root / "deep-skill" / "deep-thinking.md").write_text(
        "---\ndescription: Think\nsub-skills:\n  - child\nsub-tools:\n"
        "  - echo\n---\nThink"
    )
    (root / "deep-skill" / "child.md").write_text(
        "---\ndescription: Child\nsub-tools:\n  - echo\n---\nChild"
    )
    (root / "tools" / "echo" / "main.py").write_text(
        "from pathlib import Path\n"
        "from exmekh_lib.core.agents.tools.decorator import deep_tool\n"
        "@deep_tool(name='echo', description='Echo text')\n"
        "def run(work_root: Path, text: str) -> str:\n"
        "    return text\n"
    )
    (root / "exit-tool" / "finish" / "instructions.md").write_text("Finish")
    (root / "exit-tool" / "finish" / "main.py").write_text(
        "from exmekh_lib.core.agents.exit_tool import ExitToolData, exit_tool\n"
        "@exit_tool(name='exit_agent', description='Finish', instructions_path='instructions.md')\n"
        "def finish(text: str, items: list[str]) -> ExitToolData:\n"
        "    return ExitToolData(text=text, items=items)\n"
    )
    responses = [
        (AssistantMessage(None, tool_calls=[_call("deep-thinking", "parent")]), None),
        (AssistantMessage(None, tool_calls=[_call("child", "child-open")]), None),
        (
            AssistantMessage(
                None, tool_calls=[_call("echo", "child-tool", text="child")]
            ),
            "child trace",
        ),
        (
            AssistantMessage(
                None,
                tool_calls=[
                    _call("exit_agent", "child-exit", text="child done", items=[])
                ],
            ),
            None,
        ),
        (
            AssistantMessage(
                None, tool_calls=[_call("echo", "parent-tool", text="parent")]
            ),
            "parent trace",
        ),
        (
            AssistantMessage(
                None,
                tool_calls=[
                    _call("exit_agent", "parent-exit", text="parent done", items=[])
                ],
            ),
            None,
        ),
        (AssistantMessage("complete"), None),
    ]
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(session, responses)
        node = Node([root], name_hds="main")
        node.initialize(provider, work_root=tmp_path)
        node.run("investigate")
        assert (
            _header(_tool_content(provider.requests[3], "child-tool")).find("thinking")
            is None
        )
        assert (
            _header(_tool_content(provider.requests[5], "parent-tool")).findtext(
                "thinking"
            )
            == "parent trace"
        )


def test_existing_header_can_receive_thinking_without_changing_body() -> None:
    content = XmlHeader("Tool", "saved", "item-1").prepend_to("body")
    updated = XmlHeader.add_field_to_content(content, "thinking", "summary & detail")
    assert _header(updated).findtext("thinking") == "summary & detail"
    assert updated.endswith("\nbody")
    assert (
        XmlHeader.add_field_to_content("<other>body</other>", "thinking", "x")
        == "<other>body</other>"
    )


@pytest.mark.parametrize("summary_text", ["", "   "])
def test_empty_thinking_summary_is_omitted_and_cached_for_the_turn(
    tmp_path, summary_text
):
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(session, [], summary=summary_text)
        annotations = ThinkingAnnotationResolver(provider)
        assert annotations.resolve("x" * 201) is None
        assert annotations.resolve("x" * 201) is None
        assert len(provider.summary_requests) == 1
        request = provider.summary_requests[0]
        assert request.get_tool_choice() == "none"
        assert request.get_tools() is None
        assert len(request.get_messages()) == 2


def test_thinking_summary_rejects_unexpected_tool_calls(tmp_path):
    with SessionCache(tmp_path / "cache") as session:
        provider = _Provider(session, [])
        provider.call = lambda request: ProviderResponse(
            AssistantMessage("summary", tool_calls=[_call("unexpected", "call")])
        )
        annotations = ThinkingAnnotationResolver(provider)
        assert annotations.resolve("x" * 201) is None
