from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from dotenv import load_dotenv

import exmekh_lib
from exmekh_lib.config.cache import CACHE_FORMAT
from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.tool_calls import ToolCallReader
from exmekh_lib.utils.file import read_json_from_path
from tests.event_handler import as_agent_handler
from tests.live.model_support import (
    CacheTestTokenizer,
    LiveModelProvider,
    fresh_live_context_id,
)

pytestmark = pytest.mark.live

_TRANSIENT_TOOL_NAMES = {
    RUNTIME.exit_tool_name,
    RUNTIME.history_tool_name,
    RUNTIME.search_tool_name,
    RUNTIME.read_tool_name,
    RUNTIME.execution_tool_name,
}


class RecordingLocalProvider(LiveModelProvider):
    """Observe real model responses without replacing their content or thinking."""

    def __init__(self, directory: Path) -> None:
        self.responses: list[ProviderResponse] = []
        self.requests: list[ProviderRequest] = []
        super().__init__(
            model=os.getenv("EXMEKH_LIB_LIVE_MODEL", DEFAULT_PROVIDER_SETTINGS.model),
            reasoning_effort=ReasoningEffortLMH.LOW,
            api="responses",
            tokenizer=CacheTestTokenizer("live-tool-cache", directory),
        )

    def call(self, request: ProviderRequest) -> ProviderResponse:
        if len(self.responses) >= 16:
            calls = [
                [call.get_name() for call in response.get_tool_calls() or []]
                for response in self.responses
            ]
            raise AssertionError(
                f"MODEL: exceeded the live test turn budget; calls={calls}"
            )
        self.requests.append(request)
        response = super().call(request)
        self.responses.append(response)
        return response


@pytest.fixture
def live_provider(tmp_path: Path) -> Iterator[RecordingLocalProvider]:
    load_dotenv()
    provider = RecordingLocalProvider(tmp_path / "cache")
    try:
        yield provider
    finally:
        session = provider.get_session()
        session.close()
        provider.client.close()


def initialize_node(
    provider: RecordingLocalProvider, work_root: Path, *, streaming: bool = False
) -> Node:
    node = Node(
        capabilities_paths=[Path(exmekh_lib.__file__).parent / "capabilities"],
        name_hds="main",
        persistent_history=False,
        agent_context_id=fresh_live_context_id("live-tool-cache"),
    )
    node.initialize(
        provider,
        work_root=work_root,
        on_stream=as_agent_handler((lambda event: None) if streaming else None),
    )
    return node


def verify_saved_calls(
    node: Node, provider: RecordingLocalProvider
) -> list[ToolCallRecord]:
    transient_names = _TRANSIENT_TOOL_NAMES | {node.get_hds().get_exit_tool_name()}
    executed_ids = {
        message.get_tool_call_id()
        for request in provider.requests
        for message in request.get_messages()
        if isinstance(message, ToolMessage)
    }
    expected = {
        call.id: (
            call,
            response.thinking.thoughts if response.thinking else None,
            request,
        )
        for request, response in zip(provider.requests, provider.responses, strict=True)
        for call in response.get_tool_calls() or []
        if call.get_name() not in transient_names and call.id in executed_ids
    }
    assert expected, "MODEL: expected at least one executable tool call"
    files = list(
        (provider.get_session().session_dir / CACHE_FORMAT.tool_calls_folder).rglob(
            f"*{CACHE_FORMAT.record_suffix}"
        )
    )
    payloads = {}
    for path in files:
        payload = read_json_from_path(path)
        payloads[payload["id"]] = (path, payload)
    records = [
        record
        for directory in sorted({path.parent for path in files})
        for record in ToolCallReader(directory=directory).read_all()
    ]
    assert len(records) == len(expected), "CACHE: missing or unexpected saved outputs"
    assert {record.tool_call_id for record in records} == set(expected)
    for record in records:
        call, thinking, request = expected[record.tool_call_id]
        tools = request.get_tools()
        assert tools is not None
        tool = tools.get_tool(call.get_name())
        assert tool is not None
        assert record.name == call.get_name()
        assert record.args == tool.get_work_arguments(call.get_arguments())
        assert record.thinking == thinking, (
            "CACHE: thinking does not match its calling turn"
        )
        path, payload = payloads[record.id]
        assert "thinking" in payload
        assert payload == record.to_dict(), (
            "CACHE: disk payload differs from the record"
        )
        assert ToolCallReader(directory=path.parent).get(record.id) == record
    return records


def get_tool_response(
    provider: RecordingLocalProvider,
    tool_call_id: str,
) -> str:
    responses = [
        message.get_message()
        for request in provider.requests
        for message in request.get_messages()
        if isinstance(message, ToolMessage)
        and message.get_tool_call_id() == tool_call_id
    ]
    assert responses, f"CLIENT: no response found for tool call {tool_call_id!r}"
    assert len(set(responses)) == 1, (
        f"CLIENT: inconsistent responses found for tool call {tool_call_id!r}"
    )
    return responses[-1]


@pytest.mark.parametrize("streaming", [False, True], ids=["non-streaming", "streaming"])
def test_live_regular_tool_saves_output_and_calling_thinking(
    live_provider, tmp_path, streaming
):
    node = initialize_node(live_provider, tmp_path, streaming=streaming)
    token = "LIVE-CACHE-ROOT"
    response = node.run(
        f"Call mock-tool with text='{token}', then reply only with {token} without other tools."
    )

    records = verify_saved_calls(node, live_provider)
    mock_records = [record for record in records if record.name == "mock-tool"]
    assert mock_records, "MODEL: expected mock-tool among saved calls"
    assert any(
        token.casefold() in record.result.casefold() and record.skill_path == "/"
        for record in mock_records
    )
    record = next(
        record
        for record in mock_records
        if token.casefold() in record.result.casefold() and record.skill_path == "/"
    )
    calling_turn = next(
        response
        for response in live_provider.responses
        if any(
            call.id == record.tool_call_id for call in response.get_tool_calls() or []
        )
    )
    if calling_turn.thinking is None:
        assert record.thinking is None
    else:
        assert isinstance(record.thinking, str) and record.thinking.strip()
    tools = live_provider.requests[0].get_tools()
    assert tools is not None
    search_tool = tools.get_tool(RUNTIME.search_tool_name)
    assert search_tool is not None
    page = search_tool.execute(query=token, skill_path="/")
    assert isinstance(page, PreformattedToolResult)
    search_xml = ET.fromstring(page.content)
    assert any(
        item.findtext("id") == record.id
        for item in search_xml.findall("./records/record")
    )
    assert token.casefold() in (response.get_message() or "").casefold()


def test_live_agent_reads_saved_sub_tool_output_and_matches_exit_code(
    live_provider,
    tmp_path,
):
    node = initialize_node(live_provider, tmp_path)
    exit_tool_name = node.get_hds().get_exit_tool_name()
    response = node.run(
        "Enter mock-skill. Choose a 6-12 character alphanumeric code. Call "
        f"mock-sub-tool with text=code, then {exit_tool_name} with text=code and items=[]. "
        f"At root, {RUNTIME.search_tool_name} for the code and tool_name='mock-sub-tool' "
        "with skill_path='/mock-skill'. Read the matching output ID with "
        f"{RUNTIME.read_tool_name} and cursor=null to read the complete result. "
        "Reply only with the saved code."
    )

    calls = [
        call
        for provider_response in live_provider.responses
        for call in provider_response.get_tool_calls() or []
    ]
    expected_call_names = (
        "mock-skill",
        "mock-sub-tool",
        exit_tool_name,
        RUNTIME.search_tool_name,
        RUNTIME.read_tool_name,
    )
    calls_by_name = {
        name: [call for call in calls if call.get_name() == name]
        for name in expected_call_names
    }
    actual_call_names = [call.get_name() for call in calls]
    assert all(calls_by_name.values()), (
        f"MODEL: expected each requested tool among calls={actual_call_names}"
    )
    records = verify_saved_calls(node, live_provider)
    answer = response.get_message()
    assert isinstance(answer, str), "MODEL: missing final code"
    sub_tool_record = next(
        (
            record
            for record in records
            if record.name == "mock-sub-tool"
            and record.skill_path == "/mock-skill"
            and isinstance(record.args.get("text"), str)
            and 6 <= len(record.args["text"]) <= 12
            and record.args["text"].isalnum()
            and record.args["text"].casefold() in answer.casefold()
        ),
        None,
    )
    assert sub_tool_record is not None, (
        "MODEL: final code does not match a saved tool call"
    )
    sub_tool_call = next(
        call for call in calls if call.id == sub_tool_record.tool_call_id
    )
    code = sub_tool_record.args["text"]
    assert code.casefold() in sub_tool_record.result.casefold()

    skill_records = [record for record in records if record.name == "mock-skill"]
    exit_call = next(
        (
            call
            for call in calls_by_name[exit_tool_name]
            if calls.index(call) > calls.index(sub_tool_call)
            and code.casefold() in str(call.get_arguments().get("text", "")).casefold()
            and call.get_arguments().get("items") == []
            and any(
                record.source == "DeepSkill" and record.result == call.get_arguments()
                for record in skill_records
            )
        ),
        None,
    )
    assert exit_call is not None, "MODEL: no successful exit returned the saved code"

    search_call = next(
        (
            call
            for call in calls_by_name[RUNTIME.search_tool_name]
            if call.get_arguments().get("skill_path") == "/mock-skill"
            and calls.index(call) > calls.index(exit_call)
            and code.casefold() in str(call.get_arguments().get("query", "")).casefold()
            and call.get_arguments().get("tool_name") == "mock-sub-tool"
        ),
        None,
    )
    assert search_call is not None, (
        "MODEL: no search used the absolute mock-skill path: "
        f"{[call.get_arguments() for call in calls_by_name[RUNTIME.search_tool_name]]!r}"
    )
    search_arguments = search_call.get_arguments()
    assert code.casefold() in search_arguments.get("query", "").casefold()
    assert search_arguments.get("tool_name") == "mock-sub-tool"
    search_xml = ET.fromstring(get_tool_response(live_provider, search_call.id))
    assert int(search_xml.findtext("total") or 0) >= 1
    assert any(
        record.findtext("name") == "mock-sub-tool"
        and record.findtext("id") == sub_tool_record.id
        and code.casefold() in (record.findtext("preview") or "").casefold()
        for record in search_xml.findall("./records/record")
    )

    read_attempts = [
        (call, get_tool_response(live_provider, call.id))
        for call in calls_by_name[RUNTIME.read_tool_name]
        if call.get_arguments().get("output_id") == sub_tool_record.id
        and calls.index(call) > calls.index(search_call)
    ]
    successful_read = next(
        (
            content
            for _, content in read_attempts
            if content.startswith("<header>") and "</header>" in content
        ),
        None,
    )
    assert successful_read is not None, (
        f"MODEL: no successful read of saved output; attempts={read_attempts!r}"
    )
    header_xml, _, saved_output = successful_read.partition("</header>")
    header = ET.fromstring(header_xml + "</header>")
    assert header.findtext("name") == "mock-sub-tool"
    assert header.findtext("type") == "Tool"
    assert header.findtext("id") == sub_tool_record.id
    assert header.findtext("./args/text") == code
    assert code.casefold() in saved_output.casefold()
    assert node.depth == 0
