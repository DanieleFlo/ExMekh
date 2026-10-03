from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from dotenv import load_dotenv

import exmekh_lib
from exmekh_lib.capabilities.tools.search_tool_outputs.tool_call_search import (
    ToolCallSearch,
)
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.tool_calls import ToolCallReader
from exmekh_lib.utils.file import read_json_from_path
from tests.event_handler import as_agent_handler
from tests.live.model_support import CacheTestTokenizer, LiveModelProvider

pytestmark = pytest.mark.live

_TRANSIENT_TOOL_NAMES = {
    "exit_agent",
    "get_execution_history",
    "search_tool_outputs",
    "read_tool_output",
    "control_tool_execution",
}


class RecordingLocalProvider(LiveModelProvider):
    """Observe real model responses without replacing their content or thinking."""

    def __init__(self, directory: Path) -> None:
        self.responses: list[ProviderResponse] = []
        self.requests: list[ProviderRequest] = []
        super().__init__(
            model=os.getenv("EXMEKH_LIB_LIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S"),
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
    )
    node.initialize(
        provider,
        work_root=work_root,
        on_stream=as_agent_handler((lambda event: None) if streaming else None),
    )
    return node


def saved_cache(node: Node) -> NodeCache:
    assert node._runtime is not None
    cache = node._runtime._tool_cache
    assert cache is not None, "CACHE: model tool calls produced no node cache"
    return cache


def verify_saved_calls(
    node: Node, provider: RecordingLocalProvider
) -> list[ToolCallRecord]:
    executed_ids = {
        message.get_tool_call_id()
        for request in provider.requests
        for message in request.get_messages()
        if isinstance(message, ToolMessage)
    }
    expected = {
        call.id: (call, response.thinking.thoughts if response.thinking else None)
        for response in provider.responses
        for call in response.get_tool_calls() or []
        if call.get_name() not in _TRANSIENT_TOOL_NAMES and call.id in executed_ids
    }
    assert expected, "MODEL: expected at least one executable tool call"
    cache = saved_cache(node)
    records = cache.get_all()
    assert len(records) == len(expected), "CACHE: missing or unexpected saved outputs"
    assert {record.tool_call_id for record in records} == set(expected)
    files = list((cache.session_dir / "tool_calls").rglob("*.json"))
    payloads = {}
    for path in files:
        payload = read_json_from_path(path)
        payloads[payload["id"]] = (path, payload)
    for record in records:
        call, thinking = expected[record.tool_call_id]
        assert record.name == call.get_name()
        assert record.args == call.get_arguments()
        assert record.thinking == thinking, (
            "CACHE: thinking does not match its calling turn"
        )
        assert cache.get(record.id) == record
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
    page = ToolCallSearch(saved_cache(node)).search(query=token, skill_path="/")
    assert any(item["id"] == record.id for item in page["records"])
    assert token.casefold() in (response.get_message() or "").casefold()


def test_live_agent_reads_saved_sub_tool_output_and_matches_exit_code(
    live_provider,
    tmp_path,
):
    node = initialize_node(live_provider, tmp_path)
    response = node.run(
        "Enter mock-skill. Choose a 6-12 character alphanumeric code. Call "
        "mock-sub-tool with text=code, then exit_agent with text=code and items=[]. "
        "At root, search_tool_outputs for the code and tool_name='mock-sub-tool' "
        "with skill_path='/mock-skill'. Read the matching output ID with "
        "read_tool_output and cursor=null to read the complete result. "
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
        "exit_agent",
        "search_tool_outputs",
        "read_tool_output",
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
            for call in calls_by_name["exit_agent"]
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
            for call in calls_by_name["search_tool_outputs"]
            if call.get_arguments().get("skill_path") == "/mock-skill"
            and calls.index(call) > calls.index(exit_call)
            and code.casefold() in str(call.get_arguments().get("query", "")).casefold()
            and call.get_arguments().get("tool_name") == "mock-sub-tool"
        ),
        None,
    )
    assert search_call is not None, (
        "MODEL: no search used the absolute mock-skill path: "
        f"{[call.get_arguments() for call in calls_by_name['search_tool_outputs']]!r}"
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
        for call in calls_by_name["read_tool_output"]
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
