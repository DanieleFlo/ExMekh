from __future__ import annotations

import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
from dotenv import load_dotenv
from openai import OpenAI

import qs
from qs.core.agents.node import Node
from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.reasoning_effort import ReasoningEffortLMH
from qs.core.models.tool_message import ToolMessage
from qs.core.tokenizers.sequential_tokenizer import SequentialTokenizer
from qs.core.tool_calls import ToolCallReader, ToolCallSearch
from qs.providers.local import LocalProvider
from qs.utils.file import read_json_from_path
from tests.event_handler import as_agent_handler

pytestmark = pytest.mark.live

_TRANSIENT_TOOL_NAMES = {
    "exit_agent",
    "get_execution_history",
    "search_saved_tool_outputs",
    "read_saved_tool_output",
}


class CacheTestTokenizer(SequentialTokenizer):
    """Exercise real session/ID storage without downloading a vocabulary."""

    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

    def _load_vocabulary_tokens(self) -> list[str]:
        return [f"token{index:02d}" for index in range(32)]


class RecordingLocalProvider(LocalProvider):
    """Observe real model responses without replacing their content or thinking."""

    def __init__(self, directory: Path) -> None:
        self.responses: list[ProviderResponse] = []
        self.requests: list[ProviderRequest] = []
        super().__init__(
            model=os.getenv("QS_LIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S"),
            reasoning_effort=ReasoningEffortLMH.LOW,
            api="responses",
            tokenizer=CacheTestTokenizer("live-tool-cache", directory),
        )

    def _build_client(self) -> OpenAI:
        client = super()._build_client()
        return client.with_options(timeout=60.0, max_retries=0)

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
        capabilities_paths=[Path(qs.__file__).parent / "capabilities"],
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
    node: Node, responses: list[ProviderResponse]
) -> list[ToolCallRecord]:
    expected = {
        call.id: (call, response.thinking.thoughts if response.thinking else None)
        for response in responses
        for call in response.get_tool_calls() or []
        if call.get_name() not in _TRANSIENT_TOOL_NAMES
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
    node.run(
        f"Call mock-tool exactly once with text='{token}', then reply briefly without other tools."
    )

    records = verify_saved_calls(node, live_provider.responses)
    assert len(records) == 1, "MODEL: expected exactly one mock-tool call"
    record = records[0]
    assert record.name == "mock-tool"
    assert record.result == f"Mock tool result: {token}"
    assert record.skill_path == "/"
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
    assert page["records"] == [record.to_dict()]


@pytest.mark.parametrize("streaming", [False, True], ids=["non-streaming", "streaming"])
def test_live_nested_tools_save_path_and_each_turn_thinking(
    live_provider, tmp_path, streaming
):
    node = initialize_node(live_provider, tmp_path, streaming=streaming)
    token = "LIVE-CACHE-NESTED"
    node.run(
        "Follow this sequence without skipping steps: open mock-skill, then nested-skill; "
        f"call nested-sub-tool exactly once with text='{token}'; "
        "call exit_agent with text='nested done' "
        "and items=[]; in the parent skill call exit_agent with text='parent done' "
        "and items=[]. Finally, reply briefly without other tools."
    )

    records = verify_saved_calls(node, live_provider.responses)
    assert len(records) == 3, (
        "MODEL: expected one regular tool and two saved skill outputs"
    )
    regular = [record for record in records if record.name == "nested-sub-tool"]
    skills = [record for record in records if record.source == "DeepSkill"]
    assert len(regular) == 1 and len(skills) == 2
    assert regular[0].skill_path == "/mock-skill/nested-skill"
    assert token in regular[0].result
    skill_records = {record.name: record for record in skills}
    assert skill_records["mock-skill"].skill_path == "/mock-skill"
    assert skill_records["nested-skill"].skill_path == "/mock-skill/nested-skill"
    exit_calls = [
        call
        for response in live_provider.responses
        for call in response.get_tool_calls() or []
        if call.get_name() == "exit_agent"
    ]
    assert len(exit_calls) == 2
    for skill_name, exit_call in zip(
        ("nested-skill", "mock-skill"), exit_calls, strict=True
    ):
        arguments = exit_call.get_arguments()
        assert isinstance(arguments.get("text"), str) and arguments["text"].strip()
        assert arguments.get("items") == []
        assert skill_records[skill_name].result == arguments
    assert node.depth == 0


def test_live_agent_reads_saved_sub_tool_output_and_matches_exit_code(
    live_provider,
    tmp_path,
):
    node = initialize_node(live_provider, tmp_path)
    response = node.run(
        "Follow this sequence exactly. Enter mock-skill. Inside the skill, invent "
        "a unique alphanumeric code containing 6 to 12 letters or digits. Call "
        "mock-sub-tool exactly once with that code as text. Then call exit_agent "
        "with the exact same code as text and items=[]. Back at the root, "
        "call search_saved_tool_outputs exactly once with query set to the code "
        "and tool_name='mock-sub-tool'. Take the id from the returned XML record "
        "and pass it to read_saved_tool_output exactly once. Compare the code in "
        "the complete saved output with the code returned by exit_agent. If they "
        "are identical, answer exactly 'MATCH: <code>', replacing <code> with the "
        "actual code. Otherwise answer exactly 'MISMATCH'. Do not call any other "
        "tools."
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
        "search_saved_tool_outputs",
        "read_saved_tool_output",
    )
    calls_by_name = {
        name: [call for call in calls if call.get_name() == name]
        for name in expected_call_names
    }
    actual_call_names = [call.get_name() for call in calls]
    assert actual_call_names == list(expected_call_names), (
        "MODEL: expected exactly one ordered call for each step; calls="
        f"{actual_call_names}"
    )

    sub_tool_call = calls_by_name["mock-sub-tool"][0]
    code = sub_tool_call.get_arguments().get("text")
    assert isinstance(code, str) and 6 <= len(code) <= 12 and code.isalnum(), (
        f"MODEL: expected a 6-12 character alphanumeric code, got {code!r}"
    )
    exit_arguments = calls_by_name["exit_agent"][0].get_arguments()
    exit_text = exit_arguments.get("text")
    assert isinstance(exit_text, str) and code in exit_text
    assert exit_arguments.get("items") == []

    records = saved_cache(node).get_all()
    sub_tool_records = [record for record in records if record.name == "mock-sub-tool"]
    assert len(sub_tool_records) == 1, (
        "CACHE: expected exactly one saved mock-sub-tool output"
    )
    sub_tool_record = sub_tool_records[0]
    assert sub_tool_record.args == {"text": code}
    assert sub_tool_record.result == f"Mock sub-tool result: {code}"

    skill_records = [record for record in records if record.name == "mock-skill"]
    assert len(skill_records) == 1
    assert skill_records[0].source == "DeepSkill"
    assert skill_records[0].result == exit_arguments

    search_call = calls_by_name["search_saved_tool_outputs"][0]
    search_arguments = search_call.get_arguments()
    assert code in search_arguments.get("query", "")
    assert search_arguments.get("tool_name") == "mock-sub-tool"
    search_xml = ET.fromstring(get_tool_response(live_provider, search_call.id))
    assert search_xml.findtext("total") == "1"
    assert search_xml.findtext("./records/record/name") == "mock-sub-tool"
    assert search_xml.findtext("./records/record/id") == sub_tool_record.id
    assert code in (search_xml.findtext("./records/record/preview") or "")

    read_call = calls_by_name["read_saved_tool_output"][0]
    assert read_call.get_arguments() == {"output_id": sub_tool_record.id}
    read_response = get_tool_response(live_provider, read_call.id)
    header_xml, saved_output = read_response.split("</header>\n", maxsplit=1)
    header = ET.fromstring(header_xml + "</header>")
    assert header.findtext("name") == "mock-sub-tool"
    assert header.findtext("type") == "Tool"
    assert header.findtext("id") == sub_tool_record.id
    assert header.findtext("./args/text") == code
    assert saved_output == f"Mock sub-tool result: {code}"
    answer = response.get_message()
    assert re.search(r"\bMATCH\b", answer, flags=re.IGNORECASE)
    assert not re.search(r"\bMISMATCH\b", answer, flags=re.IGNORECASE)
    assert code in answer


def test_live_cache_isolates_same_named_nodes_and_retains_previous_runs(
    live_provider, tmp_path
):
    first = initialize_node(live_provider, tmp_path)
    second = initialize_node(live_provider, tmp_path)
    observed: dict[str, list[ProviderResponse]] = {"first": [], "second": []}
    for label, node, token in [
        ("first", first, "LIVE-FIRST-ONE"),
        ("second", second, "LIVE-SECOND"),
        ("first", first, "LIVE-FIRST-TWO"),
    ]:
        start = len(live_provider.responses)
        node.run(
            f"Call mock-tool exactly once with text='{token}', then reply without other tools."
        )
        observed[label].extend(live_provider.responses[start:])

    first_records = verify_saved_calls(first, observed["first"])
    second_records = verify_saved_calls(second, observed["second"])
    assert len(first_records) == 2 and len(second_records) == 1
    assert {record.args["text"] for record in first_records} == {
        "LIVE-FIRST-ONE",
        "LIVE-FIRST-TWO",
    }
    assert second_records[0].args == {"text": "LIVE-SECOND"}
    assert saved_cache(first).node_id != saved_cache(second).node_id
    assert {record.id for record in first_records}.isdisjoint(
        record.id for record in second_records
    )
    assert ToolCallSearch(saved_cache(first)).search(query="LIVE-SECOND")["total"] == 0


def test_live_reported_capability_check(live_provider, tmp_path):
    # Keep this cache smoke test focused; the deep-thinking skill has its own live suite.
    live_provider._reasoning_effort = None
    node = initialize_node(live_provider, tmp_path, streaming=True)
    response = node.run(
        "Check these three capabilities only: mock-skill, mock-tool, and the "
        "mock-sub-tool inside mock-skill. Call each once, finish the skill, "
        "then summarize the results without opening other skills or tools."
    )
    names = [
        call.get_name()
        for turn in live_provider.responses
        for call in turn.get_tool_calls() or []
    ]
    assert "mock-skill" in names and "mock-tool" in names, (
        f"MODEL: did not execute the requested capability check: {names}; "
        f"answer={response.get_message()}"
    )
    assert any(name in names for name in ("mock-sub-tool", "nested-sub-tool"))
    verify_saved_calls(node, live_provider.responses)
    assert node.depth == 0
    assert response.get_message()


@pytest.mark.parametrize("streaming", [False, True], ids=["non-streaming", "streaming"])
def test_live_exit_uses_instructions_available_only_in_markdown(
    live_provider, tmp_path, streaming
):
    root = tmp_path / "capabilities"
    shutil.copytree(Path(qs.__file__).parent / "capabilities", root)
    token = "EXIT-MARKDOWN-7319"
    (root / "exit-tool" / "finish" / "instructions.md").write_text(
        f"Finish the skill by calling {{{{!exit_agent}}}} with text='{token}' and items=[]. "
        "The token is the required result for this test.",
        encoding="utf-8",
    )
    node = Node([root], name_hds="main", persistent_history=False)
    node.initialize(
        live_provider,
        work_root=tmp_path,
        on_stream=as_agent_handler((lambda event: None) if streaming else None),
    )
    response = node.run(
        "Open mock-skill exactly once. Follow its exit instructions to finish it. Then reply briefly without other tools."
    )
    skills = [
        record
        for record in verify_saved_calls(node, live_provider.responses)
        if record.name == "mock-skill"
    ]
    assert len(skills) == 1
    assert skills[0].source == "DeepSkill"
    assert skills[0].result == {"text": token, "items": []}
    assert node.depth == 0
    assert response.get_message()
    assert (
        "exit_agent"
        not in live_provider.requests[0].get_extra()["agentic"]["allowed_tools"]
    )
    assert (
        "exit_agent"
        not in live_provider.requests[-1].get_extra()["agentic"]["allowed_tools"]
    )
    for request in live_provider.requests:
        assert "finish" not in request.get_tools().list_tools()
    skill_messages = [
        message.get_message()
        for request in live_provider.requests
        for message in request.get_messages()
        if message.get_role_name() == "tool" and "<exit-tool" in message.get_message()
    ]
    assert skill_messages
    assert all(token in message and "{{!" not in message for message in skill_messages)


def test_live_exit_items_position_and_error_retry(live_provider, tmp_path):
    node = initialize_node(live_provider, tmp_path)
    token = "LIVE-EXIT-ITEM-4286"
    response = node.run(
        "Follow this exact sequence to verify error handling. "
        "Open mock-skill; inside it call mock-sub-tool exactly once with "
        f"text='{token}'. Then call exit_agent with text='test' and "
        "items=['missing-test-id']: this call must produce an error. "
        "After the error, use get_execution_history to read the call ID and "
        "saved-output-id of mock-sub-tool. Call exit_agent again with "
        "text='Before {{<saved-output-id>}} After', replacing the placeholder "
        "with the actual saved-output-id, and items=['<call-id>'], replacing "
        "with the actual call ID of the same mock-sub-tool. Do not include "
        "angle brackets in the values. At the root, reply briefly."
    )

    calls = [
        call for turn in live_provider.responses for call in turn.get_tool_calls() or []
    ]
    exit_calls = [call for call in calls if call.get_name() == "exit_agent"]
    assert len(exit_calls) == 2, "MODEL: expected invalid exit followed by correction"
    assert exit_calls[0].get_arguments()["items"] == ["missing-test-id"]
    assert "Correct the error and call exit_agent again" in get_tool_response(
        live_provider, exit_calls[0].id
    )
    records = verify_saved_calls(node, live_provider.responses)
    tool_record = next(record for record in records if record.name == "mock-sub-tool")
    skill_record = next(record for record in records if record.name == "mock-skill")
    assert exit_calls[1].get_arguments()["items"] == [tool_record.tool_call_id]
    assert f"{{{{{tool_record.id}}}}}" in exit_calls[1].get_arguments()["text"]
    content = skill_record.result["text"]
    assert content.startswith("Before <tool-output")
    assert content.endswith("</tool-output> After")
    assert token in content
    assert node.depth == 0
    assert response.get_message()
