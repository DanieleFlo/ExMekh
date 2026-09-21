from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

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
from qs.core.tokenizers.sequential_tokenizer import SequentialTokenizer
from qs.core.tool_calls import ToolCallReader, ToolCallSearch
from qs.providers.local import LocalProvider
from qs.utils.file import read_json_from_path

pytestmark = pytest.mark.live


class CacheTestTokenizer(SequentialTokenizer):
    """Exercise real session/ID storage without downloading a vocabulary."""

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
        on_stream=(lambda event: None) if streaming else None,
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
        if call.get_name() not in {"mock-skill", "nested-skill"}
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


@pytest.mark.parametrize("streaming", [False, True], ids=["non-streaming", "streaming"])
def test_live_regular_tool_saves_output_and_calling_thinking(
    live_provider, tmp_path, streaming
):
    node = initialize_node(live_provider, tmp_path, streaming=streaming)
    token = "LIVE-CACHE-ROOT"
    node.run(
        f"Chiama mock-tool esattamente una volta con text='{token}', poi rispondi brevemente senza altri tool."
    )

    records = verify_saved_calls(node, live_provider.responses)
    assert len(records) == 1, "MODEL: expected exactly one mock-tool call"
    record = records[0]
    assert record.name == "mock-tool"
    assert record.result == f"Mock tool result: {token}"
    assert record.skill_path == "/"
    assert isinstance(record.thinking, str) and record.thinking.strip(), (
        "PROVIDER: this test requires thinking on the tool-calling response; "
        "configure a model/backend that exposes it"
    )
    page = ToolCallSearch(saved_cache(node)).search(query=token, skill_path="/")
    assert page["records"] == [record.to_dict()]


@pytest.mark.parametrize("streaming", [False, True], ids=["non-streaming", "streaming"])
def test_live_nested_tools_save_path_and_each_turn_thinking(
    live_provider, tmp_path, streaming
):
    node = initialize_node(live_provider, tmp_path, streaming=streaming)
    token = "LIVE-CACHE-NESTED"
    node.run(
        "Esegui questa sequenza senza saltare passaggi: apri mock-skill, poi nested-skill; "
        f"chiama nested-sub-tool una sola volta con text='{token}'; "
        "chiama exit_agent con text='nested done' "
        "e items=['nested']; nella skill padre chiama exit_agent con text='parent done' "
        "e items=['parent']. Infine rispondi brevemente senza altri tool."
    )

    records = verify_saved_calls(node, live_provider.responses)
    assert len(records) == 3, "MODEL: expected one regular tool and two exits"
    regular = [record for record in records if record.name == "nested-sub-tool"]
    exits = [record for record in records if record.name == "exit_agent"]
    assert len(regular) == 1 and len(exits) == 2
    assert regular[0].skill_path == "/mock-skill/nested-skill"
    assert token in regular[0].result
    assert {record.skill_path for record in exits} == {
        "/mock-skill",
        "/mock-skill/nested-skill",
    }
    assert all(record.result == record.args for record in exits)
    assert node.depth == 0
    assert all(record.name not in {"mock-skill", "nested-skill"} for record in records)


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
            f"Chiama mock-tool esattamente una volta con text='{token}', poi rispondi senza altri tool."
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
    # Match the CLI configuration and the reported prompt, not a forced tool script.
    live_provider._reasoning_effort = None
    node = initialize_node(live_provider, tmp_path, streaming=True)
    response = node.run(
        "testa una skill e un tool, anche il tool nella skill, fai un check completo delle capacità"
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
        f"Concludi la skill chiamando {{{{!exit_agent}}}} con text='{token}' e items=['verified']. "
        "Il token è il risultato richiesto per questa prova.",
        encoding="utf-8",
    )
    node = Node([root], name_hds="main", persistent_history=False)
    node.initialize(
        live_provider,
        work_root=tmp_path,
        on_stream=(lambda event: None) if streaming else None,
    )
    response = node.run(
        "Apri mock-skill una sola volta. Per concluderla segui le istruzioni di uscita fornite dalla skill. Poi rispondi brevemente senza altri tool."
    )
    exits = [
        record
        for record in verify_saved_calls(node, live_provider.responses)
        if record.name == "exit_agent"
    ]
    assert len(exits) == 1
    assert exits[0].result == {"text": token, "items": ["verified"]}
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
