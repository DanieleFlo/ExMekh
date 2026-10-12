"""Fail closed on malformed wire data, missing receipts and unsupported providers."""

import json
from copy import deepcopy
from dataclasses import replace
from typing import Annotated, NotRequired, TypedDict

import httpx2 as httpx
import openai
import pytest
from pydantic import BaseModel

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.cache.user_message_codec import deserialize_user_message
from exmekh_lib.core.file_inspection.interactive import FileInteractiveCalls
from exmekh_lib.core.file_inspection.runtime import FileToolRuntime
from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall
from exmekh_lib.core.models.interactive import (
    InteractiveDefinition,
    InteractiveProtocolError,
)
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.providers import interactive_openai, interactive_transport
from exmekh_lib.core.providers.interactive_contract import InteractiveContract
from exmekh_lib.core.providers.interactive_openai import InteractiveResponseOpenAI
from exmekh_lib.core.providers.interactive_transport import InteractiveTransport
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.interfaces.provider import ProviderRequestRejectedError
from exmekh_lib.providers.local import LocalProvider
from exmekh_lib.utils.type_hints import typed_dict_required_keys
from tests.interactive_support import (
    CAPABILITIES,
    ScriptedServer,
    edit_arguments,
    pause,
    response,
)
from tests.live.model_support import CacheTestTokenizer


def contract():
    return InteractiveContract(
        {"edit-file": {"behavior": "edit_file.v2", "fields": {"document": "edits"}}}
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("exchange_id", ""),
        ("call_id", 1),
        ("tool_name", None),
        ("request", []),
        ("request", {"kind": "wrong"}),
        ("request", {"kind": "file_snapshot", "edit_index": 1, "values": {"path": ""}}),
    ],
)
def test_invalid_pause_metadata(field, value):
    event = pause()
    event[field] = value
    with pytest.raises(InteractiveProtocolError):
        contract().record_pause(event)


def test_no_implicit_acceptance_and_no_exchange_reuse():
    state = contract()
    with pytest.raises(InteractiveProtocolError):
        _ = state.continuation_id
    with pytest.raises(InteractiveProtocolError):
        state.record_acceptance([])
    state.record_pause(pause())
    with pytest.raises(InteractiveProtocolError, match="exchange"):
        state.record_pause(pause(edit_index=2))
    with pytest.raises(InteractiveProtocolError, match="omitted"):
        state.validate_completed_calls([])


def test_duplicate_exchange_does_not_change_authorized_paths_or_indices():
    state = contract()
    state.record_pause(pause("first.txt"))
    with pytest.raises(InteractiveProtocolError, match="exchange"):
        state.record_pause(pause("unapproved.txt", edit_index=2))
    state.record_pause(pause("second.txt", edit_index=2, exchange="second-exchange"))
    state.validate_completed_calls(
        [
            ToolCall(
                "edit",
                FunctionCall(
                    "edit-file",
                    {
                        "edits": edit_arguments("first.txt", "<all/>", "A")["edits"]
                        + edit_arguments("second.txt", "<all/>", "B")["edits"]
                    },
                ),
            )
        ]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("call_id", None),
        ("call_id", "unknown"),
        ("tool_name", "other"),
        ("values", {"path": "other.txt"}),
        ("edit_index", True),
        ("edit_index", 2),
        ("exchange_id", "other"),
    ],
)
def test_terminal_rejection_must_match_the_exact_exchange(field, value):
    state = contract()
    event = pause()
    state.record_pause(event)
    state.record_reply(event, {"status": "reject"})
    rejected = {
        "call_id": "edit",
        "tool_name": "edit-file",
        "values": {"path": "notes.txt"},
        "edit_index": 1,
        "exchange_id": "exchange-edit",
    }
    rejected[field] = value
    with pytest.raises(InteractiveProtocolError):
        state.validate_interruption(rejected)


def test_unsolicited_rejection_and_post_hook_arguments_cannot_bypass_authorization():
    state = contract()
    state.record_pause(pause())
    with pytest.raises(InteractiveProtocolError, match="rejected exchange"):
        state.validate_interruption(
            {
                "call_id": "edit",
                "tool_name": "edit-file",
                "values": {"path": "notes.txt"},
                "edit_index": 1,
                "exchange_id": "exchange-edit",
            }
        )
    with pytest.raises(InteractiveProtocolError):
        state.validate_calls(
            [ToolCall("edit", FunctionCall("edit-file", {"extra": "invalid"}))]
        )


@pytest.mark.parametrize(
    "output",
    [
        None,
        [None],
        [{"type": "function_call", "name": "edit-file", "arguments": "{}"}],
        [
            {
                "type": "function_call",
                "name": "edit-file",
                "call_id": "edit",
                "arguments": None,
            }
        ],
        [
            {
                "type": "function_call",
                "name": "edit-file",
                "call_id": "edit",
                "arguments": "broken",
            }
        ],
        [
            {
                "type": "function_call",
                "name": "edit-file",
                "call_id": "edit",
                "arguments": "[]",
            }
        ],
    ],
)
def test_raw_output_errors_remain_protocol_errors(output):
    with pytest.raises(InteractiveProtocolError):
        contract().validate_raw_calls(output)


class Handler:
    def __init__(self, reply=None):
        self.reply = {"status": "create"} if reply is None else reply

    def begin(self):
        pass

    def prepare(self, event, definition):
        return self.reply

    def validate_completed(self, calls):
        pass

    def cancellation_requested(self):
        return False


def make_provider(
    tmp_path,
    monkeypatch,
    steps,
    *,
    api="responses",
    handler=None,
    scoped=True,
    streaming=False,
):
    def edit(edits: str):
        pytest.fail("Provider tests must never execute a tool")

    server = ScriptedServer(steps)
    monkeypatch.setattr(LocalProvider, "_build_client", lambda self: server.client)
    provider = LocalProvider(
        "model",
        api=api,
        base_url="http://test/v1",
        tokenizer=CacheTestTokenizer("errors", tmp_path / "cache"),
    )
    tool = Tool(
        edit,
        name="edit-file",
        background_enabled=False,
        interactive=InteractiveDefinition("edit_file.v2", {"document": "edits"}),
    )
    request = ProviderRequest(
        [SystemMessage("edit"), UserMessage("create")],
        tools=Tools([tool]),
        stream=streaming,
        extra={
            "agentic": {
                "agent_context_id": "errors",
                "allowed_tools": ["edit-file"] if scoped else [],
                "allowed_skills": [],
            }
        },
    )
    request.interactive_handler = Handler() if handler is None else handler
    return provider, server, request


def completed():
    return response(
        ("edit", "edit-file", edit_arguments("notes.txt", "<all/>", "NEW")),
        continuation="cont-edit",
    )


def terminal():
    return {
        "type": "interactive.cancelled",
        "version": 2,
        "continuation_id": "cont-edit",
    }


@pytest.mark.parametrize(
    "capabilities",
    [
        [],
        {"interactive": []},
        {"interactive": {**CAPABILITIES["interactive"], "limits": []}},
        {
            "interactive": {
                **CAPABILITIES["interactive"],
                "limits": {"snapshot_bytes": 0},
            }
        },
    ],
)
def test_malformed_capabilities_fail_before_inference(
    tmp_path, monkeypatch, capabilities
):
    provider, server, request = make_provider(tmp_path, monkeypatch, [])
    server.capabilities = capabilities
    try:
        with pytest.raises(InteractiveProtocolError):
            provider.call(request)
        assert not server.requests
    finally:
        provider.get_session().close()
        server.client.close()


@pytest.mark.parametrize("kind", ["chat", "structured", "missing_handler", "unscoped"])
def test_incompatible_requests_are_rejected_without_fallback(
    tmp_path, monkeypatch, kind
):
    provider, server, request = make_provider(
        tmp_path,
        monkeypatch,
        [response(text="ordinary")],
        api="chat_completions" if kind == "chat" else "responses",
        scoped=kind != "unscoped",
    )
    try:
        if kind == "unscoped":
            assert (
                InteractiveResponseOpenAI(server.client, "model")
                .call(request)
                .get_message()
                == "ordinary"
            )
            assert "interactive" not in server.requests[0]["agentic"]
        elif kind == "missing_handler":
            request.interactive_handler = None
            with pytest.raises(InteractiveProtocolError, match="handler"):
                provider.call(request)
        else:
            if kind == "structured":

                class Output(BaseModel):
                    text: str

                request = StructuredProviderRequest(
                    request.get_messages(),
                    Output,
                    tools=request.get_tools(),
                    extra=request.get_extra(),
                )
            with pytest.raises(ProviderRequestRejectedError):
                provider.call_structured(
                    request
                ) if kind == "structured" else provider.call(request)
    finally:
        provider.get_session().close()
        server.client.close()


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "accepted_empty",
        "deltas",
        "retry_exhausted",
        "conflict",
        "poll",
        "status_timeout",
        "budget",
        "cleanup_error",
        "nonstring_snapshot",
        "snapshot_limit",
        "usage_details",
        "usage_missing",
    ],
)
def test_transport_receipts_limits_and_retry_failure(tmp_path, monkeypatch, kind):
    accepted = {
        "type": "interactive.accepted",
        "version": 2,
        "continuation_id": "cont-edit",
    }
    status = {
        "type": "interactive.status",
        "version": 2,
        "continuation_id": "cont-edit",
        "status": "running",
    }
    handler = Handler()
    steps = [pause(), completed()]
    events = []
    expect_error = kind not in {
        "accepted_empty",
        "deltas",
        "poll",
        "snapshot_limit",
        "usage_missing",
    }
    if kind == "empty":
        steps = [
            httpx.Response(
                200,
                text="data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        ]
    elif kind == "accepted_empty":
        steps = [
            httpx.Response(
                200,
                text="data: " + json.dumps(accepted) + "\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            ),
            {
                "type": "interactive.status",
                "version": 2,
                "continuation_id": "cont-edit",
                "result": response(text="done", continuation="cont-edit"),
            },
        ]
    elif kind == "deltas":
        wire = "".join(
            "data: " + json.dumps(event) + "\n\n"
            for event in [
                accepted,
                {"type": "response.output_text.delta", "delta": "text"},
                {"type": "response.reasoning_text.delta", "delta": "reason"},
                {"type": "response.function_call_arguments.delta", "delta": "partial"},
                {"type": "interactive.result", "response": pause()},
            ]
        )
        steps = [
            httpx.Response(
                200, text=wire, headers={"content-type": "text/event-stream"}
            ),
            completed(),
        ]
    elif kind == "retry_exhausted":
        steps = [httpx.ConnectError("lost")] * 3
    elif kind == "conflict":
        steps = [httpx.Response(409, json={"error": {"message": "conflict"}})]
    elif kind in {"poll", "status_timeout"}:
        steps = [httpx.Response(200, json=status)]
        if kind == "poll":
            steps += [
                {
                    "type": "interactive.status",
                    "version": 2,
                    "continuation_id": "cont-edit",
                    "result": response(text="done", continuation="cont-edit"),
                }
            ]
        else:
            monkeypatch.setattr(
                interactive_openai,
                "FILE_EDITING",
                replace(FILE_EDITING, protocol_timeout_seconds=0.01),
            )
            steps += [status] * 2 + [terminal()]
    elif kind == "budget":
        monkeypatch.setattr(
            interactive_openai, "FILE_EDITING", replace(FILE_EDITING, max_exchanges=0)
        )
        steps = [pause(), terminal()]
    elif kind == "cleanup_error":
        steps = [
            pause(),
            response(continuation="cont-edit"),
            httpx.Response(
                200,
                text="data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            ),
        ]
    elif kind == "nonstring_snapshot":
        handler.reply = {"status": "provide", "text": False}
        steps = [pause(), terminal()]
    elif kind == "snapshot_limit":
        handler.reply = {
            "status": "provide",
            "text": "€€",
            "encoding": "utf-8",
            "line_count": 1,
        }
        steps = [
            pause(),
            {
                "type": "interactive.rejected",
                "version": 2,
                "continuation_id": "cont-edit",
                "call_id": "edit",
                "tool_name": "edit-file",
                "exchange_id": "exchange-edit",
                "edit_index": 1,
                "values": {"path": "notes.txt"},
                "output": [],
            },
        ]
    elif kind in {"usage_details", "usage_missing"}:
        final = completed()
        if kind == "usage_details":
            final["usage"]["input_tokens_details"] = None
            steps = [pause(), final, terminal()]
        else:
            final["usage"] = None
            steps = [pause(), final]
    provider, server, request = make_provider(
        tmp_path, monkeypatch, steps, handler=handler, streaming=True
    )
    if kind == "snapshot_limit":
        server.capabilities = deepcopy(CAPABILITIES)
        server.capabilities["interactive"]["limits"] = {"snapshot_bytes": 1}

    class Events(ProviderEventHandler):
        def handle(self, event):
            events.append(event)

    request.on_event = Events()
    try:
        if expect_error:
            with pytest.raises(
                (InteractiveProtocolError, httpx.HTTPError, RuntimeError, ValueError)
            ) as caught:
                provider.call(request)
            if kind == "cleanup_error":
                assert any(
                    "Cancellation was not confirmed" in note
                    for note in caught.value.__notes__
                )
        else:
            result = provider.call(request)
            if kind == "snapshot_limit":
                assert not result.get_tool_calls()
                assert (
                    server.requests[-1]["agentic"]["interactive"]["reply"]["code"]
                    == "snapshot_limit"
                )
            if kind == "deltas":
                assert [event.content for event in events if event.content] == [
                    "text",
                    "reason",
                ]
        assert not (tmp_path / "notes.txt").exists()
    finally:
        provider.get_session().close()
        server.client.close()


@pytest.mark.parametrize(
    "wire,content_type",
    [
        (b"[]", "application/json"),
        (b"\ndata: []\n\n", "text/event-stream"),
        (b"data: {}\r", "text/event-stream"),
        (b"data: {}\r\n\r\n", "text/event-stream"),
    ],
)
def test_raw_transport_objects_and_final_sse_lines_are_bounded_and_closed(
    wire, content_type
):
    server = ScriptedServer(
        [httpx.Response(200, content=wire, headers={"content-type": content_type})]
    )
    try:
        transport = InteractiveTransport(server.client)
        if b"[]" in wire:
            with pytest.raises(InteractiveProtocolError):
                list(transport.events({}, 1))
        else:
            list(transport.events({}, 1))
    finally:
        server.client.close()


@pytest.mark.parametrize(
    "wire,content_type",
    [
        (b"{", "application/json"),
        (b"\xff", "application/json"),
        (b"data: {\n\n", "text/event-stream"),
        (b"data: \xff\n\n", "text/event-stream"),
        (b"data: \xe2", "text/event-stream"),
    ],
)
def test_malformed_http_and_sse_are_protocol_errors_without_inference_retry(
    tmp_path, monkeypatch, wire, content_type
):
    malformed = httpx.Response(
        200, content=wire, headers={"content-type": content_type}
    )
    provider, server, request = make_provider(
        tmp_path, monkeypatch, [pause(), malformed, terminal()]
    )
    try:
        with pytest.raises(InteractiveProtocolError, match="Malformed"):
            provider.call(request)
        assert [
            body["agentic"]["interactive"]["operation"] for body in server.requests
        ] == ["configure", "resume", "cancel"]
        assert malformed.is_closed
    finally:
        provider.get_session().close()
        server.client.close()


@pytest.mark.parametrize("streaming", [False, True])
def test_scope_changed_at_a_pause_cancels_before_sending_a_snapshot(
    tmp_path, monkeypatch, streaming
):
    scope = ("node", "first-opening")

    def change_scope(body):
        nonlocal scope
        scope = ("node", "reopened-skill")
        return pause()

    handler = FileInteractiveCalls(
        FileToolRuntime(), tmp_path, lambda: (), lambda: False, lambda: scope
    )
    provider, server, request = make_provider(
        tmp_path,
        monkeypatch,
        [change_scope, terminal()],
        handler=handler,
        streaming=streaming,
    )
    server.streaming = streaming
    try:
        result = provider.call(request)
        assert not result.get_tool_calls()
        assert [
            body["agentic"]["interactive"]["operation"] for body in server.requests
        ] == ["configure", "cancel"]
        assert all(
            "reply" not in body["agentic"]["interactive"] for body in server.requests
        )
        assert not (tmp_path / "notes.txt").exists()
    finally:
        provider.get_session().close()
        server.client.close()


@pytest.mark.parametrize("kind", ["bytes", "timeout", "split_cr"])
def test_transport_budget_timeout_and_cr_fragmentation(monkeypatch, kind):
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            if kind == "timeout":
                raise httpx.ReadTimeout("timeout")
            if kind == "split_cr":
                yield b"data: {}\r"
                yield b"\n\r\n"
            else:
                yield b"{}"

    server = ScriptedServer(
        [
            httpx.Response(
                200,
                stream=Stream(),
                headers={
                    "content-type": "text/event-stream"
                    if kind == "split_cr"
                    else "application/json"
                },
            )
        ]
    )
    if kind == "bytes":
        monkeypatch.setattr(
            interactive_transport,
            "FILE_EDITING",
            replace(FILE_EDITING, max_response_bytes=1),
        )
    try:
        if kind == "split_cr":
            assert list(InteractiveTransport(server.client).events({}, 1)) == [{}]
        else:
            with pytest.raises(
                (InteractiveProtocolError, RuntimeError, openai.APIError)
            ):
                list(InteractiveTransport(server.client).events({}, 1))
    finally:
        server.client.close()


def test_cached_and_stream_metadata_are_not_coerced():
    with pytest.raises(ValueError, match="object"):
        deserialize_user_message({"interactive": []})
    with pytest.raises(TypeError):
        ProviderStreamEvent(ProviderStreamEventType.INTERACTIVE)
    with pytest.raises(ValueError):
        ProviderStreamEvent(ProviderStreamEventType.END, interactive={})

    class Fields(TypedDict):
        value: Annotated[NotRequired[str], "metadata"]

    assert (
        typed_dict_required_keys(
            Fields, {"value": Annotated[NotRequired[str], "metadata"]}
        )
        == frozenset()
    )


def test_expired_budget_stops_before_any_request(tmp_path, monkeypatch):
    monkeypatch.setattr(
        interactive_openai,
        "FILE_EDITING",
        replace(FILE_EDITING, protocol_timeout_seconds=0),
    )
    provider, server, request = make_provider(tmp_path, monkeypatch, [])
    try:
        with pytest.raises(InteractiveProtocolError, match="logical-call timeout"):
            provider.call(request)
        assert not server.requests
    finally:
        provider.get_session().close()
        server.client.close()


def test_scope_invalidated_while_polling_does_not_start_another_inference(
    tmp_path, monkeypatch
):
    class Request(ProviderRequest):
        invalidated = False

        def get_extra(self):
            return {"agentic": None} if self.invalidated else super().get_extra()

    def pending_receipt(body):
        request.invalidated = True
        return {
            "type": "interactive.status",
            "version": 2,
            "continuation_id": "cont-edit",
            "status": "running",
        }

    provider, server, original = make_provider(
        tmp_path, monkeypatch, [pending_receipt, terminal()]
    )
    request = Request(
        original.get_messages(), tools=original.get_tools(), extra=original.get_extra()
    )
    request.interactive_handler = Handler()
    try:
        with pytest.raises(InteractiveProtocolError, match="scope"):
            provider.call(request)
        assert [
            body["agentic"]["interactive"]["operation"] for body in server.requests
        ] == ["configure", "cancel"]
    finally:
        provider.get_session().close()
        server.client.close()
