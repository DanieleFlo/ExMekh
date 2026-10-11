"""Negotiation, frozen fields, raw transport and idempotent replay with SDK mocks."""

import json
from copy import deepcopy
from dataclasses import replace
from time import sleep

import httpx2 as httpx
import pytest

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.cache.user_message_codec import (
    deserialize_user_message,
    serialize_user_message,
)
from exmekh_lib.core.models.interactive import (
    InteractiveDefinition,
    InteractiveProtocolError,
    InterruptedMessage,
)
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.providers import interactive_openai
from exmekh_lib.core.providers.interactive_contract import InteractiveContract
from exmekh_lib.core.providers.to_openai_response_mapper import ToOpenAIResponseMapper
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.providers.local import LocalProvider
from tests.interactive_support import (
    CAPABILITIES,
    ScriptedServer,
    enter,
    exit_skill,
    make_sdk_node,
    pause,
    response,
)
from tests.live.model_support import CacheTestTokenizer


def cancelled():
    return {
        "type": "interactive.cancelled",
        "version": 1,
        "continuation_id": "cont-edit",
        "code": "cancelled",
        "message": "cancelled",
        "output": [],
    }


@pytest.mark.parametrize("definition_kind", ["tool", "decorator"])
def test_invalid_interactive_metadata_is_rejected_at_definition_time(definition_kind):
    with pytest.raises(TypeError, match="interactive must be an InteractiveDefinition"):
        if definition_kind == "tool":
            Tool(lambda: None, name="edit", background_enabled=False, interactive={})
        else:
            deep_tool(background_enabled=False, interactive={})


@pytest.mark.parametrize("definition_kind", ["tool", "decorator"])
def test_interactive_background_policy_is_rejected_at_definition_time(definition_kind):
    interactive = InteractiveDefinition(
        "edit_file.v1",
        {"resource": "path", "selection": "old_text", "replacement": "new_text"},
        ("path",),
        {},
    )
    with pytest.raises(
        ValueError, match="Interactive tools must execute synchronously"
    ):
        if definition_kind == "tool":
            Tool(lambda: None, name="edit", interactive=interactive)
        else:
            deep_tool(interactive=interactive)


@pytest.mark.parametrize(
    "attack",
    [
        "version",
        "scope",
        "behavior",
        "missing_field",
        "extra_field",
        "wrong_type",
        "changed_id",
        "changed_path",
        "duplicate_call",
        "coercion",
        "duplicate_key",
        "no_exchange",
        "no_acceptance",
    ],
)
def test_protocol_errors_cannot_dispatch_or_create_files(tmp_path, monkeypatch, attack):
    event = pause()
    steps = [enter()]
    if attack == "version":
        event["version"] = True
    elif attack == "scope":
        event["tool_name"] = "forbidden"
    elif attack == "behavior":
        event["behavior"] = "unknown.v1"
    elif attack == "missing_field":
        event["request"]["values"] = {}
    elif attack == "extra_field":
        event["request"]["values"]["old_text"] = "early"
    elif attack == "wrong_type":
        event["request"]["values"]["path"] = False
    call = (
        "edit",
        "edit-file",
        {
            "path": "notes.txt",
            "old_text": FILE_EDITING.all_marker,
            "new_text": "unsafe",
        },
    )
    final = response(call, continuation="cont-edit")
    if attack == "changed_id":
        final["interactive"]["continuation_id"] = "another-context"
    elif attack == "changed_path":
        final = response(
            ("edit", "edit-file", {**call[2], "path": "other.txt"}),
            continuation="cont-edit",
        )
    elif attack == "duplicate_call":
        final = response(call, call, continuation="cont-edit")
    elif attack == "coercion":
        final["output"][0]["arguments"] = json.dumps({**call[2], "old_text": False})
    elif attack == "duplicate_key":
        final["output"][0]["arguments"] = (
            json.dumps(call[2])[:-1] + ',"new_text":"other"}'
        )
    elif attack == "no_acceptance":
        final.pop("interactive")
    if attack == "no_exchange":
        steps += [final, cancelled()]
    else:
        steps += [event]
        if attack not in {
            "version",
            "scope",
            "behavior",
            "missing_field",
            "extra_field",
            "wrong_type",
        }:
            steps += [final]
        steps += [cancelled()]
    node, server, provider = make_sdk_node(tmp_path, monkeypatch, steps)
    try:
        with pytest.raises(InteractiveProtocolError):
            node.run("attack")
        assert not (tmp_path / "notes.txt").exists()
        assert not (tmp_path / "other.txt").exists()
    finally:
        provider.get_session().close()
        server.client.close()


def test_rejected_uninspected_call_is_replayed_without_orphan_outputs_and_recovers_once(
    tmp_path, monkeypatch
):
    (tmp_path / "notes.txt").write_text("KEEP")

    def reject(body):
        reply = body["agentic"]["interactive"]["reply"]
        assert reply["status"] == "reject" and reply["code"] == "uninspected"
        assert "text" not in reply
        return {
            "type": "interactive.rejected",
            "version": 1,
            "continuation_id": "cont-edit",
            "call_id": "edit",
            "tool_name": "edit-file",
            "values": {"path": "notes.txt"},
            "code": "uninspected",
            "message": "inspect first",
            "output": [],
        }

    def after_inspection(body):
        items = body["input"]
        interrupted = next(
            item for item in items if item.get("type") == "interactive.rejected"
        )
        assert interrupted["values"] == {"path": "notes.txt"}
        assert not any(
            item.get("call_id") == "edit"
            and item.get("type") in {"function_call", "function_call_output"}
            for item in items
        )
        inspection = [
            item
            for item in items
            if item.get("type") == "function_call"
            and item.get("name") == "inspect-file"
        ]
        assert len(inspection) == 1
        return exit_skill()

    node, server, provider = make_sdk_node(
        tmp_path,
        monkeypatch,
        [enter(), pause(), reject, after_inspection, response(text="done")],
    )
    try:
        result = node.run("edit without reading")
        assert (tmp_path / "notes.txt").read_text() == "KEEP"
        record = next(
            message
            for message in result.skill_executions[0].history
            if isinstance(message, InterruptedMessage)
        )
        encoded = serialize_user_message(record)
        assert ToOpenAIResponseMapper.map_message(
            deserialize_user_message(encoded)
        ) == [record.record]
        assert json.loads(record.message_dump()["interactive"])["call_id"] == "edit"
    finally:
        provider.get_session().close()
        server.client.close()


@pytest.mark.parametrize("disconnect", ["connect", "stream"])
def test_disconnect_retries_exact_resume_payload_without_duplicating_the_edit(
    tmp_path, monkeypatch, disconnect
):
    resume_bodies = []
    closed = []

    class BrokenStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"event: interactive.result\ndata: {"
            raise httpx.ReadError("connection lost during SSE")

        def close(self):
            closed.append(True)

    def lost_reply(body):
        resume_bodies.append(body)
        if disconnect == "connect":
            return httpx.ConnectError("response lost")
        return httpx.Response(
            200,
            stream=BrokenStream(),
            headers={"content-type": "text/event-stream"},
        )

    def receipt(body):
        resume_bodies.append(body)
        assert body == resume_bodies[0]
        return response(
            (
                "edit",
                "edit-file",
                {
                    "path": "notes.txt",
                    "old_text": FILE_EDITING.all_marker,
                    "new_text": "ONCE",
                },
            ),
            continuation="cont-edit",
        )

    node, server, provider = make_sdk_node(
        tmp_path,
        monkeypatch,
        [enter(), pause(), lost_reply, receipt, exit_skill(), response(text="done")],
        streaming=disconnect == "stream",
    )
    try:
        node.run("create")
        assert (tmp_path / "notes.txt").read_text() == "ONCE"
        assert len(resume_bodies) == 2
        assert closed == ([True] if disconnect == "stream" else [])
        assert (
            len(
                [
                    body
                    for body in server.requests
                    if body.get("agentic", {}).get("interactive", {}).get("operation")
                    == "configure"
                ]
            )
            == 2
        )
    finally:
        provider.get_session().close()
        server.client.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("model_supported", False),
        ("apis", ["chat_completions"]),
        ("model_syntax", "unsupported"),
        ("behaviors", []),
        ("syntaxes", None),
        ("apis", "responses"),
    ],
)
def test_negotiation_requires_actual_model_and_behavior_support(
    tmp_path, monkeypatch, field, value
):
    node, server, provider = make_sdk_node(tmp_path, monkeypatch, [enter()])
    server.capabilities = deepcopy(CAPABILITIES)
    server.capabilities["interactive"][field] = value
    try:
        with pytest.raises(
            InteractiveProtocolError, match="does not support|Malformed"
        ):
            node.run("edit")
        assert len(server.requests) == 1
    finally:
        provider.get_session().close()
        server.client.close()


def test_generic_roles_and_multiple_priority_fields_are_frozen_without_path_assumptions(
    tmp_path, monkeypatch
):
    def edit(target: str, format: bool, replacement: str, selection: str):
        return None

    tool = Tool(
        edit,
        name="renamed-edit",
        background_enabled=False,
        interactive=InteractiveDefinition(
            "edit_file.v1",
            {
                "resource": "target",
                "selection": "selection",
                "replacement": "replacement",
            },
            ("target", "format"),
            {
                "line": FILE_EDITING.line_family,
                "all": FILE_EDITING.all_marker,
                "delete": FILE_EDITING.delete_marker,
            },
        ),
    )
    event = pause(
        name="renamed-edit", fields={"target": 'quoted"\\name', "format": True}
    )

    def resume(body):
        assert set(body) == {"agentic"}
        assert "operation" not in body["agentic"]
        return response(
            (
                "edit",
                "renamed-edit",
                {
                    "target": 'quoted"\\name',
                    "format": True,
                    "selection": FILE_EDITING.all_marker,
                    "replacement": "new",
                },
            ),
            continuation="cont-edit",
        )

    server = ScriptedServer([event, resume])
    monkeypatch.setattr(LocalProvider, "_build_client", lambda self: server.client)
    provider = LocalProvider(
        "model",
        api="responses",
        base_url="http://test/v1",
        tokenizer=CacheTestTokenizer("roles", tmp_path / "cache"),
    )

    class Handler:
        def begin(self):
            pass

        def prepare(self, event, definition):
            assert definition["first_fields"] == ["target", "format"]
            return {"status": "create"}

        def validate_completed(self, calls):
            pass

        def cancellation_requested(self):
            return False

    request = ProviderRequest(
        [SystemMessage("edit"), UserMessage("new")],
        tools=Tools([tool]),
        extra={
            "agentic": {
                "agent_context_id": "scope",
                "allowed_tools": ["renamed-edit"],
                "allowed_skills": [],
                "operation": "return",
                "skill_call_id": "opening",
            }
        },
    )
    request.interactive_handler = Handler()
    try:
        result = provider.call(request)
        assert result.get_tool_calls()[0].get_arguments()["format"] is True
        assert server.requests[0]["agentic"]["operation"] == "return"
        assert result.usage.total_tokens == 30
    finally:
        provider.get_session().close()
        server.client.close()


def test_consumed_continuation_recovers_a_terminal_status_receipt(
    tmp_path, monkeypatch
):
    final = response(
        (
            "edit",
            "edit-file",
            {
                "path": "notes.txt",
                "old_text": FILE_EDITING.all_marker,
                "new_text": "ONCE",
            },
        ),
        continuation="cont-edit",
    )

    def conflict(body):
        return httpx.Response(409, json={"error": {"message": "already consumed"}})

    def status(body):
        control = body["agentic"]["interactive"]
        assert control["operation"] == "status" and "request_id" not in control
        return {
            "type": "interactive.status",
            "version": 1,
            "continuation_id": "cont-edit",
            "status": "completed",
            "result": final,
        }

    node, server, provider = make_sdk_node(
        tmp_path,
        monkeypatch,
        [enter(), pause(), conflict, status, exit_skill(), response(text="done")],
    )
    try:
        node.run("create")
        assert (tmp_path / "notes.txt").read_text() == "ONCE"
    finally:
        provider.get_session().close()
        server.client.close()


def test_steering_cancels_before_snapshot_delivery_and_prevents_dispatch(
    tmp_path, monkeypatch
):
    def pause_and_steer(body):
        node.steer("stop this edit")
        return pause()

    def cancel(body):
        assert body["agentic"]["interactive"]["operation"] == "cancel"
        assert "reply" not in body["agentic"]["interactive"]
        return cancelled()

    node, server, provider = make_sdk_node(
        tmp_path,
        monkeypatch,
        [enter(), pause_and_steer, cancel, exit_skill(), response(text="done")],
    )
    try:
        result = node.run("create")
        assert not (tmp_path / "notes.txt").exists()
        assert any(
            message["content"] == "stop this edit"
            for message in server.requests[-2]["input"]
            if message.get("role") == "user"
        )
        assert any(
            isinstance(message, InterruptedMessage)
            for message in result.skill_executions[0].history
        )
    finally:
        provider.get_session().close()
        server.client.close()


def test_steering_at_rejection_discards_earlier_complete_edits_and_recovery(
    tmp_path, monkeypatch
):
    (tmp_path / "notes.txt").write_text("KEEP")

    def reject_and_steer(body):
        assert body["agentic"]["interactive"]["reply"]["status"] == "reject"
        node.steer("stop all edits")
        return {
            "type": "interactive.rejected",
            "version": 1,
            "continuation_id": "cont-edit",
            "call_id": "reject",
            "tool_name": "edit-file",
            "values": {"path": "notes.txt"},
            "code": "uninspected",
            "message": "inspect first",
            "output": response(
                (
                    "edit",
                    "edit-file",
                    {
                        "path": "new.txt",
                        "old_text": FILE_EDITING.all_marker,
                        "new_text": "BAD",
                    },
                )
            )["output"],
        }

    node, server, provider = make_sdk_node(
        tmp_path,
        monkeypatch,
        [
            enter(),
            pause(path="new.txt"),
            pause(call_id="reject", exchange="next-exchange"),
            reject_and_steer,
            exit_skill(),
            response(text="done"),
        ],
    )
    try:
        result = node.run("create and edit")
        assert not (tmp_path / "new.txt").exists()
        assert (tmp_path / "notes.txt").read_text() == "KEEP"
        assert any(
            isinstance(message, InterruptedMessage)
            and message.record.get("code") == "steered"
            for message in result.skill_executions[0].history
        )
        assert not any(
            item.get("name") == "inspect-file"
            for body in server.requests
            for item in body.get("input", [])
        )
    finally:
        provider.get_session().close()
        server.client.close()


def test_recovery_inspection_stays_in_the_skill_that_requested_it(
    tmp_path, monkeypatch
):
    (tmp_path / "notes.txt").write_text("KEEP")
    rejected = {
        "type": "interactive.rejected",
        "version": 1,
        "continuation_id": "cont-edit",
        "call_id": "edit",
        "tool_name": "edit-file",
        "values": {"path": "notes.txt"},
        "code": "uninspected",
        "message": "inspect first",
        "output": exit_skill()["output"],
    }
    node, server, provider = make_sdk_node(
        tmp_path,
        monkeypatch,
        [
            enter(),
            pause(),
            rejected,
            response(text="return complete"),
            response(text="done"),
        ],
    )
    try:
        node.run("edit")
        assert node.depth == 0
        assert (tmp_path / "notes.txt").read_text() == "KEEP"
        assert server.requests[-2]["agentic"]["operation"] == "return"
        assert server.requests[-2]["input"][-1]["call_id"] == "open"
        assert "operation" not in server.requests[-1]["agentic"]
        assert not any(
            item.get("name") == "inspect-file"
            for body in server.requests
            for item in body.get("input", [])
        )
    finally:
        provider.get_session().close()
        server.client.close()


def test_logical_timeout_is_not_reset_after_the_pause(tmp_path, monkeypatch):
    monkeypatch.setattr(
        interactive_openai,
        "FILE_EDITING",
        replace(FILE_EDITING, protocol_timeout_seconds=0.1),
    )

    def delayed_pause(body):
        class DelayedStream(httpx.SyncByteStream):
            def __iter__(self):
                accepted = {
                    "type": "interactive.accepted",
                    "version": 1,
                    "continuation_id": "cont-edit",
                }
                yield f"data: {json.dumps(accepted)}\n\n".encode()
                sleep(0.15)
                yield f"data: {json.dumps(pause())}\n\n".encode()

        return httpx.Response(
            200,
            stream=DelayedStream(),
            headers={"content-type": "text/event-stream"},
        )

    node, server, provider = make_sdk_node(
        tmp_path, monkeypatch, [enter(), delayed_pause, cancelled()]
    )
    try:
        with pytest.raises(InteractiveProtocolError, match="timeout"):
            node.run("create")
        assert server.requests[-1]["agentic"]["interactive"]["operation"] == "cancel"
        assert not (tmp_path / "notes.txt").exists()
    finally:
        provider.get_session().close()
        server.client.close()


def test_server_cannot_complete_an_edit_after_the_client_rejected_it(
    tmp_path, monkeypatch
):
    (tmp_path / "notes.txt").write_text("KEEP")
    final = response(
        (
            "edit",
            "edit-file",
            {
                "path": "notes.txt",
                "old_text": FILE_EDITING.all_marker,
                "new_text": "BAD",
            },
        ),
        continuation="cont-edit",
    )
    node, server, provider = make_sdk_node(
        tmp_path, monkeypatch, [enter(), pause(), final, cancelled()]
    )
    try:
        with pytest.raises(
            InteractiveProtocolError, match="after its resource was rejected"
        ):
            node.run("edit")
        assert (tmp_path / "notes.txt").read_text() == "KEEP"
    finally:
        provider.get_session().close()
        server.client.close()


def test_fragmented_sse_utf8_and_escaped_json_resume_the_same_call(
    tmp_path, monkeypatch
):
    filename = 'quoted"\\name-\u00e8.txt'
    text = 'emoji \U0001f60a\r\n"quotes"\\backslash'

    class Chunks(httpx.SyncByteStream):
        def __init__(self, content):
            self.content = content

        def __iter__(self):
            for index in range(len(self.content)):
                yield self.content[index : index + 1]

    def resume(body):
        final = response(
            (
                "edit",
                "edit-file",
                {
                    "path": filename,
                    "old_text": FILE_EDITING.all_marker,
                    "new_text": text,
                },
            ),
            continuation="cont-edit",
        )
        event = {"type": "interactive.result", "response": final}
        wire = f"event: interactive.result\ndata: {json.dumps(event, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
        return httpx.Response(
            200,
            stream=Chunks(wire.encode()),
            headers={"content-type": "text/event-stream"},
        )

    node, server, provider = make_sdk_node(
        tmp_path,
        monkeypatch,
        [enter(), pause(path=filename), resume, exit_skill(), response(text="done")],
        streaming=True,
    )
    try:
        node.run("create")
        assert (tmp_path / filename).read_bytes() == text.encode()
    finally:
        provider.get_session().close()
        server.client.close()


def test_rejected_priority_values_preserve_json_types():
    definition = {
        "behavior": "edit_file.v1",
        "first_fields": ["target", "flag"],
    }
    contract = InteractiveContract(
        {"renamed": definition},
        {
            "renamed": {
                "properties": {
                    "target": {"type": "string"},
                    "flag": {"type": "boolean"},
                }
            }
        },
    )
    event = pause()
    event.update(tool_name="renamed")
    event["request"]["values"] = {"target": "notes.txt", "flag": True}
    contract.record_pause(event)
    contract.record_reply(event, {"status": "reject"})
    with pytest.raises(InteractiveProtocolError, match="does not match"):
        contract.validate_interruption(
            {
                "call_id": "edit",
                "tool_name": "renamed",
                "values": {"target": "notes.txt", "flag": 1},
            },
        )


def test_interrupted_record_cannot_diverge_from_its_saved_or_provider_content():
    original = {"type": "interactive.rejected", "values": {"path": "notes.txt"}}
    message = InterruptedMessage(original)
    saved = message.message_dump()
    original["values"]["path"] = "changed.txt"
    exposed = message.record
    exposed["values"]["path"] = "changed-again.txt"
    assert message.message_dump() == saved
    assert (
        serialize_user_message(message)["interactive"]["values"]["path"] == "notes.txt"
    )
    assert ToOpenAIResponseMapper.map_message(message) == [
        {"type": "interactive.rejected", "values": {"path": "notes.txt"}}
    ]


@pytest.mark.parametrize(
    "outcome", ["rejected", "cancelled", "expired", "failed", "steered"]
)
def test_interrupted_stream_reports_final_usage_once(tmp_path, monkeypatch, outcome):
    def edit(path: str, old_text: str, new_text: str):
        pytest.fail("Interrupted calls must never execute")

    tool = Tool(
        edit,
        name="edit-file",
        background_enabled=False,
        interactive=InteractiveDefinition(
            "edit_file.v1",
            {"resource": "path", "selection": "old_text", "replacement": "new_text"},
            ("path",),
            {},
        ),
    )
    events = []

    class Events(ProviderEventHandler):
        def handle(self, event: ProviderStreamEvent) -> None:
            events.append(event)

    class Handler:
        cancelled = False

        def begin(self):
            pass

        def prepare(self, event, definition):
            if outcome == "rejected":
                return {"status": "reject", "code": "unread", "message": "read first"}
            return {"status": "create"}

        def cancellation_requested(self):
            return self.cancelled

        def validate_completed(self, calls):
            pytest.fail("Interrupted calls must never reach dispatch validation")

    handler = Handler()

    def terminal(body):
        complete = response(
            (
                "edit",
                "edit-file",
                {
                    "path": "notes.txt",
                    "old_text": FILE_EDITING.all_marker,
                    "new_text": "new",
                },
            ),
            continuation="cont-edit",
        )
        if outcome == "steered":
            handler.cancelled = True
            return complete
        return {
            "type": f"interactive.{outcome}",
            "version": 1,
            "continuation_id": "cont-edit",
            "call_id": "edit",
            "tool_name": "edit-file",
            "values": {"path": "notes.txt"},
            "code": outcome,
            "message": outcome,
            "output": [],
            "usage": complete["usage"],
        }

    server = ScriptedServer([pause(), terminal], streaming=True)
    monkeypatch.setattr(LocalProvider, "_build_client", lambda self: server.client)
    provider = LocalProvider(
        "model",
        api="responses",
        base_url="http://test/v1",
        tokenizer=CacheTestTokenizer("interrupted-usage", tmp_path / "cache"),
    )
    request = ProviderRequest(
        [SystemMessage("edit"), UserMessage("new")],
        tools=Tools([tool]),
        stream=True,
        on_event=Events(),
        extra={
            "agentic": {
                "agent_context_id": "scope",
                "allowed_tools": ["edit-file"],
                "allowed_skills": [],
            }
        },
    )
    request.interactive_handler = handler
    try:
        result = provider.call(request)
        assert result.interruptions and not result.get_tool_calls()
        assert result.usage.total_tokens == 30
        totals = [
            event.usage
            for event in events
            if event.type == ProviderStreamEventType.TOTALS
        ]
        assert totals == [result.usage]
        assert events[-1].type == ProviderStreamEventType.END
    finally:
        provider.get_session().close()
        server.client.close()
