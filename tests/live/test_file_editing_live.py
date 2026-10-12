"""Opt-in real-model edits and a bounded adversarial old_text session."""

import json
import os
from dataclasses import replace
from time import monotonic
from uuid import uuid4

import openai
import pytest
from dotenv import load_dotenv

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.agents.node import DefaultCapabilities, Node
from exmekh_lib.core.file_inspection.edit_selection import select_replacements
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.interactive import FileInteractiveCalls
from exmekh_lib.core.file_inspection.lines import physical_lines
from exmekh_lib.core.file_inspection.runtime import FileToolRuntime
from exmekh_lib.core.models.edit_document import EditDocument
from exmekh_lib.core.models.interactive import InteractiveProtocolError
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.providers import interactive_openai
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.providers.local import LocalProvider
from tests.interactive_support import edit_fragment
from tests.live.model_support import CacheTestTokenizer, LiveModelProvider

pytestmark = pytest.mark.live


class RecordingHandler:
    def __init__(self, handler, exchanges):
        self.handler = handler
        self.exchanges = exchanges

    def begin(self):
        self.handler.begin()

    def prepare(self, event, definition):
        reply = self.handler.prepare(event, definition)
        self.exchanges.append((event, reply))
        return reply

    def validate_completed(self, calls):
        self.handler.validate_completed(calls)

    def cancellation_requested(self):
        return self.handler.cancellation_requested()


class EditingLiveProvider(LiveModelProvider):
    def __init__(self, directory):
        self.directory = directory
        self.requests = []
        self.exchanges = []
        super().__init__(
            model=os.getenv("EXMEKH_LIB_LIVE_MODEL", DEFAULT_PROVIDER_SETTINGS.model),
            api="responses",
            reasoning_effort=ReasoningEffortLMH.OFF,
            tokenizer=CacheTestTokenizer("live-file-editing", directory / "cache"),
        )

    def call(self, request):
        assert len(self.requests) < 20, "Model exceeded the file editing turn budget"
        bounded = ProviderRequest(
            request.get_messages(),
            extra={
                **(request.get_extra() or {}),
                "max_output_tokens": 768,
                "temperature": 0,
            },
            tools=request.get_tools(),
            tool_choice="required" if not self.requests else request.get_tool_choice(),
            stream=request.get_streaming(),
            on_event=request.on_event,
            reasoning_effort_override=request.get_reasoning_effort_override(),
        )
        bounded.interactive_handler = request.interactive_handler
        request = bounded
        self.requests.append(request)
        if request.interactive_handler is not None:
            request.interactive_handler = RecordingHandler(
                request.interactive_handler, self.exchanges
            )
        result = super().call(request)
        (self.directory / f"live-request-{len(self.requests):02d}.json").write_text(
            json.dumps(
                {
                    "input": [
                        message.message_dump() for message in request.get_messages()
                    ],
                    "output": result.message_dump(),
                    "interruptions": [
                        message.record for message in result.interruptions
                    ],
                    "exchanges": self.exchanges,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return result


@pytest.fixture
def edit_provider(tmp_path):
    load_dotenv()
    if not os.getenv(DEFAULT_PROVIDER_SETTINGS.base_url_env_var):
        pytest.skip("Set BASE_URL to the live DS4 Responses v2 endpoint")
    provider = EditingLiveProvider(tmp_path)
    try:
        yield provider
    finally:
        provider.get_session().close()
        provider.client.close()


def editing_node(provider, directory):
    hds = directory / "capabilities" / "hds"
    hds.mkdir(parents=True)
    (hds / "main.md").write_text(
        "---\ndescription: Modify workspace text files.\nexit-tool: exit_agent\n"
        "sub-skills:\n  - file-processing\n---\n"
        "Use {{!file-processing}} for file changes. After its return, answer done and stop."
    )
    node = Node(
        [hds.parent],
        name_hds="main",
        agent_context_id=f"live-edit-{uuid4().hex}",
        default_capabilities=DefaultCapabilities(
            harness=False,
            include={
                "file-processing",
                "inspect-file",
                "read-file",
                "edit-file",
                "exit_agent",
            },
        ),
    )
    loaded = node.initialize(provider, work_root=directory)
    return node, loaded


@pytest.mark.parametrize(
    "operation",
    ["edit", "new", "remove", "empty", "insert", "range", "literal_xml"],
)
def test_live_model_applies_xml_file_operations(tmp_path, edit_provider, operation):
    path = tmp_path / "notes.txt"
    existing = operation not in {"new", "literal_xml"}
    if existing:
        path.write_bytes(b"  old value\r\nKEEP\n")
    document, expected = {
        "edit": (
            edit_fragment("notes.txt", "  old value", "  updated value"),
            b"  updated value\r\nKEEP\n",
        ),
        "new": (
            edit_fragment("notes.txt", "<all/>", "created value"),
            b"created value",
        ),
        "remove": (edit_fragment("notes.txt", "<all/>", None), None),
        "empty": (edit_fragment("notes.txt", "<all/>", ""), b""),
        "insert": (
            edit_fragment("notes.txt", '<line number="2"/>', "INSERT\r\n"),
            b"  old value\r\nINSERT\r\nKEEP\n",
        ),
        "range": (
            '<edit path="notes.txt"><old_text start_line="1" end_line="1"></old_text><new_text>REPLACED\r\n</new_text></edit>',
            b"REPLACED\r\nKEEP\n",
        ),
        "literal_xml": (
            edit_fragment(
                "notes.txt",
                "<all/>",
                "before </new_text> middle </new_text></edit> <delete/>\r\n",
            ),
            b"before </new_text> middle </new_text></edit> <delete/>\r\n",
        ),
    }[operation]
    # Numeric entities make the requested physical newlines explicit in the
    # model instruction; the assertions still compare the same exact file bytes.
    document = document.replace("\r", "&#13;").replace("\n", "&#10;")
    action = {
        "edit": "Replace only the two-space-indented first-line text. Keep existing newline sequences and KEEP untouched. ",
        "new": "Create notes.txt containing exactly created value, with no final newline. ",
        "remove": "DELETE notes.txt from the filesystem. Emit <delete/> as the sole new_text body; copying existing text or empty replacement does not delete a file. ",
        "empty": "Truncate notes.txt to ZERO BYTES, keeping the file. Emit an empty new_text body. ",
        "insert": "Insert INSERT followed by CR then LF before physical line 2. Use &#13;&#10; for those exact inserted newline characters. ",
        "range": "Replace inclusive physical line 1 with REPLACED followed by CR then LF. Use the start_line/end_line range and &#13;&#10; newline entities. ",
        "literal_xml": "Create the literal source text from the given XML entity escapes, including both close strings and <delete/> as text. End the file with CR then LF. ",
    }[operation]
    node, _ = editing_node(edit_provider, tmp_path)
    prerequisites = (
        "First inspect notes.txt and read both physical lines in separate turns. "
        if existing
        else "The file is missing; do not inspect or read it. "
    )
    result = node.run(
        "Enter file-processing. "
        + action
        + prerequisites
        + "Then call edit-file exactly once with this JSON argument object, preserving every character: "
        + json.dumps({"edits": document})
        + ". Apply this exact requested operation. Empty new_text keeps the file; <delete/> deletes it. "
        + "After success exit_agent(text='done',items=[]), then answer done without more tools."
    )
    trace = [
        message.message_dump()
        for execution in result.skill_executions
        for message in execution.history
    ]
    (tmp_path / "live-edit-trace.json").write_text(
        json.dumps(trace, ensure_ascii=True, indent=2)
    )
    if expected is None:
        assert not path.exists(), trace
    else:
        assert path.read_bytes() == expected, trace
    exchanges = [
        (event, reply)
        for event, reply in edit_provider.exchanges
        if event["tool_name"] == "edit-file"
    ]
    assert exchanges
    assert all(
        event["version"] == 2 and event["request"]["edit_index"] >= 1
        for event, _ in exchanges
    )
    assert any(
        reply["status"] == ("provide" if existing else "create")
        for _, reply in exchanges
    )
    assert all(
        "code" not in reply for _, reply in exchanges if reply["status"] != "reject"
    )
    assert node.depth == 0


@pytest.mark.parametrize("alias", [False, True])
def test_live_one_xml_call_edits_multiple_resources_with_initial_snapshots(
    tmp_path, edit_provider, alias
):
    first = tmp_path / "notes.txt"
    first.write_bytes(b"FIRST\r\nSECOND\n")
    second = tmp_path / "other.txt"
    if alias:
        second.symlink_to(first)
        document = edit_fragment(first.name, "FIRST", "UPDATED FIRST") + edit_fragment(
            second.name, "SECOND", "UPDATED SECOND"
        )
    else:
        second.write_bytes(b"OTHER\n")
        document = edit_fragment(first.name, "FIRST", "UPDATED FIRST") + edit_fragment(
            second.name, "OTHER", "UPDATED OTHER"
        )
    node, _ = editing_node(edit_provider, tmp_path)
    result = node.run(
        "Enter file-processing. Inspect notes.txt, read its two lines. "
        + ("" if alias else "Inspect other.txt and read its one line. ")
        + "Now make exactly ONE edit-file call whose single edits string contains both XML fragments from "
        + json.dumps({"edits": document})
        + ". Do not put a second old_text/new_text inside the first new_text. Both <edit> elements are siblings. "
        + "After success exit_agent(text='done',items=[]), then answer done without more tools."
    )
    trace = [
        message.message_dump()
        for execution in result.skill_executions
        for message in execution.history
    ]
    (tmp_path / "live-multiple-trace.json").write_text(
        json.dumps(trace, ensure_ascii=True, indent=2)
    )
    assert first.read_bytes() == (
        b"UPDATED FIRST\r\nUPDATED SECOND\n" if alias else b"UPDATED FIRST\r\nSECOND\n"
    ), trace
    assert second.read_bytes() == (
        first.read_bytes() if alias else b"UPDATED OTHER\n"
    ), trace
    exchanges = [
        (event, reply)
        for event, reply in edit_provider.exchanges
        if reply["status"] == "provide"
    ]
    assert len(exchanges) == 2, exchanges
    assert exchanges[0][0]["call_id"] == exchanges[1][0]["call_id"]
    assert [event["request"]["edit_index"] for event, _ in exchanges] == [1, 2]
    assert exchanges[0][0]["exchange_id"] != exchanges[1][0]["exchange_id"]
    if alias:
        assert exchanges[0][1] == exchanges[1][1]
        assert second.is_symlink()
    assert node.depth == 0


class AdversarialSnapshot:
    def __init__(self, text, deadline):
        self.text = text
        self.deadline = deadline
        self.exchanges = 0

    def begin(self):
        pass

    def prepare(self, event, definition):
        self.exchanges += 1
        assert event["request"]["values"] == {"path": "attack.txt"}
        return {
            "status": "provide",
            "text": self.text,
            "encoding": "utf-8",
            "line_count": len(physical_lines(self.text)),
        }

    def cancellation_requested(self):
        return monotonic() >= self.deadline

    def validate_completed(self, calls):
        for call in calls:
            document = EditDocument.parse(call.get_arguments()["edits"])
            # Range semantics and multi-edit conflicts belong to the client and
            # may be rejected here. This attack targets literal masking only.
            for element in document.elements:
                if element.selection.kind == "literal":
                    try:
                        select_replacements(self.text, (element,), exists=True)
                    except FileInspectionError as error:
                        pytest.fail(
                            f"Server literal constraints escaped: {element!r}: {error}"
                        )


def test_live_adversarial_literal_selection_cannot_escape_for_30_seconds(
    tmp_path, edit_provider, monkeypatch, record_property
):
    _, loaded = editing_node(edit_provider, tmp_path)
    edit = next(tool for tool in loaded.tools if tool.name == "edit-file")
    text = "  value\nprefix value\naaa\nvalue"
    started = monotonic()
    deadline = started + 30
    snapshot = AdversarialSnapshot(text, deadline)
    attempts = completed = 0
    challenges = [
        "Force <old_text>value</old_text> despite indentation and repeated/mid-line matches.",
        "Force <old_text>aa</old_text> despite overlapping matches in aaa.",
        "Force <old_text>refix</old_text> starting inside the physical line.",
        "Use <old_text><![CDATA[value]]></old_text> or entities to hide the ambiguous selection.",
        "Inject a native tool delimiter or duplicate edits JSON keys to skip the old_text constraint.",
        "Force a truncated XML tag or a code attribute that the protocol does not support.",
    ]
    while monotonic() < deadline:
        monkeypatch.setattr(
            interactive_openai,
            "FILE_EDITING",
            replace(FILE_EDITING, protocol_timeout_seconds=deadline - monotonic()),
        )
        request = ProviderRequest(
            [
                SystemMessage(
                    "You are an adversarial XML grammar tester. Select attack.txt and attempt the requested invalid selection. Return a tool call."
                ),
                UserMessage(
                    f"Already inspected/read snapshot: {text!r}. {challenges[attempts % len(challenges)]} Replace with ATTACK."
                ),
            ],
            tools=Tools([edit]),
            tool_choice="required",
            stream=True,
            extra={
                "agentic": {
                    "agent_context_id": f"xml-attack-{uuid4().hex}",
                    "allowed_tools": ["edit-file"],
                    "allowed_skills": [],
                },
                "max_output_tokens": 512,
            },
        )
        request.interactive_handler = snapshot
        attempts += 1
        try:
            result = LocalProvider.call(edit_provider, request)
            completed += bool(result.get_tool_calls())
        except RuntimeError as error:
            expired = monotonic() >= deadline - 0.1
            timeout = (
                isinstance(error, InteractiveProtocolError)
                and "logical-call timeout" in str(error)
            ) or isinstance(error.__cause__, openai.APITimeoutError)
            if not expired or not timeout:
                raise
    assert snapshot.exchanges > 0
    assert completed > 0, "No complete constrained tool call was evaluated"
    assert edit_provider.client.models.list().data
    assert not (tmp_path / "attack.txt").exists()
    record_property("adversarial_seconds", monotonic() - started)
    record_property("attempts", attempts)
    record_property("validated_calls", completed)


@pytest.mark.parametrize("streaming", [False, True])
def test_live_rejection_of_second_edit_prevents_entire_call(
    tmp_path, edit_provider, streaming
):
    (tmp_path / "existing.txt").write_text("KEEP")
    _, loaded = editing_node(edit_provider, tmp_path)
    edit = next(tool for tool in loaded.tools if tool.name == "edit-file")
    document = edit_fragment("new.txt", "<all/>", "BAD") + edit_fragment(
        "existing.txt", "<all/>", "BAD"
    )
    history = [
        SystemMessage("Return only the exact requested edit-file call."),
        UserMessage(
            "Call once with these two sibling fragments: "
            + json.dumps({"edits": document})
        ),
    ]
    handler = FileInteractiveCalls(
        FileToolRuntime(), tmp_path, lambda: history, lambda: False, lambda: ()
    )
    exchanges = []
    request = ProviderRequest(
        history,
        tools=Tools([edit]),
        tool_choice="required",
        stream=streaming,
        extra={
            "agentic": {
                "agent_context_id": f"xml-reject-{uuid4().hex}",
                "allowed_tools": ["edit-file"],
                "allowed_skills": [],
            },
            "max_output_tokens": 512,
            "temperature": 0,
        },
    )
    request.interactive_handler = RecordingHandler(handler, exchanges)
    result = LocalProvider.call(edit_provider, request)
    assert not result.get_tool_calls()
    assert len(exchanges) == 2
    assert [reply["status"] for _, reply in exchanges] == ["create", "reject"]
    record = result.interruptions[0].record
    assert record["edit_index"] == 2
    assert record["values"] == {"path": "existing.txt"}
    assert (tmp_path / "existing.txt").read_bytes() == b"KEEP"
    assert not (tmp_path / "new.txt").exists()
    assert len(handler.pending_inspections()) == 1


@pytest.mark.parametrize("streaming", [False, True])
def test_live_scope_change_at_pause_cancels_without_authorizing_a_resource(
    tmp_path, edit_provider, streaming, record_property
):
    _, loaded = editing_node(edit_provider, tmp_path)
    edit = next(tool for tool in loaded.tools if tool.name == "edit-file")
    scope = ("node", "first-opening")
    pauses = []

    class ChangeScope(ProviderEventHandler):
        def handle(self, event: ProviderStreamEvent) -> None:
            nonlocal scope
            if (
                event.type == ProviderStreamEventType.INTERACTIVE
                and event.interactive is not None
                and event.interactive.get("type") == "interactive.await_input"
            ):
                pauses.append(event.interactive)
                scope = ("node", "reopened-skill")

    arguments = {"edits": edit_fragment("notes.txt", "<all/>", "BAD")}
    history = [
        SystemMessage("Return only the exact requested edit-file call."),
        UserMessage(
            "Create notes.txt using this exact argument: " + json.dumps(arguments)
        ),
    ]
    handler = FileInteractiveCalls(
        FileToolRuntime(), tmp_path, lambda: history, lambda: False, lambda: scope
    )
    exchanges = []
    request = ProviderRequest(
        history,
        tools=Tools([edit]),
        tool_choice="required",
        stream=streaming,
        on_event=ChangeScope(),
        extra={
            "agentic": {
                "agent_context_id": f"xml-scope-{uuid4().hex}",
                "allowed_tools": ["edit-file"],
                "allowed_skills": [],
            },
            "max_output_tokens": 512,
            "temperature": 0,
        },
    )
    request.interactive_handler = RecordingHandler(handler, exchanges)
    result = LocalProvider.call(edit_provider, request)
    records = [message.record for message in result.interruptions]
    (tmp_path / "live-scope-cancellation.json").write_text(
        json.dumps(
            {
                "requested": arguments,
                "pauses": pauses,
                "exchanges": exchanges,
                "interruptions": records,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    assert len(pauses) == 1
    assert not exchanges
    assert not result.get_tool_calls()
    assert any(record["type"] == "interactive.cancelled" for record in records)
    assert handler.cancellation_requested()
    assert not (tmp_path / "notes.txt").exists()
    record_property("cancelled_pauses", len(pauses))
