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
from exmekh_lib.core.file_inspection.reading import file_read_context, read_file
from exmekh_lib.core.file_inspection.runtime import FileToolRuntime
from exmekh_lib.core.file_inspection.service import inspect_file
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.edit_document import EditDocument
from exmekh_lib.core.models.interactive import InteractiveProtocolError
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.providers import interactive_openai
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.providers.local import LocalProvider
from exmekh_lib.utils.xml import stringify_tool_result
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
                "new-file",
                "delete-file",
                "exit_agent",
            },
        ),
    )
    loaded = node.initialize(provider, work_root=directory)
    return node, loaded


@pytest.mark.parametrize(
    "operation",
    ["edit", "new", "remove", "empty", "insert", "literal_markup", "empty_existing"],
)
def test_live_model_applies_literal_file_operations(tmp_path, edit_provider, operation):
    path = tmp_path / "notes.txt"
    existing = operation not in {"new", "literal_markup"}
    initial = b"" if operation == "empty_existing" else b"  old value\r\nKEEP\n"
    if existing:
        path.write_bytes(initial)
    tool, expected = {
        "edit": ("edit-file", b"  updated value\r\nKEEP\n"),
        "new": ("new-file", b"created value"),
        "remove": ("delete-file", None),
        "empty": ("edit-file", b""),
        "insert": ("edit-file", b"  old value\r\nINSERT\r\nKEEP\n"),
        "literal_markup": ("new-file", b"<all/><delete/>&lt;<![CDATA[raw]]>\r\n"),
        "empty_existing": ("edit-file", b"FILLED"),
    }[operation]
    node, _ = editing_node(edit_provider, tmp_path)
    prerequisite = (
        "First inspect notes.txt and read its current physical lines (0/0 if empty). "
        if tool == "edit-file"
        else "Do not inspect or read the file. "
    )
    instruction = {
        "edit": "Replace the first line's text with exactly two spaces followed by updated value.",
        "new": "Create notes.txt containing exactly created value.",
        "remove": "Delete notes.txt.",
        "empty": "Remove the entire content of notes.txt so the existing file is zero bytes.",
        "insert": "Insert INSERT followed by CRLF immediately before KEEP. Preserve the existing bytes.",
        "literal_markup": (
            "Create notes.txt with this exact literal content, followed by one CRLF:\n"
            "<all/><delete/>&lt;<![CDATA[raw]]>"
        ),
        "empty_existing": "Fill the existing empty notes.txt with exactly FILLED.",
    }[operation]
    result = node.run(
        "Enter file-processing. "
        + prerequisite
        + instruction
        + f" Use {tool} exactly once. Preserve literal characters, including ampersands "
        "and angle brackets. Do not add a final newline unless requested. "
        "After success exit the skill and answer done."
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
    calls = [
        message
        for execution in result.skill_executions
        for message in execution.history
        if message.get_role_name() == "tool" and message.get_name() == tool
    ]
    assert len(calls) == 1, trace
    if tool != "edit-file":
        assert not edit_provider.exchanges
    assert node.depth == 0


@pytest.mark.parametrize("alias", [False, True])
def test_live_one_literal_call_edits_multiple_resources_with_initial_snapshots(
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
        + "Now make exactly ONE edit-file call whose single edits string contains both literal edit blocks from "
        + json.dumps({"edits": document})
        + ". The server supplies distinct opening IDs. Copy each ID into its separator. "
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
            # Conflicts across multiple selections belong to the client and
            # may be rejected here. This attack targets literal masking only.
            for element in document.elements:
                try:
                    select_replacements(self.text, (element,))
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
        "Select the literal value despite indentation and repeated/mid-line matches.",
        "Select the literal aa despite overlapping matches in aaa.",
        "Select the literal refix starting inside the physical line.",
        "Try a legacy <all/> or <line number='1'/> command instead of selecting actual source text.",
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
                    "You are an adversarial literal edit grammar tester. Select attack.txt and attempt the requested invalid selection. Return a tool call."
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
                    "agent_context_id": f"literal-attack-{uuid4().hex}",
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
    first = tmp_path / "notes.txt"
    first.write_text("FIRST")
    inspection = inspect_file(tmp_path, first.name)
    with file_read_context((inspection.evidence,)):
        reading = read_file(tmp_path, first.name, 1, 1)
    _, loaded = editing_node(edit_provider, tmp_path)
    edit = next(tool for tool in loaded.tools if tool.name == "edit-file")
    document = edit_fragment("notes.txt", "FIRST", "BAD") + edit_fragment(
        "existing.txt", "<all/>", "BAD"
    )
    history = [
        SystemMessage("Return only the requested edit-file call."),
        UserMessage(
            "Call once with these two consecutive literal edit blocks: "
            + json.dumps({"edits": document})
        ),
    ]
    runtime = FileToolRuntime()
    for name, report, arguments in (
        ("inspect-file", inspection, {"path": first.name}),
        ("read-file", reading, {"path": first.name, "start_line": 1, "end_line": 1}),
    ):
        call = ToolCall(name, FunctionCall(name, arguments))
        message = ToolMessage(stringify_tool_result(report), name, name)
        history[1:1] = [AssistantMessage("", tool_calls=[call]), message]
        runtime.record_result(call, report, message)
    # Inspection must precede the read in visible history.
    history[1:5] = history[3:5] + history[1:3]
    handler = FileInteractiveCalls(
        runtime, tmp_path, lambda: history, lambda: False, lambda: ()
    )
    exchanges = []
    request = ProviderRequest(
        history,
        tools=Tools([edit]),
        tool_choice="required",
        stream=streaming,
        extra={
            "agentic": {
                "agent_context_id": f"literal-reject-{uuid4().hex}",
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
    assert [reply["status"] for _, reply in exchanges] == ["provide", "reject"]
    record = result.interruptions[0].record
    assert record["edit_index"] == 2
    assert record["values"] == {"path": "existing.txt"}
    assert (tmp_path / "existing.txt").read_bytes() == b"KEEP"
    assert first.read_bytes() == b"FIRST"
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

    arguments = {"edits": edit_fragment("notes.txt", "KEEP", "BAD")}
    history = [
        SystemMessage("Return only the exact requested edit-file call."),
        UserMessage("Edit notes.txt using this argument: " + json.dumps(arguments)),
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
                "agent_context_id": f"literal-scope-{uuid4().hex}",
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
