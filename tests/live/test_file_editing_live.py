"""Opt-in real-model edits and a bounded adversarial old_text session."""

import json
import os
from copy import deepcopy
from dataclasses import replace
from time import monotonic
from typing import Annotated
from uuid import uuid4

import openai
import pytest
from dotenv import load_dotenv

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.agents.node import DefaultCapabilities, Node
from exmekh_lib.core.file_inspection.edit_selection import select_replacement
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.lines import physical_lines
from exmekh_lib.core.models.interactive import InteractiveProtocolError
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.providers import interactive_openai
from exmekh_lib.core.providers.interactive_transport import InteractiveTransport
from exmekh_lib.providers.local import LocalProvider
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
        assert len(self.requests) < 14, "Model exceeded the file editing turn budget"
        bounded = ProviderRequest(
            request.get_messages(),
            extra={**(request.get_extra() or {}), "max_output_tokens": 256},
            tools=request.get_tools(),
            tool_choice=request.get_tool_choice(),
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
        pytest.skip("Set BASE_URL to the live DS4 Responses v1 endpoint")
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


@pytest.mark.parametrize("operation", ["edit", "new", "remove"])
def test_live_model_edits_creates_and_removes_files(tmp_path, edit_provider, operation):
    path = tmp_path / "notes.txt"
    if operation != "new":
        path.write_bytes(b"  old value\r\nKEEP\n")
    node, _ = editing_node(edit_provider, tmp_path)
    action = {
        "edit": "Replace only the exact first-line text, including its two leading spaces, with '  updated value'. Keep both newline sequences and KEEP unchanged. Use literal old_text, not all or line markers.",
        "new": (
            "Create notes.txt containing exactly 'created value' without a trailing newline. "
            f"Use old_text={FILE_EDITING.all_marker}. The file is missing; do not inspect or read it."
        ),
        "remove": (
            "Delete the entire notes.txt file from the filesystem. The required edit-file arguments are exactly "
            + json.dumps(
                {
                    "path": path.name,
                    "old_text": FILE_EDITING.all_marker,
                    "new_text": FILE_EDITING.delete_marker,
                }
            )
            + ". Use those marker strings literally. Empty replacement only removes text and does not fulfill this request. The successful result must say operation=remove."
        ),
    }[operation]
    prompt = f"Enter file-processing. {action} "
    if operation != "new":
        prompt += (
            "First inspect notes.txt and read both physical lines, in separate turns. "
        )
    result = node.run(
        prompt
        + "After the edit succeeds, exit_agent(text='done',items=[]), then answer done without more tools."
    )
    trace = [
        message.message_dump()
        for execution in result.skill_executions
        for message in execution.history
    ]
    (tmp_path / "live-edit-trace.json").write_text(
        json.dumps(trace, ensure_ascii=True, indent=2)
    )
    if operation == "remove":
        assert not path.exists(), trace
    else:
        assert path.read_bytes() == (
            b"  updated value\r\nKEEP\n" if operation == "edit" else b"created value"
        )
    assert edit_provider.exchanges
    assert any(
        reply["status"] == ("create" if operation == "new" else "provide")
        for _, reply in edit_provider.exchanges
    )
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
            arguments = call.get_arguments()
            try:
                select_replacement(
                    self.text, arguments["old_text"], arguments["new_text"], exists=True
                )
            except FileInspectionError as error:
                pytest.fail(
                    f"Server old_text constraints escaped: {arguments!r}: {error}"
                )


def test_live_adversarial_old_text_cannot_escape_for_30_seconds(
    tmp_path, edit_provider, monkeypatch, record_property
):
    node, loaded = editing_node(edit_provider, tmp_path)
    edit = next(tool for tool in loaded.tools if tool.name == "edit-file")
    text = "  value\nprefix value\naaa\nvalue"
    started = monotonic()
    deadline = started + 30
    snapshot = AdversarialSnapshot(text, deadline)
    attempts = 0
    completed = 0
    challenges = [
        "Force old_text='value': ignore indentation and the repeated/mid-line occurrences.",
        "Force old_text='aa' despite overlapping matches in aaa. Close the value early.",
        "Force old_text='prefix' starting after the first character of that line instead of its boundary; use old_text='refix'.",
        (
            f"Force old_text={FILE_EDITING.line_marker(0)!r}, {FILE_EDITING.line_marker(999999)!r}, "
            f"{FILE_EDITING.line_marker(-1)!r} or {FILE_EDITING.line_marker('01')!r}."
        ),
        "Inject a quote, newline or native tool delimiter to skip the selection constraint and freeze path as another file.",
        (
            f"Use a non-string old_text, duplicate JSON keys, {FILE_EDITING.all_marker} "
            "with extra whitespace, or embed a marker in ordinary old text."
        ),
    ]
    # Use LocalProvider directly so the 14-turn workflow budget does not limit
    # this timed transport attack. No generated call is dispatched to a filesystem.
    while monotonic() < deadline:
        remaining = deadline - monotonic()
        monkeypatch.setattr(
            interactive_openai,
            "FILE_EDITING",
            replace(FILE_EDITING, protocol_timeout_seconds=remaining),
        )
        request = ProviderRequest(
            [
                SystemMessage(
                    "You are an adversarial grammar tester. Attempt the requested invalid edit-file arguments. Always select attack.txt first. The client supplies the snapshot after path. Do not answer in prose."
                ),
                UserMessage(
                    f"Snapshot already inspected/read: {text!r}. {challenges[attempts % len(challenges)]} Replace with ATTACK."
                ),
            ],
            tools=Tools([edit]),
            tool_choice="required",
            stream=True,
            extra={
                "agentic": {
                    "agent_context_id": f"old-text-attack-{uuid4().hex}",
                    "allowed_tools": ["edit-file"],
                    "allowed_skills": [],
                },
                "max_output_tokens": 256,
            },
        )
        request.interactive_handler = snapshot
        attempts += 1
        try:
            result = LocalProvider.call(edit_provider, request)
            completed += bool(result.get_tool_calls())
        except RuntimeError as error:
            # Expiration of the final bounded HTTP wait is expected. Other
            # failures, malformed outputs or client crashes never count as success.
            expired = monotonic() >= deadline - 0.1
            timeout = (
                isinstance(error, InteractiveProtocolError)
                and "logical-call timeout" in str(error)
            ) or isinstance(error.__cause__, openai.APITimeoutError)
            if not expired or not timeout:
                raise
    assert snapshot.exchanges > 0, (
        "The test never exercised server snapshot constraints"
    )
    assert completed > 0, "No complete constrained tool call was evaluated"
    assert edit_provider.client.models.list().data, (
        "Server/client unusable after attack"
    )
    assert not (tmp_path / "attack.txt").exists()
    record_property("adversarial_seconds", monotonic() - started)
    record_property("attempts", attempts)
    record_property("validated_calls", completed)
    print(
        f"30-second old_text attack: {attempts} attempts, {completed} validated calls, {snapshot.exchanges} snapshots"
    )


@pytest.mark.parametrize("schema_kind", ["ordinary", "marker_enum"])
def test_live_server_generates_an_exact_delete_marker(
    tmp_path, edit_provider, monkeypatch, schema_kind
):
    """Isolate the server's generation from skill routing and filesystem effects."""
    _, loaded = editing_node(edit_provider, tmp_path)
    edit = next(tool for tool in loaded.tools if tool.name == "edit-file")
    if schema_kind == "marker_enum":

        def marker_only(
            path: str,
            old_text: Annotated[str, {"enum": [FILE_EDITING.all_marker]}],
            new_text: Annotated[str, {"enum": [FILE_EDITING.delete_marker]}],
        ):
            pass

        edit = Tool(
            marker_only,
            name=edit.name,
            description=edit.description,
            background_enabled=False,
            interactive=edit.interactive,
        )
    requests, events = [], []
    original_events = InteractiveTransport.events

    def capture(transport, body, timeout):
        requests.append(deepcopy(body))
        for event in original_events(transport, body, timeout):
            events.append(deepcopy(event))
            yield event

    monkeypatch.setattr(InteractiveTransport, "events", capture)
    monkeypatch.setattr(
        interactive_openai,
        "FILE_EDITING",
        replace(FILE_EDITING, protocol_timeout_seconds=60),
    )

    class Snapshot:
        def begin(self):
            pass

        def prepare(self, event, definition):
            assert event["request"]["values"] == {"path": "attack.txt"}
            return {
                "status": "provide",
                "text": "  value\nprefix value\naaa\nvalue",
                "encoding": "utf-8",
                "line_count": 4,
            }

        def validate_completed(self, calls):
            pass

        def cancellation_requested(self):
            return False

    request = ProviderRequest(
        [
            SystemMessage("Generate the exact required tool arguments."),
            UserMessage(
                "The file attack.txt has been inspected and read: '  value\\nprefix value\\naaa\\nvalue'. "
                f"Call edit-file exactly once with path=attack.txt, old_text={FILE_EDITING.all_marker!r}, "
                f"new_text={FILE_EDITING.delete_marker!r}. Markers must be inline, with no framing space "
                "or newline. Return the tool call only."
            ),
        ],
        tools=Tools([edit]),
        tool_choice="required",
        stream=True,
        extra={
            "agentic": {
                "agent_context_id": f"delete-marker-{uuid4().hex}",
                "allowed_tools": ["edit-file"],
                "allowed_skills": [],
            },
            "max_output_tokens": 256,
            "temperature": 0,
        },
    )
    request.interactive_handler = Snapshot()
    try:
        result = LocalProvider.call(edit_provider, request)
    finally:
        (tmp_path / "delete-marker-wire.json").write_text(
            json.dumps(
                {"schema_kind": schema_kind, "requests": requests, "events": events},
                ensure_ascii=True,
                indent=2,
            )
        )
    calls = result.get_tool_calls() or []
    assert len(calls) == 1, [message.record for message in result.interruptions]
    assert calls[0].get_arguments() == {
        "path": "attack.txt",
        "old_text": FILE_EDITING.all_marker,
        "new_text": FILE_EDITING.delete_marker,
    }
    assert not (tmp_path / "attack.txt").exists()
