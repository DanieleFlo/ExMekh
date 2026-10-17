"""Opt-in real-model edits and a bounded adversarial old_text session."""

import json
import os
import re
from dataclasses import replace
from time import monotonic
from uuid import uuid4

import openai
import pytest
from dotenv import load_dotenv

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.capabilities.defaults import BUILTIN_CAPABILITIES_PATH
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
from exmekh_lib.core.models.creation_document import CreationDocument
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
from exmekh_lib.utils.tool_errors import ToolInputError
from exmekh_lib.utils.xml import stringify_tool_result
from tests.interactive_support import edit_fragment
from tests.live.model_support import CacheTestTokenizer, LiveModelProvider
from tests.tool_result_support import parse_tool_result

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
        bounded = ProviderRequest(
            request.get_messages(),
            extra={
                **(request.get_extra() or {}),
                "max_output_tokens": 768,
                "temperature": 0,
            },
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
        pytest.skip("Set BASE_URL to the live DS4 Responses v2 endpoint")
    provider = EditingLiveProvider(tmp_path)
    try:
        yield provider
    finally:
        provider.get_session().close()
        provider.client.close()


def file_tools(directory):
    loaded = Capabilities([BUILTIN_CAPABILITIES_PATH]).load(work_root=directory)
    return {tool.get_name(): tool for tool in loaded.tools}


def execute_live_file_call(provider, directory, tool_name, arguments, initial):
    """Dispatch one real call after real prerequisite reads, without a model strategy loop."""
    tools = file_tools(directory)
    history = [
        SystemMessage("Return only the exact requested file tool call."),
        UserMessage(
            f"Call {tool_name} once with this exact argument object: "
            + json.dumps(arguments)
            + ". Opening IDs are assigned by the server. Copy each ID into its separator; preserve all literal body characters."
        ),
    ]
    runtime = FileToolRuntime()
    if tool_name == "edit-file":
        for path in initial:
            resource = path.relative_to(directory).as_posix()
            line_count = len(physical_lines(initial[path].decode()))
            start_index = 1 if line_count else 0
            inspection = inspect_file(
                directory, resource, tokenizer=provider.get_tokenizer()
            )
            with file_read_context((inspection.evidence,)):
                reading = read_file(directory, resource, start_index, line_count)
            for name, report, inputs in (
                ("inspect-file", inspection, {"path": resource}),
                (
                    "read-file",
                    reading,
                    {
                        "path": resource,
                        "start_index": start_index,
                        "end_index": line_count,
                    },
                ),
            ):
                call = ToolCall(f"{name}-{len(history)}", FunctionCall(name, inputs))
                message = ToolMessage(
                    stringify_tool_result(report), call.get_id(), name
                )
                history.extend([AssistantMessage("", tool_calls=[call]), message])
                runtime.record_result(call, report, message)
    handler = FileInteractiveCalls(
        runtime, directory, lambda: history, lambda: False, lambda: ()
    )
    request = ProviderRequest(
        history,
        tools=Tools([tools[tool_name]]),
        tool_choice="required",
        stream=True,
        extra={
            "agentic": {
                "agent_context_id": f"live-file-call-{uuid4().hex}",
                "allowed_tools": [tool_name],
                "allowed_skills": [],
            }
        },
    )
    request.interactive_handler = handler
    result = provider.call(request)
    calls = result.get_tool_calls() or []
    assert len(calls) == 1 and calls[0].get_name() == tool_name, (
        f"COVERAGE: expected one complete {tool_name} call, got {calls!r}"
    )
    call = calls[0]
    history.append(AssistantMessage(result.get_message(), tool_calls=calls))
    with handler.execution_context(call.get_id()):
        receipt = tools[tool_name].execute_with_hooks(
            call.get_arguments(),
            call_id=call.get_id(),
            work_root=directory,
            history=tuple(history),
        )
    history.append(
        ToolMessage(stringify_tool_result(receipt), call.get_id(), tool_name)
    )
    (directory / "live-edit-trace.json").write_text(
        json.dumps(
            [message.message_dump() for message in history], ensure_ascii=True, indent=2
        )
    )
    verify_file_effects(history, directory, initial)
    assert receipt.get("ok") is True, receipt
    return call, receipt


def _expected_edit_bytes(initial, elements):
    assert initial is not None, "IMPLEMENTATION: edit accepted a missing file"
    text = initial.decode()
    assert text or len(elements) == 1, (
        "IMPLEMENTATION: multiple selections of an empty file were accepted"
    )
    line_starts = {0, *(match.end() for match in re.finditer(r"\r\n|\r|\n", text))}
    replacements = []
    for element in elements:
        starts = (
            [
                match.start()
                for match in re.finditer(f"(?={re.escape(element.old_text)})", text)
            ]
            if element.old_text
            else [0]
            if not text
            else []
        )
        assert len(starts) == 1, (
            "IMPLEMENTATION: accepted selection is absent or ambiguous"
        )
        start = starts[0]
        assert start in line_starts, (
            "IMPLEMENTATION: accepted selection starts inside a physical line"
        )
        end = start + len(element.old_text)
        assert all(
            max(start, other_start) >= min(end, other_end)
            for other_start, other_end, _ in replacements
        ), "IMPLEMENTATION: overlapping edits were accepted"
        replacements.append((start, end, element.replacement))
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text.encode()


def _partial_edit_paths(directory, targets, receipt):
    committed = [(directory / entry["path"]).resolve() for entry in receipt["files"]]
    assert committed and len(committed) < len(targets), (
        "IMPLEMENTATION: partial receipt must identify earlier committed files"
    )
    assert committed == targets[: len(committed)], (
        "IMPLEMENTATION: partial receipt reports the wrong committed files"
    )
    assert (directory / receipt["failed_path"]).resolve() == targets[len(committed)], (
        "IMPLEMENTATION: partial receipt reports the wrong failed path"
    )
    return committed


def _verify_creation_effects(document, receipt, directory, expected, directories):
    try:
        elements = CreationDocument.parse(document).elements
    except ToolInputError:
        assert receipt["ok"] is False, "IMPLEMENTATION: invalid document was accepted"
        return
    outcomes = receipt["results"]
    assert len(outcomes) == len(elements), "IMPLEMENTATION: missing creation outcomes"
    assert receipt["ok"] == all(item["ok"] for item in outcomes)
    for index, (element, item) in enumerate(zip(elements, outcomes, strict=True), 1):
        assert item["index"] == index, "IMPLEMENTATION: creation order changed"
        if isinstance(element, ToolInputError):
            assert item["ok"] is False, "IMPLEMENTATION: malformed block was accepted"
            continue
        assert item["path"] == element.path
        assert item["resource_type"] == element.resource_type
        path = (directory / element.path).resolve()
        valid_content = (
            len(element.content) >= FILE_EDITING.min_new_file_characters
            and "\x00" not in element.content
            and not any(0xD800 <= ord(char) <= 0xDFFF for char in element.content)
            and len(element.content.encode("utf-8", "surrogatepass"))
            <= FILE_EDITING.max_snapshot_bytes
        )
        if item["ok"]:
            assert expected.get(path) is None and path not in directories, (
                "IMPLEMENTATION: creation overwrote an existing resource"
            )
            if element.resource_type == "file":
                assert valid_content, (
                    "IMPLEMENTATION: invalid file content was accepted"
                )
                expected[path] = element.content.encode()
            else:
                directories.add(path)
        elif element.resource_type == "file" and path in expected:
            valid = (
                expected[path] is None
                and valid_content
                and (path.parent in directories or path.parent.is_dir())
            )
            assert not valid, f"IMPLEMENTATION: valid new-file call failed: {item!r}"


def verify_file_effects(history, directory, initial):
    """Check actual writes against emitted arguments before judging user intent."""
    expected = {path.resolve(): content for path, content in initial.items()}
    directories = set()
    calls = {}
    seen_results = set()
    successful = []
    for message in history:
        if isinstance(message, AssistantMessage):
            calls.update(
                {call.get_id(): call for call in message.get_tool_calls() or []}
            )
        if not isinstance(message, ToolMessage):
            continue
        call = calls.get(message.get_tool_call_id())
        if call is None or call.get_name() not in {
            "new-file",
            "edit-file",
            "delete-file",
        }:
            continue
        if call.get_id() in seen_results:
            continue
        seen_results.add(call.get_id())
        receipt = parse_tool_result(message.get_message())
        arguments = call.get_arguments()
        if call.get_name() == "new-file":
            _verify_creation_effects(
                arguments["document"], receipt, directory, expected, directories
            )
            if receipt["ok"]:
                successful.append(call)
            continue
        if call.get_name() == "delete-file":
            outcomes = receipt["results"]
            assert [item["path"] for item in outcomes] == arguments["paths"], (
                "IMPLEMENTATION: deletion receipt must report every input path in order"
            )
            assert receipt["ok"] == all(item["ok"] for item in outcomes)
            for item in outcomes:
                path = (directory / item["path"]).resolve()
                if item["ok"]:
                    for target in list(expected):
                        if target == path or path in target.parents:
                            expected[target] = None
                    directories.difference_update(
                        target
                        for target in tuple(directories)
                        if target == path or path in target.parents
                    )
                elif path in expected:
                    assert expected[path] is None, (
                        f"IMPLEMENTATION: valid delete-file call failed: {item!r}"
                    )
            if receipt["ok"]:
                successful.append(call)
            continue
        partial = (
            isinstance(receipt, dict)
            and receipt.get("ok") is False
            and receipt.get("error_kind") == "partial_file_edit"
            and call.get_name() == "edit-file"
        )
        if not isinstance(receipt, dict) or (
            receipt.get("ok") is not True and not partial
        ):
            continue
        if call.get_name() == "edit-file":
            document = EditDocument.parse(arguments["edits"])
            groups = {}
            for element in document.elements:
                groups.setdefault((directory / element.path).resolve(), []).append(
                    element
                )
            changes = {
                path: _expected_edit_bytes(expected[path], elements)
                for path, elements in groups.items()
            }
            committed = (
                _partial_edit_paths(directory, list(groups), receipt)
                if partial
                else list(groups)
            )
            expected.update({path: changes[path] for path in committed})
        if not partial:
            successful.append(call)
    for path in directories:
        assert path.is_dir(), f"IMPLEMENTATION: created directory is missing: {path}"
    for path, content in expected.items():
        if content is None:
            assert not path.exists(), (
                "IMPLEMENTATION: delete did not match its successful receipt"
            )
        else:
            assert path.exists() and path.read_bytes() == content, (
                f"IMPLEMENTATION: file bytes differ from executed arguments: {path}"
            )
    return successful


def test_live_model_creates_mixed_resource_batch(tmp_path, edit_provider):
    document = (
        '<new-exmekh-dir path="reports"/>ignored'
        '<new-exmekh-file path="reports/first.txt"/>FIRST\r\n'
        '<new-exmekh-file path="reports/second.txt"/>SECOND&<data/>'
    )
    _, receipt = execute_live_file_call(
        edit_provider,
        tmp_path,
        "new-file",
        {"document": document},
        {tmp_path / "reports/first.txt": None, tmp_path / "reports/second.txt": None},
    )
    assert [item["resource_type"] for item in receipt["results"]] == [
        "directory",
        "file",
        "file",
    ], "COVERAGE: mixed resource creation was not exercised"
    assert receipt["results"][0].get("warnings"), (
        "COVERAGE: ignored directory content was not exercised"
    )
    assert not edit_provider.exchanges


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
    arguments = (
        {"paths": [path.name]}
        if tool == "delete-file"
        else {"document": f'<new-exmekh-file path="{path.name}"/>' + expected.decode()}
        if tool == "new-file"
        else {
            "edits": edit_fragment(
                path.name,
                {
                    "edit": "  old value",
                    "empty": initial.decode(),
                    "insert": "KEEP",
                    "empty_existing": "",
                }[operation],
                {
                    "edit": "  updated value",
                    "empty": "",
                    "insert": "INSERT\r\nKEEP",
                    "empty_existing": "FILLED",
                }[operation],
            )
        }
    )
    call, receipt = execute_live_file_call(
        edit_provider, tmp_path, tool, arguments, {path: initial if existing else None}
    )
    if tool != "edit-file":
        assert not edit_provider.exchanges
        if operation == "literal_markup":
            (element,) = CreationDocument.parse(
                call.get_arguments()["document"]
            ).elements
            assert not isinstance(element, ToolInputError)
            assert any(character in element.content for character in "<&"), (
                "COVERAGE: literal markup characters were not exercised"
            )
    else:
        assert len(edit_provider.exchanges) == 1
        assert receipt["files"][0]["size_bytes"] == len(path.read_bytes())
        (element,) = EditDocument.parse(call.get_arguments()["edits"]).elements
        if operation == "empty":
            assert element.replacement == "", (
                "COVERAGE: empty replacement was not exercised"
            )
        if operation == "empty_existing":
            assert element.old_text == "", (
                "COVERAGE: editing an empty file was not exercised"
            )


def test_live_batch_deletes_files_and_directory_trees(tmp_path, edit_provider):
    binary = tmp_path / "binary.dat"
    binary.write_bytes(b"\x00\xff")
    child = tmp_path / "folder" / "nested" / "keep.txt"
    child.parent.mkdir(parents=True)
    child.write_bytes(b"REMOVE")
    call, receipt = execute_live_file_call(
        edit_provider,
        tmp_path,
        "delete-file",
        {"paths": ["binary.dat", "folder"]},
        {binary: b"\x00\xff", child: b"REMOVE"},
    )
    if call.get_arguments() != {"paths": ["binary.dat", "folder"]}:
        pytest.skip(
            "MODEL_LIMITATION: model did not select both requested deletion paths"
        )
    assert [item["path"] for item in receipt["results"]] == ["binary.dat", "folder"]
    assert not binary.exists() and not (tmp_path / "folder").exists()


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
    initial = {first: b"FIRST\r\nSECOND\n"}
    if not alias:
        initial[second] = b"OTHER\n"
    call, receipt = execute_live_file_call(
        edit_provider, tmp_path, "edit-file", {"edits": document}, initial
    )
    trace = call.get_arguments()
    assert len(EditDocument.parse(trace["edits"]).elements) == 2, (
        "COVERAGE: multi-edit call did not contain both resources"
    )
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
    assert len(receipt["files"]) == (1 if alias else 2)


class AdversarialSnapshot:
    def __init__(self, text, deadline):
        self.text = text
        self.deadline = deadline
        self.exchanges = 0

    def begin(self):
        pass

    def prepare(self, event, definition):
        self.exchanges += 1
        values = event["request"]["values"]
        assert set(values) == {"path"}
        assert isinstance(values["path"], str) and values["path"]
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
                    _expected_edit_bytes(self.text.encode(), (element,))
                except AssertionError as error:
                    pytest.fail(
                        f"Server literal constraints escaped: {element!r}: {error}"
                    )


def test_live_adversarial_literal_selection_cannot_escape_for_30_seconds(
    tmp_path, edit_provider, monkeypatch, record_property
):
    edit = file_tools(tmp_path)["edit-file"]
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
    assert edit_provider.client.models.list().data
    assert not (tmp_path / "attack.txt").exists()
    assert completed > 0, (
        "COVERAGE: no complete constrained call emitted; adversarial selections not verified"
    )
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
    inspection = inspect_file(
        tmp_path, first.name, tokenizer=edit_provider.get_tokenizer()
    )
    with file_read_context((inspection.evidence,)):
        reading = read_file(tmp_path, first.name, 1, 1)
    edit = file_tools(tmp_path)["edit-file"]
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
        ("read-file", reading, {"path": first.name, "start_index": 1, "end_index": 1}),
    ):
        call = ToolCall(name, FunctionCall(name, arguments))
        message = ToolMessage(stringify_tool_result(report), name, name)
        history.extend([AssistantMessage("", tool_calls=[call]), message])
        runtime.record_result(call, report, message)
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
    assert (tmp_path / "existing.txt").read_bytes() == b"KEEP"
    assert first.read_bytes() == b"FIRST"
    assert not result.get_tool_calls()
    assert len(exchanges) == 2, (
        "COVERAGE: second edit was not reached; rejection boundary not verified"
    )
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
    edit = file_tools(tmp_path)["edit-file"]
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
    if not pauses:
        assert not result.get_tool_calls(), (
            "IMPLEMENTATION: edit returned without its snapshot pause"
        )
    assert len(pauses) == 1, (
        "COVERAGE: edit pause not reached exactly once; scope cancellation not verified"
    )
    assert not exchanges
    assert not result.get_tool_calls()
    assert any(record["type"] == "interactive.cancelled" for record in records)
    assert handler.cancellation_requested()
    assert not (tmp_path / "notes.txt").exists()
    record_property("cancelled_pauses", len(pauses))
