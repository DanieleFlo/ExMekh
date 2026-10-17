"""Model intent must never hide a mismatch between tool arguments and writes."""

import pytest

from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.edit_document import EditDocument
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.providers.local import LocalProvider
from exmekh_lib.utils.xml import stringify_tool_result
from tests.core.agents.node.test_node import SequenceProvider
from tests.history_validation import validate_request_history
from tests.interactive_support import edit_fragment
from tests.live.model_support import CacheTestTokenizer
from tests.live.test_file_editing_live import (
    AdversarialSnapshot,
    EditingLiveProvider,
    execute_live_file_call,
    verify_file_effects,
)
from tests.live.test_file_editing_live import (
    test_live_model_applies_literal_file_operations as run_literal_file_case,
)
from tests.live.test_file_editing_live import (
    test_live_model_creates_mixed_resource_batch as run_mixed_creation_case,
)
from tests.live.test_file_reading_live import (
    test_live_client_reads_files_and_identifies_their_contents as run_live_read_case,
)


class FileFixtureProvider:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments
        self.requests = []
        self.exchanges = []

    def call(self, request):
        validate_request_history(request)
        self.requests.append(request)
        handler = request.interactive_handler
        handler.begin()
        if self.name == "edit-file":
            for index, element in enumerate(
                EditDocument.parse(self.arguments["edits"]).elements, 1
            ):
                event = {
                    "call_id": "observed-call",
                    "request": {"edit_index": index, "values": {"path": element.path}},
                }
                reply = handler.prepare(event, {"behavior": "edit_file.v3"})
                self.exchanges.append((event, reply))
                assert reply["status"] == "provide"
        calls = [ToolCall("observed-call", FunctionCall(self.name, self.arguments))]
        handler.validate_completed(calls)
        return ProviderResponse(AssistantMessage("", tool_calls=calls))


def history(name, arguments, receipt):
    return [
        AssistantMessage(
            "", tool_calls=[ToolCall("call", FunctionCall(name, arguments))]
        ),
        ToolMessage(stringify_tool_result(receipt), "call", name),
    ]


@pytest.mark.parametrize("name", ["new-file", "edit-file", "delete-file"])
def test_observed_effects_validate_emitted_arguments_before_user_intent(tmp_path, name):
    path = tmp_path / "notes.txt"
    initial = None if name == "new-file" else b"OLD"
    arguments = (
        {"edits": edit_fragment("notes.txt", "OLD", "MODEL_CHOICE")}
        if name == "edit-file"
        else {"paths": ["notes.txt"]}
        if name == "delete-file"
        else {"document": '<new-exmekh-file path="notes.txt"/>MODEL_CHOICE'}
    )
    receipt = (
        {"ok": True, "results": [{"path": "notes.txt", "ok": True}]}
        if name == "delete-file"
        else {
            "ok": True,
            "results": [
                {"index": 1, "path": "notes.txt", "resource_type": "file", "ok": True}
            ],
        }
        if name == "new-file"
        else {"ok": True}
    )
    if name != "delete-file":
        path.write_bytes(b"MODEL_CHOICE")
    calls = verify_file_effects(
        history(name, arguments, receipt), tmp_path, {path: initial}
    )
    assert len(calls) == 1
    path.write_bytes(b"WRONG_ENGINE_RESULT")
    with pytest.raises(AssertionError, match="IMPLEMENTATION"):
        verify_file_effects(
            history(name, arguments, receipt), tmp_path, {path: initial}
        )


def test_rejected_call_cannot_change_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"OLD")
    trace = history(
        "new-file",
        {"document": '<new-exmekh-file path="notes.txt"/>MODEL_CHOICE'},
        {
            "ok": False,
            "results": [
                {"index": 1, "path": "notes.txt", "resource_type": "file", "ok": False}
            ],
        },
    )
    assert verify_file_effects(trace, tmp_path, {path: b"OLD"}) == []
    path.write_bytes(b"CHANGED")
    with pytest.raises(AssertionError, match="IMPLEMENTATION"):
        verify_file_effects(trace, tmp_path, {path: b"OLD"})


def test_alias_edits_use_one_initial_snapshot(tmp_path):
    path = tmp_path / "notes.txt"
    alias = tmp_path / "alias.txt"
    alias.symlink_to(path)
    path.write_bytes(b"FIRST\nSECOND")
    document = edit_fragment(path.name, "FIRST", "ONE") + edit_fragment(
        alias.name, "SECOND", "TWO"
    )
    path.write_bytes(b"ONE\nTWO")
    verify_file_effects(
        history("edit-file", {"edits": document}, {"ok": True}),
        tmp_path,
        {path: b"FIRST\nSECOND"},
    )


def test_valid_creation_failure_is_not_a_model_limitation(tmp_path):
    path = tmp_path / "notes.txt"
    trace = history(
        "new-file",
        {"document": '<new-exmekh-file path="notes.txt"/>VALID'},
        {
            "ok": False,
            "results": [
                {"index": 1, "path": "notes.txt", "resource_type": "file", "ok": False}
            ],
        },
    )
    with pytest.raises(AssertionError, match="valid new-file call failed"):
        verify_file_effects(trace, tmp_path, {path: None})


def test_replayed_history_does_not_apply_the_same_call_twice(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"NEW")
    trace = history(
        "edit-file", {"edits": edit_fragment(path.name, "OLD", "NEW")}, {"ok": True}
    )
    assert len(verify_file_effects(trace + trace, tmp_path, {path: b"OLD"})) == 1


@pytest.mark.parametrize(
    ("initial", "old", "actual"),
    [(b"aaa", "aa", b"NEWa"), (b"prefix OLD", "OLD", b"prefix NEW")],
)
def test_successful_receipt_cannot_hide_an_invalid_literal_selection(
    tmp_path, initial, old, actual
):
    path = tmp_path / "notes.txt"
    path.write_bytes(actual)
    trace = history(
        "edit-file", {"edits": edit_fragment(path.name, old, "NEW")}, {"ok": True}
    )
    with pytest.raises(AssertionError, match="IMPLEMENTATION"):
        verify_file_effects(trace, tmp_path, {path: initial})


def test_successful_receipt_cannot_hide_overlapping_edits(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"ONE")
    document = edit_fragment(path.name, "HEAD\nTAIL", "ALL") + edit_fragment(
        path.name, "HEAD", "ONE"
    )
    with pytest.raises(AssertionError, match="IMPLEMENTATION"):
        verify_file_effects(
            history("edit-file", {"edits": document}, {"ok": True}),
            tmp_path,
            {path: b"HEAD\nTAIL"},
        )


def test_successful_receipt_cannot_hide_multiple_empty_selections(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"ONETWO")
    document = edit_fragment(path.name, "", "ONE") + edit_fragment(path.name, "", "TWO")
    with pytest.raises(AssertionError, match="IMPLEMENTATION"):
        verify_file_effects(
            history("edit-file", {"edits": document}, {"ok": True}),
            tmp_path,
            {path: b""},
        )


@pytest.mark.parametrize("unreported_write", [False, True])
def test_partial_receipt_accounts_for_only_the_committed_files(
    tmp_path, unreported_write
):
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_bytes(b"NEW")
    second.write_bytes(b"NEW" if unreported_write else b"OLD")
    document = edit_fragment(first.name, "OLD", "NEW") + edit_fragment(
        second.name, "OLD", "NEW"
    )
    trace = history(
        "edit-file",
        {"edits": document},
        {
            "ok": False,
            "error_kind": "partial_file_edit",
            "failed_path": second.name,
            "files": [{"path": first.name, "operation": "edit", "size_bytes": 3}],
        },
    )
    if unreported_write:
        with pytest.raises(AssertionError, match="IMPLEMENTATION"):
            verify_file_effects(trace, tmp_path, {first: b"OLD", second: b"OLD"})
    else:
        assert (
            verify_file_effects(trace, tmp_path, {first: b"OLD", second: b"OLD"}) == []
        )


@pytest.mark.parametrize("tool_name", ["new-file", "edit-file", "delete-file"])
def test_live_file_fixture_executes_observed_arguments_once(tmp_path, tool_name):
    path = tmp_path / "notes.txt"
    initial = None if tool_name == "new-file" else b"OLD"
    if initial is not None:
        path.write_bytes(initial)
    requested = (
        {"edits": edit_fragment(path.name, "OLD", "REQUESTED")}
        if tool_name == "edit-file"
        else {"document": f'<new-exmekh-file path="{path.name}"/>REQUESTED'}
        if tool_name == "new-file"
        else {"paths": [path.name]}
    )
    observed = (
        {"edits": edit_fragment(path.name, "OLD", "OBSERVED")}
        if tool_name == "edit-file"
        else {"document": f'<new-exmekh-file path="{path.name}"/>OBSERVED'}
        if tool_name == "new-file"
        else {"paths": [path.name]}
    )

    provider = FileFixtureProvider(tool_name, observed)
    call, receipt = execute_live_file_call(
        provider, tmp_path, tool_name, requested, {path: initial}
    )
    assert len(provider.requests) == 1
    assert call.get_arguments() == observed
    assert receipt["ok"] is True
    assert (path.read_bytes() if path.exists() else None) == (
        None if tool_name == "delete-file" else b"OBSERVED"
    )


@pytest.mark.parametrize("operation", ["empty", "literal_markup"])
def test_live_case_does_not_mistake_model_choice_for_wrong_file_effects(
    tmp_path, operation
):
    provider = FileFixtureProvider(
        "edit-file" if operation == "empty" else "new-file",
        {"edits": edit_fragment("notes.txt", "  old value\r\nKEEP", "")}
        if operation == "empty"
        else {"document": '<new-exmekh-file path="notes.txt"/>&lt;MODEL_CHOICE&gt;'},
    )
    run_literal_file_case(tmp_path, provider, operation)
    assert len(provider.requests) == 1
    assert (tmp_path / "notes.txt").read_bytes() == (
        b"\n" if operation == "empty" else b"&lt;MODEL_CHOICE&gt;"
    )


def test_live_batch_fixture_verifies_directory_and_file_effects(tmp_path):
    document = (
        '<new-exmekh-dir path="reports"/>ignored'
        '<new-exmekh-file path="reports/first.txt"/>FIRST\r\n'
        '<new-exmekh-file path="reports/second.txt"/>SECOND&<data/>'
    )
    provider = FileFixtureProvider("new-file", {"document": document})
    run_mixed_creation_case(tmp_path, provider)
    assert len(provider.requests) == 1
    assert (tmp_path / "reports/first.txt").read_bytes() == b"FIRST\r\n"
    assert (tmp_path / "reports/second.txt").read_bytes() == b"SECOND&<data/>"


def test_live_provider_records_the_real_response_without_extra_inference(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        DEFAULT_PROVIDER_SETTINGS.base_url_env_var, "http://127.0.0.1:1/v1"
    )
    monkeypatch.setenv(DEFAULT_PROVIDER_SETTINGS.api_key_env_var, "offline-test-key")
    monkeypatch.setenv("EXMEKH_LIB_LIVE_MODEL", "offline-test-model")
    monkeypatch.setattr(
        LocalProvider, "_get_available_models", lambda self: {"offline-test-model"}
    )
    seen = []

    def call(self, request):
        seen.append(request)
        return ProviderResponse(
            AssistantMessage(
                "", tool_calls=[ToolCall("repeat", FunctionCall("file-processing", {}))]
            )
        )

    monkeypatch.setattr(LocalProvider, "call", call)
    provider = EditingLiveProvider(tmp_path)
    path = tmp_path / "notes.txt"
    path.write_bytes(b"CREATED")
    completed = history(
        "new-file",
        {"document": f'<new-exmekh-file path="{path.name}"/>CREATED'},
        {
            "ok": True,
            "results": [
                {"index": 1, "path": path.name, "resource_type": "file", "ok": True}
            ],
        },
    )
    completed.append(ToolMessage("Complete", "skill", "file-processing"))
    try:
        response = provider.call(
            ProviderRequest(completed, extra={"agentic": {"agent_context_id": "test"}})
        )
        assert len(seen) == 1
        assert response.get_tool_calls()[0].get_id() == "repeat"
        assert seen[0].get_extra()["agentic"] == {"agent_context_id": "test"}
        assert path.read_bytes() == b"CREATED"
        assert (tmp_path / "live-request-01.json").exists()
    finally:
        provider.get_session().close()
        provider.client.close()


@pytest.mark.parametrize("kind", ["text", "image", "pdf"])
def test_live_read_fixture_checks_delivery_without_a_skill_strategy(tmp_path, kind):
    path, start, end = {
        "text": ("notes.txt", 2, 2),
        "image": ("shape.png", 0, 0),
        "pdf": ("shapes.pdf", 2, 3),
    }[kind]
    responses = []
    if kind != "image":
        responses.append(
            AssistantMessage(
                "",
                tool_calls=[
                    ToolCall("inspect", FunctionCall("inspect-file", {"path": path}))
                ],
            )
        )
    responses.extend(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    ToolCall(
                        "read",
                        FunctionCall(
                            "read-file",
                            {"path": path, "start_index": start, "end_index": end},
                        ),
                    )
                ],
            ),
            AssistantMessage("DONE"),
        ]
    )
    provider = SequenceProvider(
        responses, CacheTestTokenizer("live-read-fixture", tmp_path / "cache")
    )
    try:
        run_live_read_case(tmp_path, provider, kind)
        assert len(provider.requests) == (2 if kind == "image" else 3)
    finally:
        provider.get_session().close()


def test_optional_image_inspection_does_not_invalidate_a_correct_read(tmp_path):
    provider = SequenceProvider(
        [
            AssistantMessage(
                "",
                tool_calls=[
                    ToolCall(
                        "inspect", FunctionCall("inspect-file", {"path": "shape.png"})
                    )
                ],
            ),
            AssistantMessage(
                "",
                tool_calls=[
                    ToolCall(
                        "read",
                        FunctionCall(
                            "read-file",
                            {"path": "shape.png", "start_index": 0, "end_index": 0},
                        ),
                    )
                ],
            ),
            AssistantMessage("DONE"),
        ],
        CacheTestTokenizer("live-image-inspection", tmp_path / "cache"),
    )
    try:
        run_live_read_case(tmp_path, provider, "image")
        assert len(provider.requests) == 3
    finally:
        provider.get_session().close()


@pytest.mark.parametrize("old", ["aa", "value", "refix", ""])
def test_adversarial_snapshot_uses_an_independent_literal_oracle(old):
    snapshot = AdversarialSnapshot("  value\nprefix value\naaa\nvalue", 0)
    with pytest.raises(
        pytest.fail.Exception, match="Server literal constraints escaped"
    ):
        snapshot.validate_completed(
            [
                ToolCall(
                    "attack",
                    FunctionCall(
                        "edit-file",
                        {"edits": edit_fragment("attack.txt", old, "ATTACK")},
                    ),
                )
            ]
        )
