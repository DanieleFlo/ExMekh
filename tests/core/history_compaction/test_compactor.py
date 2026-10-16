import pytest

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.config.tool_output import ToolOutputSettings
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.cache.session_cache import SessionCache
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.history_compaction import HistoryCompactor
from exmekh_lib.core.history_compaction.models import compaction_scores_model
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.external_tool_output import (
    ExternalToolOutput,
    OutputTextFile,
)
from exmekh_lib.core.models.history import MessageHistory
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import PreformattedToolResult, Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    UserMessage,
)
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.core.tool_calls.output_preparer import ToolOutputPreparer
from exmekh_lib.utils.json import serialize_json
from tests.core.history_compaction.support import (
    CompactionProvider,
    UnvalidatedScores,
    compacted_payload,
    output_batch,
)
from tests.history_validation import validate_request_history


def history_with_output(body, **kwargs):
    return MessageHistory(
        [SystemMessage("system"), UserMessage("task"), *output_batch(body, **kwargs)]
    )


def test_replacement_preserves_retained_input_identity_without_exposing_mapping_rules():
    history = history_with_output("original body")
    original = history.get_history()
    checkpoint = MessageHistory(history.get_copy_of_history())
    result = HistoryCompactor(CompactionProvider()).compact(checkpoint)
    replacement = result.replacement_for(original)
    assert replacement[0] is original[0]
    assert replacement[1] is original[1]
    assert replacement[2] is not original[2]
    with pytest.raises(ValueError, match=HISTORY_COMPACTION.changed_history):
        result.replacement_for(original[:-1])


@pytest.mark.parametrize(
    ("score", "length", "representation"),
    [
        (1, 100, "reference"),
        (9, 100, "reference"),
        (10, 10, "summary"),
        (49, 10, "summary"),
        (50, 99, "original"),
        (79, 99, "original"),
        (50, 100, "summary"),
        (79, 100, "summary"),
        (80, 199, "original"),
        (94, 199, "original"),
        (80, 200, "summary"),
        (94, 200, "summary"),
        (95, 4999, "original"),
        (100, 4999, "original"),
        (95, 5000, "summary"),
        (100, 5000, "summary"),
    ],
)
def test_retention_boundaries_use_real_tokens_and_never_truncate_summaries(
    score, length, representation
):
    original = "a" * length
    summary = ("long summary " * 700).strip()
    provider = CompactionProvider(score=score, summary=summary)
    history = history_with_output(original)
    before = history.dump_history()
    result = HistoryCompactor(provider).compact(history)
    output = compacted_payload(result.history.get_history()[-1])["outputs"][0]
    assert output["representation"] == representation
    assert output["id"] == "output"
    assert output["name"] == "work"
    assert output["score"] == score
    assert (
        output.get("content")
        == {"reference": None, "original": original, "summary": summary}[representation]
    )
    assert history.dump_history() == before
    for request in provider.summary_requests:
        assert request.get_reasoning_effort_override() == "none"
        assert request.get_tools() is None
        assert request.get_extra() is None
        assert len(request.get_messages()) == 2


@pytest.mark.parametrize("length", [0, 10, 39, 40, 41, 100])
def test_scoring_appends_catalog_and_preserves_original_system_tools_and_history(
    length,
):
    body = "".join(chr(0x400 + index) for index in range(length))
    provider = CompactionProvider(score=9)
    history = history_with_output(body)
    tools = Tools([Tool(lambda: "", name="work")])
    result = HistoryCompactor(provider).compact(history, tools=tools)
    request = provider.score_requests[0]
    assert request.get_tools() is tools
    assert request.get_tool_choice() == "none"
    assert request.get_reasoning_effort_override() == "none"
    assert [
        message.message_dump() for message in request.get_messages()[:-1]
    ] == history.dump_history()
    preview = provider.catalogs[0]["outputs"][0]["preview"]
    assert preview == (body if length <= 40 else body[:20] + body[-20:])
    assert result.output_count == 1
    assert result.original_message_count == 4
    assert result.compacted_message_count == 4


def test_user_system_and_attachments_are_untouched_and_blocks_do_not_cross_users():
    provider = CompactionProvider(score=9)
    document = DocumentUserMessage(b"exact bytes", "source", "txt")
    first_user = UserMessage(["first", FilesUserMessage([document])], name="alice")
    steering = UserMessage("corrected constraint")
    system = SystemMessage("original instructions")
    messages = [
        system,
        first_user,
        *output_batch("first body"),
        AssistantMessage("decision"),
        steering,
        *output_batch("second body", call_id="other", output_id="other-output"),
    ]
    history = MessageHistory(messages)
    result = HistoryCompactor(provider).compact(history)
    compacted = result.history.get_history()
    assert compacted[0] is system
    assert compacted[1] is first_user
    assert compacted[4] is steering
    assert compacted_payload(compacted[3])["assistant"] == provider.summary
    assert compacted_payload(compacted[3])["outputs"][0]["id"] == "output"
    assert compacted_payload(compacted[6])["outputs"][0]["id"] == "other-output"
    validate_request_history(ProviderRequest(compacted))


def test_scoring_many_outputs_fits_the_remaining_context_without_dropping_entries():
    provider = CompactionProvider(score=9)
    history = MessageHistory([SystemMessage("system"), UserMessage("task")])
    for index in range(20):
        history.add(
            output_batch(
                "evidence <tag> & details " * 4,
                call_id=f"call-{index}",
                output_id=f"output-{index}",
            )
        )
    before = history.dump_history()

    def reject_oversized_request(request):
        # The fixture tokenizer counts characters as tokens. Include the score
        # schema too: both it and the temporary catalog need context headroom.
        prompt = serialize_json(
            {
                "messages": [
                    message.message_dump() for message in request.get_messages()
                ],
                "schema": request.get_response_model().model_json_schema(),
            }
        )
        assert provider.get_tokenizer().count_tokens(prompt) <= 20000

    provider.on_score = reject_oversized_request
    result = HistoryCompactor(provider).compact(history)
    assert result.output_count == 20
    assert len(provider.catalogs[0]["outputs"]) == 20
    assert len(compacted_payload(result.history.get_history()[-1])["outputs"]) == 20
    assert history.dump_history() == before


def test_reused_call_and_output_ids_have_independent_scores_per_message():
    provider = CompactionProvider(score=9)
    provider.score_transform = lambda scores: {
        **scores,
        list(scores)[1]: 60,
    }
    history = history_with_output("first")
    history.add(output_batch("different snapshot"))
    result = HistoryCompactor(provider).compact(history)
    entries = provider.catalogs[0]["outputs"]
    assert entries[0]["entry_id"] != entries[1]["entry_id"]
    outputs = compacted_payload(result.history.get_history()[-1])["outputs"]
    assert [item["representation"] for item in outputs] == ["reference", "original"]


@pytest.mark.parametrize("kind", ["missing", "extra", "unknown"])
def test_invalid_score_coverage_does_not_change_history(kind):
    provider = CompactionProvider()
    provider.score_transform = {
        "missing": lambda scores: {},
        "extra": lambda scores: {**scores, "unknown": 50},
        "unknown": lambda scores: {"unknown": 50},
    }[kind]
    history = history_with_output("original")
    before = history.dump_history()
    with pytest.raises(ValueError, match="every entry exactly once"):
        HistoryCompactor(provider).compact(history)
    assert history.dump_history() == before
    assert provider.summary_requests == []


def test_summary_failure_leaves_original_history_available():
    provider = CompactionProvider(score=10)
    provider.summary_error = RuntimeError("summary failed")
    history = history_with_output("original")
    before = history.dump_history()
    with pytest.raises(RuntimeError, match="summary failed"):
        HistoryCompactor(provider).compact(history)
    assert history.dump_history() == before


@pytest.mark.parametrize(
    "messages",
    [
        [ToolMessage("orphan", "orphan", "work")],
        [
            AssistantMessage(
                None, tool_calls=[ToolCall("call", FunctionCall("work", {}))]
            )
        ],
        [*output_batch("result")[:-1], UserMessage("interrupt")],
        [*output_batch("result")[:-1], ToolMessage("wrong", "call", "wrong-name")],
        [*output_batch("result"), ToolMessage("duplicate", "call", "work")],
        [
            AssistantMessage(
                None, tool_calls=[ToolCall("call", FunctionCall("work", {}))] * 2
            )
        ],
    ],
)
def test_unfinished_or_invalid_tool_batches_are_rejected_before_inference(messages):
    provider = CompactionProvider()
    history = MessageHistory([SystemMessage("system"), UserMessage("task"), *messages])
    with pytest.raises(ValueError, match="matching tool batches"):
        HistoryCompactor(provider).compact(history)
    assert provider.score_requests == []


def test_protected_skill_batch_remains_exact_and_anchors_are_remapped():
    provider = CompactionProvider(score=9)
    before = [*output_batch("old one"), *output_batch("old two", call_id="two")]
    opening = output_batch("ACTIVE SKILL INSTRUCTIONS", call_id="skill", name="skill")
    messages = [
        SystemMessage("system"),
        UserMessage("task"),
        *before,
        *opening,
        *output_batch("inner", call_id="inner"),
    ]
    history = MessageHistory(messages)
    result = HistoryCompactor(provider).compact(history, protected_indices=(6,))
    compacted = result.history.get_history()
    assert result.index_map[6] == 4
    assert compacted[4] is opening[0]
    assert compacted[5] is opening[1]
    assert "skill" not in [item["name"] for item in provider.catalogs[0]["outputs"]]
    validate_request_history(ProviderRequest(compacted))


@pytest.mark.parametrize("anchor", [-1, 999, True])
def test_invalid_protection_anchor_is_rejected(anchor):
    with pytest.raises(ValueError, match="protection index"):
        HistoryCompactor(CompactionProvider()).compact(
            history_with_output("a"), protected_indices=(anchor,)
        )


def test_existing_compaction_is_idempotent_and_does_not_resummarize():
    provider = CompactionProvider(score=9)
    compactor = HistoryCompactor(provider)
    first = compactor.compact(history_with_output("original"))
    second = compactor.compact(first.history)
    assert second.history.dump_history() == first.history.dump_history()
    assert second.output_count == 0
    assert len(provider.score_requests) == 1


def test_tool_attachments_survive_projection_in_the_provider_request():
    attachment = UserMessage("attached output")
    provider = CompactionProvider(score=9)
    result = HistoryCompactor(provider).compact(
        history_with_output("body", attachments=attachment)
    )
    request = ProviderRequest(result.history.get_history())
    assert request.get_messages()[-1].get_message() == ["attached output"]
    validate_request_history(request)


@pytest.mark.parametrize(
    ("score", "expected"), [(9, "reference"), (60, "original"), (10, "summary")]
)
def test_file_backed_outputs_use_original_text_and_keep_files_without_cache_copy(
    tmp_path, score, expected
):
    provider = CompactionProvider(score=score)
    with SessionCache(tmp_path / "cache") as session:
        cache = NodeCache("node", session)
        files = OutputFileWorkspace(
            lambda: cache, tmp_path, ToolOutputSettings(max_tokens=5)
        )
        preparer = ToolOutputPreparer(provider.get_tokenizer(), files)
        body = "start\nIMPORTANT MIDDLE\nend\r\n"
        prepared = preparer.prepare(
            ToolOutputPresentation(
                ToolOutputReference("work", "output"), PreformattedToolResult(body)
            )
        )
        external = prepared.external_output
        assert external is not None
        history = history_with_output(external.render(), external_output=external)
        result = HistoryCompactor(provider, files=files).compact(history)
        output = compacted_payload(result.history.get_history()[-1])["outputs"][0]
        assert output["representation"] == expected
        assert output["files"]["files"][0]["path"] == external.files[0].path
        if score == 10:
            assert provider.summary_requests[0].get_messages()[1].get_message() == body
        assert cache.get_all() == []


def test_cached_file_references_support_restored_messages_and_fail_on_modified_files(
    tmp_path,
):
    provider = CompactionProvider(score=10)
    with SessionCache(tmp_path / "cache") as session:
        cache = NodeCache("node", session)
        files = OutputFileWorkspace(
            lambda: cache, tmp_path, ToolOutputSettings(max_tokens=5)
        )
        prepared = ToolOutputPreparer(provider.get_tokenizer(), files).prepare(
            ToolOutputPresentation(
                ToolOutputReference("work", "output"),
                PreformattedToolResult("long body"),
            )
        )
        reference = prepared.external_output
        assert reference is not None
        record = ToolCallRecord("work", "output", {}, None, external_output=reference)
        cache.store(record)
        compactor = HistoryCompactor(provider, files=files, output_lookup=cache.get)
        history = history_with_output(reference.render())
        compactor.compact(history)
        (tmp_path / reference.files[0].path).write_text("changed")
        before = history.dump_history()
        with pytest.raises(ValueError):
            compactor.compact(history)
        assert history.dump_history() == before


def test_newer_cached_output_does_not_replace_visible_running_snapshot(tmp_path):
    provider = CompactionProvider(score=60)
    with SessionCache(tmp_path / "cache") as session:
        cache = NodeCache("node", session)
        files = OutputFileWorkspace(
            lambda: cache, tmp_path, ToolOutputSettings(max_tokens=5)
        )
        external = (
            ToolOutputPreparer(provider.get_tokenizer(), files)
            .prepare(
                ToolOutputPresentation(
                    ToolOutputReference("work", "output"),
                    PreformattedToolResult("final unseen"),
                )
            )
            .external_output
        )
        cache.store(
            ToolCallRecord("work", "output", {}, None, external_output=external)
        )
        result = HistoryCompactor(
            provider, files=files, output_lookup=cache.get
        ).compact(history_with_output("running"))
        assert (
            compacted_payload(result.history.get_history()[-1])["outputs"][0]["content"]
            == "running"
        )


def test_assistant_only_history_is_compacted_without_scoring():
    provider = CompactionProvider()
    history = MessageHistory(
        [SystemMessage("system"), UserMessage("task"), AssistantMessage("decision")]
    )
    result = HistoryCompactor(provider).compact(history)
    assert (
        compacted_payload(result.history.get_history()[-1])["assistant"]
        == provider.summary
    )
    assert provider.score_requests == []


def test_header_fields_round_trip_control_characters():
    header = XmlHeader("Tool", "name\x00", "id\x01")
    content = header.prepend_to("body")
    assert XmlHeader.field_from_content(content, "id") == "id\x01"
    assert XmlHeader.field_from_content(content, "name") == "name\x00"
    assert XmlHeader.field_from_content(content, "missing") is None
    assert XmlHeader.field_from_content("plain", "id") is None


def test_declared_descendants_are_scored_once_and_unrelated_cache_entries_are_ignored(
    tmp_path,
):
    provider = CompactionProvider(score=60)
    with SessionCache(tmp_path / "cache") as session:
        cache = NodeCache("node", session)
        cache.store(
            ToolCallRecord(
                "skill",
                "output",
                {},
                {"text": "skill result"},
                source="DeepSkill",
                used_tools=(ToolOutputReference("child", "child-output"),),
            )
        )
        cache.store(
            ToolCallRecord(
                "child",
                "child-output",
                {},
                "child result",
                used_tools=(
                    ToolOutputReference("skill", "output"),
                    ToolOutputReference("missing", "missing-output"),
                    ToolOutputReference("steering", "saved-user"),
                ),
            )
        )
        cache.store(
            ToolCallRecord(
                "steering", "saved-user", {}, "exact user", source="UserMessage"
            )
        )
        cache.store(
            ToolCallRecord("unrelated", "unrelated-output", {}, "not part of history")
        )
        history = history_with_output("skill result", name="skill")
        result = HistoryCompactor(provider, output_lookup=cache.get).compact(history)
        catalog = provider.catalogs[0]["outputs"]
        assert [item["output_id"] for item in catalog] == [
            "output",
            "child-output",
            "missing-output",
        ]
        assert catalog[1]["visibility"] == "reference"
        assert catalog[2]["available"] is False
        outputs = compacted_payload(result.history.get_history()[-1])["outputs"]
        assert outputs[0]["used-tools"] == [{"name": "child", "id": "child-output"}]
        assert outputs[1]["content"] == "child result"
        assert outputs[2]["representation"] == "reference"
        assert result.output_count == 3


def test_declared_uncached_references_are_preserved_from_the_visible_header():
    provider = CompactionProvider(score=100)
    header = XmlHeader(
        "DeepSkill",
        "skill",
        "output",
        fields={"used-tools": [{"name": "uncached", "id": "lost-output"}]},
    )
    messages = output_batch("ignored", name="skill")
    messages[-1] = ToolMessage(header.prepend_to("skill result"), "call", "skill")
    history = MessageHistory([SystemMessage("system"), UserMessage("task"), *messages])
    result = HistoryCompactor(provider).compact(history)
    outputs = compacted_payload(result.history.get_history()[-1])["outputs"]
    assert outputs[1]["id"] == "lost-output"
    assert outputs[1]["recoverable"] is False
    assert outputs[1]["representation"] == "reference"


def test_scoring_cannot_produce_executable_tool_calls():
    provider = CompactionProvider()
    provider.call_structured = lambda request: StructuredProviderResponse(
        StructuredAssistantMessage(
            UnvalidatedScores.model_validate({"entry_3": 50}),
            tool_calls=[ToolCall("unexpected", FunctionCall("work", {}))],
        )
    )
    with pytest.raises(ValueError, match="must not return tool calls"):
        HistoryCompactor(provider).compact(history_with_output("body"))


def test_registered_output_files_cannot_be_read_through_another_node(tmp_path):
    provider = CompactionProvider()
    with SessionCache(tmp_path / "cache") as session:
        first = NodeCache("first", session)
        second = NodeCache("second", session)
        first_files = OutputFileWorkspace(
            lambda: first, tmp_path, ToolOutputSettings(max_tokens=5)
        )
        second_files = OutputFileWorkspace(
            lambda: second, tmp_path, ToolOutputSettings(max_tokens=5)
        )
        reference = (
            ToolOutputPreparer(provider.get_tokenizer(), first_files)
            .prepare(
                ToolOutputPresentation(
                    ToolOutputReference("work", "output"),
                    PreformattedToolResult("long body"),
                )
            )
            .external_output
        )
        assert reference is not None
        history = history_with_output(reference.render(), external_output=reference)
        with pytest.raises(ValueError):
            HistoryCompactor(provider, files=second_files).compact(history)
        with pytest.raises(ValueError, match="node-scoped workspace"):
            HistoryCompactor(provider).compact(history)


def test_tool_message_keeps_external_metadata_when_replacing_a_result():
    reference = ExternalToolOutput(10, 1, 5, (OutputTextFile("output.txt", 10, 1),))
    message = ToolMessage("body", "call", "work")
    updated = message.with_result(
        PreformattedToolResult("manifest", external_output=reference)
    )
    assert updated.get_external_output() is reference
    assert message.get_external_output() is None
    with pytest.raises(TypeError, match="ExternalToolOutput"):
        ToolMessage("body", "call", external_output="invalid")


def test_multiple_tool_results_form_one_complete_batch():
    provider = CompactionProvider(score=9)
    calls = [
        ToolCall("first", FunctionCall("work", {})),
        ToolCall("second", FunctionCall("work", {})),
    ]
    history = MessageHistory(
        [
            SystemMessage("system"),
            UserMessage("task"),
            AssistantMessage(None, tool_calls=calls),
            ToolMessage("first result", "first", "work"),
            ToolMessage("second result", "second", "work"),
        ]
    )
    result = HistoryCompactor(provider).compact(history)
    assert result.output_count == 2
    assert len(compacted_payload(result.history.get_history()[-1])["outputs"]) == 2
    validate_request_history(ProviderRequest(result.history.get_history()))


def test_real_tool_named_compact_history_is_still_a_candidate():
    provider = CompactionProvider(score=9)
    result = HistoryCompactor(provider).compact(
        history_with_output("ordinary result", name=HISTORY_COMPACTION.tool_name)
    )
    assert result.output_count == 1


def test_scoring_schema_requires_all_candidates_before_generation_can_finish():
    provider = CompactionProvider(score=9)
    history = history_with_output("first")
    history.add(output_batch("second", call_id="second", output_id="second-output"))
    HistoryCompactor(provider).compact(history)
    response_model = provider.score_requests[0].get_response_model()
    schema = response_model.model_json_schema()
    assert set(schema["properties"]) == {"entry_3", "entry_5"}
    assert set(schema["required"]) == {"entry_3", "entry_5"}
    assert schema["additionalProperties"] is False
    for field in schema["properties"].values():
        assert field["type"] == "integer"
        assert field["minimum"] == 1 and field["maximum"] == 100
    with pytest.raises(ValueError):
        response_model.model_validate({"entry_3": 95})
    with pytest.raises(ValueError):
        response_model.model_validate({"entry_3": 95, "entry_5": 20, "unknown": 50})
    assert provider.catalogs[0]["expected_entry_ids"] == ["3", "5"]
    assert provider.catalogs[0]["expected_count"] == 2


@pytest.mark.parametrize("value", [0, 101, True, 1.5, "50"])
def test_history_specific_score_fields_reject_invalid_ratings(value):
    response_model = compaction_scores_model(("3", "8:descendant-output"))
    values = {"entry_3": 50, "entry_8:descendant-output": value}
    with pytest.raises(ValueError):
        response_model.model_validate(values)


def test_history_specific_schema_preserves_each_piece_as_a_required_argument():
    response_model = compaction_scores_model(("3", "8:descendant-output"))
    schema = response_model.model_json_schema()
    assert set(schema["required"]) == {"entry_3", "entry_8:descendant-output"}
    assert response_model.model_validate(
        {"entry_3": 80, "entry_8:descendant-output": 25}
    ).model_dump() == {"entry_3": 80, "entry_8:descendant-output": 25}


def test_newer_cached_result_cannot_supply_descendants_of_an_old_visible_snapshot():
    provider = CompactionProvider(score=60)
    records = {
        "output": ToolCallRecord(
            "work",
            "output",
            {},
            "new snapshot",
            used_tools=(ToolOutputReference("child", "new-child"),),
        ),
        "new-child": ToolCallRecord("child", "new-child", {}, "unseen child"),
    }
    result = HistoryCompactor(provider, output_lookup=records.get).compact(
        history_with_output("old snapshot")
    )
    assert [entry["output_id"] for entry in provider.catalogs[0]["outputs"]] == [
        "output"
    ]
    output = compacted_payload(result.history.get_history()[-1])["outputs"][0]
    assert "used-tools" not in output
    assert output["recoverable"] is False


def test_explicit_empty_descendants_override_cached_references():
    provider = CompactionProvider(score=60)
    records = {
        "output": ToolCallRecord(
            "work",
            "output",
            {},
            "same snapshot",
            used_tools=(ToolOutputReference("child", "cached-child"),),
        ),
        "cached-child": ToolCallRecord("child", "cached-child", {}, "unseen child"),
    }
    messages = output_batch("same snapshot")
    messages[-1] = ToolMessage(
        XmlHeader("Tool", "work", "output", fields={"used-tools": []}).prepend_to(
            "same snapshot"
        ),
        "call",
        "work",
    )
    history = MessageHistory([SystemMessage("system"), UserMessage("task"), *messages])
    HistoryCompactor(provider, output_lookup=records.get).compact(history)
    assert [entry["output_id"] for entry in provider.catalogs[0]["outputs"]] == [
        "output"
    ]


def test_referenced_outputs_preserve_normalized_execution_metadata():
    provider = CompactionProvider(score=60)
    records = {
        "output": ToolCallRecord(
            "work",
            "output",
            {},
            "same snapshot",
            used_tools=(ToolOutputReference("child", "child-output"),),
        ),
        "child-output": ToolCallRecord(
            "child",
            "child-output",
            {},
            "cancelled output",
            execution={"status": "cancelled", "response_type": "cancelled"},
        ),
    }
    result = HistoryCompactor(provider, output_lookup=records.get).compact(
        history_with_output("same snapshot")
    )
    child = compacted_payload(result.history.get_history()[-1])["outputs"][1]
    assert child["status"] == child["response-type"] == "cancelled"


def test_inconsistent_explicit_file_metadata_does_not_discard_the_reference_silently():
    reference = ExternalToolOutput(10, 1, 5, (OutputTextFile("unread.txt", 10, 1),))
    history = history_with_output("different visible body", external_output=reference)
    before = history.dump_history()
    provider = CompactionProvider()
    with pytest.raises(ValueError, match="does not match the visible manifest"):
        HistoryCompactor(provider).compact(history)
    assert history.dump_history() == before
    assert not provider.score_requests
