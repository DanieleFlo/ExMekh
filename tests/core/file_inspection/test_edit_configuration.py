"""Two-marker framing is fixed; limits and diagnostics are shared configuration."""

from dataclasses import replace

import pytest

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.config.interactive import INTERACTIVE_PROTOCOL
from exmekh_lib.core.file_inspection import edit_filesystem, edit_selection
from exmekh_lib.core.file_inspection.edit_filesystem import (
    FileWriteTarget,
    edit_snapshot,
)
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.models.edit_document import EditDocument
from exmekh_lib.core.models.interactive import (
    InteractiveDefinition,
    InteractiveProtocolError,
)
from exmekh_lib.core.providers import interactive_contract, interactive_openai
from exmekh_lib.core.providers.interactive_contract import InteractiveContract
from exmekh_lib.core.providers.interactive_openai import InteractiveResponseOpenAI
from tests.interactive_support import edit_fragment


@pytest.mark.parametrize("field", ["edits", "renamed_document"])
def test_literal_definition_requires_only_one_document_string(field):
    definition = InteractiveDefinition("edit_file.v3", {"document": field})
    schema = {"properties": {field: {"type": "string"}}, "required": [field]}
    assert definition.configuration("edit", schema) == {
        "name": "edit",
        "behavior": "edit_file.v3",
        "fields": {"document": field},
    }
    for invalid in (
        {"properties": {field: {"type": "integer"}}, "required": [field]},
        {"properties": {field: {"type": "string"}}, "required": []},
        {
            "properties": {field: {"type": "string"}, "extra": {"type": "string"}},
            "required": [field],
        },
    ):
        with pytest.raises(InteractiveProtocolError):
            definition.configuration("edit", invalid)


def test_no_v1_or_role_mapping_fallback():
    for behavior, fields in (
        ("edit_file.v1", {"document": "edits"}),
        (
            "edit_file.v3",
            {"resource": "path", "selection": "old_text", "replacement": "new_text"},
        ),
    ):
        with pytest.raises(InteractiveProtocolError):
            InteractiveDefinition(behavior, fields).configuration(
                "edit",
                {"properties": {"edits": {"type": "string"}}, "required": ["edits"]},
            )


def test_semantic_error_messages_use_configuration(monkeypatch):
    settings = replace(
        FILE_EDITING,
        selection_error="configured selection error",
    )
    monkeypatch.setattr(edit_selection, "FILE_EDITING", settings)
    for old, error in (("missing", settings.selection_error),):
        document = EditDocument.parse(edit_fragment("file.txt", old, "new"))
        with pytest.raises(FileInspectionError, match=error):
            edit_selection.select_replacements("one", document.elements)


def test_file_target_owns_validation_and_commit_with_configured_temporary_policy(
    tmp_path, monkeypatch
):
    settings = replace(
        FILE_EDITING,
        temporary_file_prefix=".configured-write-",
        temporary_file_mode=0o640,
    )
    monkeypatch.setattr(edit_filesystem, "FILE_EDITING", settings)
    target = tmp_path / "notes.txt"
    FileWriteTarget(tmp_path, target.name, target).commit(b"original")
    assert target.read_bytes() == b"original"
    assert target.stat().st_mode & 0o777 == 0o640
    approved = FileWriteTarget(
        tmp_path, target.name, target, edit_snapshot(tmp_path, target.name)
    )
    approved.validate()
    approved.commit(b"replaced")
    assert target.read_bytes() == b"replaced"
    assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]
    stale = FileWriteTarget(
        tmp_path, target.name, target, edit_snapshot(tmp_path, target.name)
    )
    target.write_bytes(b"external writer")
    with pytest.raises(FileInspectionError, match="changed"):
        stale.commit(b"must not overwrite")
    assert target.read_bytes() == b"external writer"
    assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]


def test_configured_wire_version_is_shared_by_inbound_and_outbound_contracts(
    monkeypatch,
):
    protocol = replace(INTERACTIVE_PROTOCOL, version=7)
    monkeypatch.setattr(interactive_contract, "INTERACTIVE_PROTOCOL", protocol)
    monkeypatch.setattr(interactive_openai, "INTERACTIVE_PROTOCOL", protocol)
    outgoing = InteractiveResponseOpenAI._control({}, "cancel", "owned-continuation")
    assert outgoing["agentic"]["interactive"]["version"] == 7
    contract = InteractiveContract({})
    contract.record_acceptance({"version": 7, "continuation_id": "owned-continuation"})
    assert contract.continuation_id == "owned-continuation"
    with pytest.raises(InteractiveProtocolError):
        contract.record_acceptance(
            {"version": 2, "continuation_id": "owned-continuation"}
        )
