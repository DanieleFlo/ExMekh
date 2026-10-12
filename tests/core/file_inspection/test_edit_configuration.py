"""XML grammar is fixed; semantic limits and diagnostics are shared configuration."""

from dataclasses import replace

import pytest

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.file_inspection import edit_selection
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.models.edit_document import EditDocument
from exmekh_lib.core.models.interactive import (
    InteractiveDefinition,
    InteractiveProtocolError,
)
from tests.interactive_support import edit_fragment


@pytest.mark.parametrize("field", ["edits", "renamed_document"])
def test_v2_definition_requires_only_one_document_string(field):
    definition = InteractiveDefinition("edit_file.v2", {"document": field})
    schema = {"properties": {field: {"type": "string"}}, "required": [field]}
    assert definition.configuration("edit", schema) == {
        "name": "edit",
        "behavior": "edit_file.v2",
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
            "edit_file.v2",
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
        range_error="configured range error",
        selection_error="configured selection error",
    )
    monkeypatch.setattr(edit_selection, "FILE_EDITING", settings)
    for old, error in (
        ('<line number="0"/>', settings.range_error),
        ("missing", settings.selection_error),
    ):
        document = EditDocument.parse(edit_fragment("file.txt", old, "new"))
        with pytest.raises(FileInspectionError, match=error):
            edit_selection.select_replacements("one", document.elements, exists=True)
