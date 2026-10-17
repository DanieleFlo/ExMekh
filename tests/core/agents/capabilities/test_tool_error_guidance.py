"""Real bundled tools explain the cause and permit a corrected next call."""

from pathlib import Path

import pytest

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.file_inspection.edit_filesystem import new_file
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.workspace import read_snapshot
from exmekh_lib.core.reasoning_tools.date_time import (
    DateTimeInputError,
    calculate_date_time,
)
from exmekh_lib.core.reasoning_tools.formal_logic import logic_tool_result
from exmekh_lib.utils.tool_errors import ToolInputError, tool_failure_result


@pytest.fixture
def bundled_tools(tmp_path):
    loaded = Capabilities(None, default_capabilities=DefaultCapabilities()).load(
        work_root=tmp_path
    )
    return {tool.name: tool for tool in loaded.tools}


@pytest.mark.parametrize(
    "name,arguments,kind,code",
    [
        (
            "inspect-file",
            {"path": "missing.txt"},
            "invalid_file_inspection",
            "file_not_found",
        ),
        ("inspect-file", {"path": " "}, "invalid_file_inspection", "invalid_path"),
        (
            "inspect-file",
            {"path": "."},
            "invalid_file_inspection",
            "not_a_regular_file",
        ),
        (
            "read-file",
            {"path": "notes.txt", "start_index": 3, "end_index": 1},
            "invalid_file_read",
            "invalid_selection",
        ),
        (
            "new-file",
            {"document": '<new-exmekh-file path="notes.txt"/>new'},
            "invalid_file_creation",
            "file_already_exists",
        ),
        (
            "new-file",
            {"document": '<new-exmekh-file path="missing/new.txt"/>new'},
            "invalid_file_creation",
            "parent_directory_missing",
        ),
        (
            "new-file",
            {"document": '<new-exmekh-file path="new.txt"/>x'},
            "invalid_file_creation",
            "invalid_content",
        ),
        (
            "delete-file",
            {"paths": ["missing.txt"]},
            "invalid_file_deletion",
            "file_not_found",
        ),
        ("delete-file", {"paths": ["."]}, "invalid_file_deletion", "invalid_path"),
        (
            "edit-file",
            {"edits": "invalid"},
            "invalid_file_edit",
            "interactive_exchange_required",
        ),
        (
            "calculate",
            {"operation": "evaluate", "expression": "1+"},
            "invalid_math",
            "invalid_math_syntax",
        ),
        (
            "calculate",
            {"operation": "evaluate", "expression": "x+1"},
            "invalid_math",
            "unbound_math_variables",
        ),
        (
            "calculate",
            {"operation": "integrate", "expression": "x", "lower_bound": "0"},
            "invalid_math",
            "incomplete_integral_bounds",
        ),
        (
            "calculate",
            {"operation": "inverse", "matrix": [[1, 2], [2, 4]]},
            "invalid_math",
            "singular_matrix",
        ),
        (
            "calculate",
            {"operation": "evaluate", "expression": "1/0"},
            "invalid_math",
            "undefined_math_result",
        ),
        (
            "date-time",
            {"operation": "convert", "value": "bad"},
            "invalid_date_time",
            "invalid_datetime",
        ),
        (
            "date-time",
            {"operation": "now", "timezone": "Missing/Timezone"},
            "invalid_date_time",
            "invalid_timezone",
        ),
    ],
)
def test_real_tools_preserve_error_family_and_expose_specific_recovery(
    tmp_path, bundled_tools, name, arguments, kind, code
):
    (tmp_path / "notes.txt").write_text("original")
    result = bundled_tools[name].execute(**arguments)
    if name in {"delete-file", "new-file"}:
        assert result["ok"] is False
        result = result["results"][0]
    assert result["ok"] is False
    assert result["error_kind"] == kind
    assert result["error_code"] == code
    assert result["error"] and result["hint"]
    assert (tmp_path / "notes.txt").read_text() == "original"


def test_creation_correction_uses_unused_path_without_overwriting(
    tmp_path, bundled_tools
):
    tool = bundled_tools["new-file"]
    (tmp_path / "existing.txt").write_text("keep")
    rejected = tool.execute(
        document='<new-exmekh-file path="existing.txt"/>replacement'
    )
    assert "never overwrites" in rejected["results"][0]["hint"]
    assert (
        tool.execute(document='<new-exmekh-file path="new.txt"/>replacement')["ok"]
        is True
    )
    assert (tmp_path / "existing.txt").read_text() == "keep"


def test_permission_diagnostic_survives_inspection_boundary(tmp_path, monkeypatch):
    (tmp_path / "notes.txt").write_text("original")
    original = Path.resolve

    def denied(path, *args, **kwargs):
        if path.name == "notes.txt":
            raise PermissionError("denied")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", denied)
    with pytest.raises(FileInspectionError) as caught:
        read_snapshot(tmp_path, "notes.txt", FILE_INSPECTION)
    assert caught.value.code == "permission_denied"


@pytest.mark.parametrize("root_kind", ["missing", "file"])
def test_read_and_write_boundaries_share_workroot_diagnostics(tmp_path, root_kind):
    root = tmp_path / "root"
    if root_kind == "file":
        root.write_text("not a directory")
    for operation in (
        lambda: read_snapshot(root, "notes.txt", FILE_INSPECTION),
        lambda: new_file(root, "notes.txt", "new"),
    ):
        with pytest.raises(FileInspectionError) as caught:
            operation()
        assert caught.value.code == "workroot_unavailable"
        assert "Restore" in caught.value.hint


@pytest.mark.parametrize(
    "value,code",
    [
        ("2026-03-29T02:30:00", "nonexistent_local_time"),
        ("2026-10-25T02:30:00", "ambiguous_local_time"),
    ],
)
def test_dst_correction_selects_an_explicit_offset(value, code):
    with pytest.raises(DateTimeInputError) as caught:
        calculate_date_time("convert", value, timezone="Europe/Rome")
    assert caught.value.code == code
    assert "offset" in caught.value.hint
    assert calculate_date_time("convert", value + "+02:00", timezone="Europe/Rome")[
        "utc"
    ]


def test_logic_undeclared_variable_is_correctable_without_changing_semantics():
    rejected = logic_tool_result("consistency", {}, [{"var": "P"}])
    assert rejected["error_kind"] == "parse_error"
    assert rejected["error_code"] == "undeclared_logic_variable"
    assert "Declare" in rejected["hint"]
    assert (
        logic_tool_result("consistency", {"P": "Bool"}, [{"var": "P"}])["status"]
        == "consistent"
    )


def test_control_rejection_does_not_masquerade_as_argument_binding_failure():
    result = tool_failure_result(
        ToolInputError("Unknown execution", code="unknown_tool_call_id")
    )
    assert result["error_kind"] == "tool_execution_failed"
    assert result["error_code"] == "unknown_tool_call_id"
    assert "original tool_call_id" in result["hint"]
