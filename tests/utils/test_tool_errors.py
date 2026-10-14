"""Dependency failures share one model-facing result across tools."""

import errno

import pytest

from exmekh_lib.utils.tool_errors import (
    ToolArgumentError,
    ToolDiagnosticError,
    ToolInputError,
    error_diagnostic,
    missing_dependency,
    tool_error_result,
    tool_failure_result,
)
from exmekh_lib.utils.xml import stringify_tool_result
from tests.tool_result_support import parse_tool_result


def test_missing_dependency_preserves_module_and_install_hint() -> None:
    result = missing_dependency(
        ModuleNotFoundError("No module named 'crawl4ai'", name="crawl4ai")
    )
    assert result == {
        "ok": False,
        "error_kind": "missing_dependency",
        "error": "Missing exmekh_lib runtime module: crawl4ai.",
        "hint": "Install the ExMekh-Lib wheel with its declared dependencies in this environment.",
    }


@pytest.mark.parametrize(
    "error,code",
    [
        (FileNotFoundError("missing"), "file_not_found"),
        (PermissionError("denied"), "permission_denied"),
        (IsADirectoryError("directory"), "not_a_regular_file"),
        (NotADirectoryError("parent"), "not_a_regular_file"),
        (UnicodeEncodeError("ascii", "é", 0, 1, "unsupported"), "invalid_encoding"),
        (OSError(errno.ENOSPC, "full"), "disk_full"),
        (OSError(errno.EIO, "I/O"), "filesystem_error"),
        (ZeroDivisionError("zero"), "undefined_math_result"),
        (TypeError("internal type error"), "tool_execution_failed"),
        (ValueError("internal value error"), "tool_execution_failed"),
        (RuntimeError("file not found"), "tool_execution_failed"),
        (ToolArgumentError("missing parameter"), "invalid_tool_arguments"),
    ],
)
def test_failure_classification_uses_types_not_message_text(error, code):
    result = tool_failure_result(error)
    assert result["error_code"] == code
    assert result["hint"]
    assert result["error_kind"] == (
        "invalid_tool_arguments"
        if isinstance(error, ToolArgumentError)
        else "tool_execution_failed"
    )


def test_diagnostic_fields_survive_xml_with_invalid_characters_and_keep_family():
    error = ToolInputError(
        "Invalid <input>\x00",
        code="invalid_path",
        hint="Use <path> & retry",
        details={"path": "bad\x00", "attempt": 2},
    )
    result = tool_error_result(error, "invalid_file_read")
    assert parse_tool_result(stringify_tool_result(result)) == result
    assert result["error_kind"] == "invalid_file_read"
    assert result["error_code"] == "invalid_path"
    assert isinstance(error, ValueError)
    copied = error_diagnostic(error)
    copied["details"]["attempt"] = 3
    assert error.details["attempt"] == 2


def test_nested_diagnostic_details_are_isolated_from_sources_and_readers():
    source = {"actions": [{"success": True}]}
    error = ToolInputError("Rejected", code="invalid_actions", details=source)
    source["actions"][0]["success"] = False
    assert error.details["actions"][0]["success"] is True
    result = error_diagnostic(error)
    result["details"]["actions"][0]["success"] = False
    assert error.details["actions"][0]["success"] is True
    view = error.details
    view["actions"].clear()
    assert len(error.details["actions"]) == 1


def test_unknown_diagnostic_code_does_not_imply_invalid_model_input():
    error = ToolDiagnosticError(
        "Unexpected backend failure", code="custom_backend_failure"
    )
    assert "Internal failures" in error.hint
    assert "Correct the rejected input" not in error.hint


def test_explicit_detail_updates_and_argument_context_keep_independent_snapshots():
    error = ToolInputError(
        "Bad range", code="range_out_of_bounds", details={"limits": [1, 10]}
    )
    context = {"path": "file.txt", "actions": [{"success": True}]}
    error.add_details(context)
    context["actions"].clear()
    wrapped = ToolArgumentError.from_error(error, "Invalid work arguments")
    error.add_details({"limits": [1, 20]})
    assert wrapped.code == "range_out_of_bounds"
    assert wrapped.hint == error.hint
    assert wrapped.details == {
        "limits": [1, 10],
        "path": "file.txt",
        "actions": [{"success": True}],
    }


def test_argument_context_preserves_guidance_for_plain_validation_errors():
    error = ToolArgumentError.from_error(
        ValueError("bad"), "Invalid arguments", hint="Use the documented format"
    )
    assert error.code == "invalid_tool_arguments"
    assert error.hint == "Use the documented format"
