from datetime import UTC, datetime, timezone
from unittest.mock import MagicMock
from uuid import UUID
from xml.etree import ElementTree as ET

import pytest

from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.core.tool_calls.execution_manager import ExecutionSnapshot
from exmekh_lib.core.tool_calls.output_formatter import ToolOutputFormatter
from exmekh_lib.core.tool_calls.output_view import ToolOutputView

pytestmark = pytest.mark.usefixtures("xml_clock")


@pytest.mark.parametrize("invalid_id", [None, 42, "", "   ", OSError("unavailable")])
def test_invalid_generated_id_has_a_recoverable_fallback(invalid_id):
    client = MagicMock(spec=Client)
    if isinstance(invalid_id, Exception):
        client.generate_id.side_effect = invalid_id
    else:
        client.generate_id.return_value = invalid_id
    formatter = ToolOutputFormatter(client)

    content, output_id = formatter.format_tool_result("work", "done")

    header, body = content.split("</header>\n", 1)
    assert ET.fromstring(header + "</header>").findtext("id") == output_id
    assert UUID(hex=output_id)
    assert body == "done"


@pytest.mark.parametrize("preformatted", [False, True])
def test_failed_optional_annotation_preserves_content_and_existing_id(preformatted):
    client = MagicMock(spec=Client)
    client.generate_id.return_value = "canonical"
    formatter = ToolOutputFormatter(client)
    body = "# tool-authored output"
    original = XmlHeader(source="Tool", name="work", id="canonical").prepend_to(body)
    annotation = MagicMock(side_effect=OSError("annotation unavailable"))

    if preformatted:
        content, output_id = formatter.format_tool_result(
            "work", PreformattedToolResult(original), annotation_provider=annotation
        )
        assert content == original
        assert output_id is None
    else:
        content, output_id = formatter.format_tool_result(
            "work",
            body,
            annotation_provider=annotation,
        )
        expected_header = XmlHeader(source="Tool", name="work", id="canonical")
        expected_header.add_field("response-type", "final")
        assert content == expected_header.prepend_to(body)
        assert output_id == "canonical"
    assert client.generate_id.call_count == (0 if preformatted else 1)


@pytest.mark.parametrize("annotation", [None, "", "   ", "useful annotation"])
def test_preformatted_annotation_keeps_the_authored_body(annotation):
    formatter = ToolOutputFormatter(MagicMock(spec=Client))
    body = "<tool-authored>\nraw text\n</tool-authored>"
    original = XmlHeader(source="Tool", name="work", id="existing").prepend_to(body)

    content, output_id = formatter.format_tool_result(
        "work", PreformattedToolResult(original), annotation_provider=lambda: annotation
    )

    header, actual_body = content.split("</header>\n", 1)
    root = ET.fromstring(header + "</header>")
    assert root.findtext("thinking") == (
        annotation if annotation and annotation.strip() else None
    )
    assert actual_body == body
    assert output_id is None


def test_id_and_thinking_whitespace_are_handled_independently():
    client = MagicMock(spec=Client)
    client.generate_id.return_value = " output "
    client.get_tokenizer.return_value.truncate_text.return_value = "excerpt"
    formatter = ToolOutputFormatter(client)

    content, output_id = formatter.format_skill_return(
        "skill", "done", thinking="long thinking"
    )

    header, _ = content.split("</header>\n", 1)
    root = ET.fromstring(header + "</header>")
    assert root.findtext("thinking") == "excerpt"
    assert root.findtext("id") == output_id == "output"
    client.get_tokenizer.return_value.truncate_text.assert_called_once_with(
        "long thinking", 100
    )


@pytest.mark.parametrize("status", ["running", "completed", "failed", "cancelled"])
def test_managed_reply_and_read_have_identical_execution_metadata(status):
    snapshot = ExecutionSnapshot(
        call_id="call",
        name="work",
        arguments={"text": "<&>"},
        status=status,
        started_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
        finished_at=None,
        output_id="output",
        elapsed_seconds=1,
        result="done",
    )
    formatter = ToolOutputFormatter(MagicMock(spec=Client))
    immediate, output_id = formatter.format_tool_result(
        "work", snapshot.output_result(), execution=snapshot
    )
    read = formatter.format_execution_output("output", snapshot, None)

    assert immediate == read
    assert output_id == "output"


@pytest.mark.parametrize("source", ["Tool", "DeepSkill"])
def test_inline_result_and_saved_read_both_report_final_response_type(source):
    client = MagicMock(spec=Client)
    client.generate_id.return_value = "output"
    formatter = ToolOutputFormatter(client)
    if source == "DeepSkill":
        immediate, output_id = formatter.format_skill_return("work", "done")
        result = {"text": "done", "items": []}
    else:
        immediate, output_id = formatter.format_tool_result("work", "done")
        result = "done"
    record = ToolCallRecord(
        name="work", id=output_id, args={}, result=result, source=source
    )
    saved = ToolOutputView(record).format_saved(None)

    for content in (immediate, saved):
        header, body = content.split("</header>\n", 1)
        assert ET.fromstring(header + "</header>").findtext("response-type") == "final"
        assert body == "done"


@pytest.mark.parametrize("duration", [None, "7"])
@pytest.mark.parametrize("name", ["work", "read_tool_output", "renamed-read"])
def test_preformatted_metadata_is_preserved_independently_of_registered_name(
    duration, name
):
    formatter = ToolOutputFormatter(MagicMock(spec=Client))
    original = XmlHeader("Tool", "work", "output")
    original.add_field("elapsed-seconds", duration)
    original_content = original.prepend_to("done")
    content, output_id = formatter.format_tool_result(
        name, PreformattedToolResult(original_content), elapsed_seconds=0.25
    )
    assert content == original_content
    assert output_id is None


def test_preformatted_text_without_header_is_preserved_when_timing_is_available():
    formatter = ToolOutputFormatter(MagicMock(spec=Client))
    annotation = MagicMock(return_value="unused")
    content, output_id = formatter.format_tool_result(
        "work",
        PreformattedToolResult("authored output"),
        elapsed_seconds=0.25,
        annotation_provider=annotation,
    )
    assert content == "authored output"
    assert output_id is None
    annotation.assert_not_called()


def test_reading_legacy_output_does_not_substitute_read_duration_for_missing_execution_time():
    formatter = ToolOutputFormatter(MagicMock(spec=Client))
    original = XmlHeader("Tool", "work", "legacy").prepend_to("done")
    content, output_id = formatter.format_tool_result(
        "read_tool_output", PreformattedToolResult(original), elapsed_seconds=0.25
    )
    assert content == original
    assert output_id is None


def test_preformatted_read_refreshes_time_without_replacing_original_duration(
    xml_clock,
):
    formatter = ToolOutputFormatter(MagicMock(spec=Client))
    body = "# Authored result\r\n\n"
    original = XmlHeader(
        "Tool", "work", "output", fields={"elapsed-seconds": 3723.456789}
    ).prepend_to(body)
    xml_clock.now.return_value = datetime(2026, 10, 18, 10, 0, 0, tzinfo=UTC)

    content, output_id = formatter.format_tool_result(
        "read_tool_output", PreformattedToolResult(original), elapsed_seconds=0.123456
    )
    parsed = XmlHeader.parse_content(content)

    assert parsed.field("current-time") == "2026-10-18T10:00:00Z"
    assert parsed.field("elapsed") == "1h 2m 3.5s"
    assert parsed.field("id") == "output"
    assert parsed.body == body
    assert output_id is None


def test_duration_formatting_does_not_change_tool_authored_structured_values():
    client = MagicMock(spec=Client)
    client.generate_id.return_value = "output"
    formatter = ToolOutputFormatter(client)
    payload = {"elapsed_seconds": 3.14159265, "started_at": "original timestamp"}

    content, _ = formatter.format_tool_result(
        "work", payload, elapsed_seconds=3.14159265
    )
    parsed = XmlHeader.parse_content(content)
    body = ET.fromstring(parsed.body)

    assert parsed.field("elapsed") == "3.1s"
    assert body.findtext("elapsed_seconds") == "3.14159265"
    assert body.findtext("started_at") == "original timestamp"
    assert payload["elapsed_seconds"] == 3.14159265


def test_explicit_failed_preformatted_result_preserves_content_with_failed_metadata():
    client = MagicMock(spec=Client)
    client.generate_id.return_value = "failed-output"
    formatter = ToolOutputFormatter(client)
    content, output_id = formatter.format_tool_result(
        "work", PreformattedToolResult("authored error"), failed=True
    )
    assert output_id == "failed-output"
    assert "<response-type>failed</response-type>" in content
    assert content.endswith("authored error")


def test_legacy_failed_snapshot_preserves_its_diagnostic_in_structured_output():
    snapshot = ExecutionSnapshot(
        call_id="call",
        name="work",
        arguments={},
        status="failed",
        started_at=datetime(2026, 9, 30, tzinfo=timezone.utc),
        finished_at=None,
        output_id="output",
        elapsed_seconds=1,
        error="legacy failure",
    )
    assert snapshot.output_result()["error"] == "legacy failure"
    assert "legacy failure" in snapshot.output_content()
