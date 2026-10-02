from datetime import datetime, timezone
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

    assert immediate.split("</header>", 1)[0] == read.split("</header>", 1)[0]
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
