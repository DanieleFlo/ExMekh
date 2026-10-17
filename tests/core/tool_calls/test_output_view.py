from datetime import UTC, datetime
from xml.etree import ElementTree as ET

import pytest

from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.tool_calls.output_view import ToolOutputView
from tests.tokenizer_support import PairTokenizer, StubTokenizer


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "10"])
def test_preview_rejects_invalid_token_limits(limit):
    output = ToolOutputView(ToolCallRecord("work", "output", {}, "result"))

    with pytest.raises(ValueError, match="preview limit must be a positive integer"):
        output.preview(limit, StubTokenizer())


@pytest.mark.parametrize("length", [479, 480, 481, 1000])
def test_preview_budget_counts_model_tokens_instead_of_characters(length):
    text = ("界😀àx" * 250)[:length]
    tokenizer = PairTokenizer()
    output = ToolOutputView(ToolCallRecord("work", "output", {}, text))

    preview = output.preview(240, tokenizer)

    assert tokenizer.count_tokens(preview) <= 240
    assert len(preview) > 240
    assert preview.endswith("…") is (length > 480)
    if length <= 480:
        assert preview == text
    assert output.content() == text


def test_preview_rechecks_token_count_after_appending_suffix():
    class SuffixTokenizer(PairTokenizer):
        def count_tokens(self, text):
            return super().count_tokens(text) + int(
                len(text) > 1 and text.endswith("…")
            )

    tokenizer = SuffixTokenizer()
    output = ToolOutputView(ToolCallRecord("work", "output", {}, "abcdefghij"))
    preview = output.preview(3, tokenizer)

    assert preview == "ab…"
    assert tokenizer.count_tokens(preview) == 3


def test_preview_does_not_exceed_budget_when_suffix_needs_multiple_tokens():
    class DoubleTokenizer(StubTokenizer):
        def count_tokens(self, text):
            return len(text) * 2

        def truncate_text(self, text, start_tokens, end_tokens=0):
            return text[: start_tokens // 2]

    tokenizer = DoubleTokenizer()
    output = ToolOutputView(ToolCallRecord("work", "output", {}, "abcdef"))
    preview = output.preview(1, tokenizer)
    assert preview == ""
    assert tokenizer.count_tokens(preview) <= 1


def test_saved_cursor_never_moves_backwards_when_reading_past_the_end():
    view = ToolOutputView(ToolCallRecord("work", "output", {}, "done"))
    header, body = view.format_saved(100).split("</header>\n", 1)
    assert ET.fromstring(header + "</header>").findtext("next-cursor") == "100"
    assert body == ""


@pytest.mark.parametrize("read_method", ["format_current", "format_saved"])
def test_read_time_changes_but_saved_execution_measurements_remain_exact(
    read_method, xml_clock
):
    record = ToolCallRecord(
        "work",
        "output",
        {},
        "done",
        execution={
            "status": "completed",
            "started_at": "2026-10-17T14:30:00.123456Z",
            "finished_at": "2026-10-17T14:32:00.987654Z",
            "elapsed_seconds": 120.864198,
        },
    )
    view = ToolOutputView(record)
    first = (
        view.format_saved(None)
        if read_method == "format_saved"
        else view.format_current()
    )
    xml_clock.now.return_value = datetime(2026, 10, 18, 0, 0, 1, tzinfo=UTC)
    second = (
        view.format_saved(None)
        if read_method == "format_saved"
        else view.format_current()
    )

    for text, expected_time in (
        (first, "2026-10-17T14:32:08Z"),
        (second, "2026-10-18T00:00:01Z"),
    ):
        header_text, body = text.split("</header>\n", 1)
        header = ET.fromstring(header_text + "</header>")
        assert header.findtext("current-time") == expected_time
        assert header.findtext("elapsed") == "2m 0.9s"
        assert header.findtext("started-at") == "2026-10-17T14:30:00Z"
        assert header.findtext("finished-at") == "2026-10-17T14:32:00Z"
        assert body == "done"
    assert record.execution["elapsed_seconds"] == 120.864198
    assert record.execution["started_at"] == "2026-10-17T14:30:00.123456Z"


@pytest.mark.parametrize(
    "status,response_type",
    [
        ("completed", "final"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("running", "running"),
        ("unavailable", "failed"),
    ],
)
def test_search_fields_interpret_execution_status_without_prior_normalization(
    status, response_type
):
    view = ToolOutputView(
        ToolCallRecord("work", "output", {}, "result", execution={"status": status})
    )
    assert view.search_fields()["response-type"] == response_type
    assert view.search_fields()["status"] == status
