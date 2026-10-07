from xml.etree import ElementTree as ET

import pytest

from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.tool_calls.output_view import ToolOutputView


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "10"])
def test_preview_rejects_invalid_character_limits(limit):
    output = ToolOutputView(ToolCallRecord("work", "output", {}, "result"))

    with pytest.raises(ValueError, match="preview limit must be a positive integer"):
        output.preview(limit)


def test_saved_cursor_never_moves_backwards_when_reading_past_the_end():
    view = ToolOutputView(ToolCallRecord("work", "output", {}, "done"))
    header, body = view.format_saved(100).split("</header>\n", 1)
    assert ET.fromstring(header + "</header>").findtext("next-cursor") == "100"
    assert body == ""


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
