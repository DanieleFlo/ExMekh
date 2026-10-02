import pytest

from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.tool_calls.output_view import ToolOutputView


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "10"])
def test_preview_rejects_invalid_character_limits(limit):
    output = ToolOutputView(ToolCallRecord("work", "output", {}, "result"))

    with pytest.raises(ValueError, match="preview limit must be a positive integer"):
        output.preview(limit)
