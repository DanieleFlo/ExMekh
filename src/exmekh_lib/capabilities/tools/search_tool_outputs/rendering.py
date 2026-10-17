from __future__ import annotations

from xml.etree import ElementTree as ET

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.tool_calls.output_view import ToolOutputView
from exmekh_lib.interfaces.tokenizer import Tokenizer
from exmekh_lib.utils.xml import append_text_element, render_xml

from .query import ToolOutputQuery
from .tool_call_search import ToolCallSearch


def render_search_results(
    search: ToolCallSearch,
    query: ToolOutputQuery,
    tokenizer: Tokenizer,
) -> PreformattedToolResult:
    records, total, next_offset = search.find(query)
    root = ET.Element("tool-outputs")
    append_text_element(root, "total", str(total))
    if next_offset is not None:
        append_text_element(root, "next-offset", str(next_offset))
    records_element = ET.SubElement(root, "records")
    for record in records:
        _append_search_record(records_element, record, tokenizer)
    return PreformattedToolResult(render_xml(root))


def _append_search_record(
    root: ET.Element, record: ToolCallRecord, tokenizer: Tokenizer
) -> None:
    element = ET.SubElement(root, "record")
    output = ToolOutputView(record)
    for name, value in output.search_fields().items():
        if value is not None:
            append_text_element(element, name, str(value))
    append_text_element(
        element, "preview", output.preview(RUNTIME.memory_preview_tokens, tokenizer)
    )
