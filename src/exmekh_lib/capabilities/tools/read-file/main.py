"""Expose direct image reads and inspection-gated physical text line ranges."""

from pathlib import Path

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.utils.tool_errors import missing_dependency


@deep_tool(
    name="read-file",
    description=(
        "Read text line ranges, supported images or PDF pages inside the workroot. "
        "For PDF, inspect-file is required to learn the page count: "
        "start_line/end_line select inclusive "
        f"1-based page numbers, at most {FILE_INSPECTION.max_pdf_render_pages} pages per call, rendered as images at "
        "configured DPI. Invalid or oversized page intervals return errors. "
        "For images, no inspection is required: set start_line and end_line to 0; "
        "their values are ignored even when nonzero. Returns the image as an "
        "attachment with metadata; animated formats return the first frame. "
        "For text, call inspect-file first and choose start_line/end_line from its "
        "report; a full analysis already in visible history can be reused only "
        "for unchanged content. path, start_line and end_line are all required; "
        "text indices must be positive integers, numbered from 1. Never defaults to "
        "the whole file or adjusts indices. Missing/stale analysis returns an "
        "error followed by an automatic inspect-file call; choose new indices "
        "and call read-file again. Returns numbered text lines; other binary formats "
        "are explicit errors. For an inspected empty text file use 0/0 for a "
        "successful empty read. Reads run synchronously so the "
        "prerequisite error and automatic inspection precede the next inference."
    ),
    background_enabled=False,
    argument_guidance=(
        "Supply path, start_line and end_line; use 0/0 for images or choose "
        "positive inclusive text line indices from inspect-file; for PDF, select "
        f"1-based inclusive page indices with at most {FILE_INSPECTION.max_pdf_render_pages} pages, then call read-file again."
    ),
)
def run(
    work_root: Path, path: str, start_line: int, end_line: int
) -> dict[str, object]:
    try:
        from exmekh_lib.core.file_inspection.reading import read_file

        return read_file(work_root, path, start_line, end_line)
    except ModuleNotFoundError as error:
        return missing_dependency(error)
    except FileInspectionError as error:
        return {"ok": False, "error_kind": "invalid_file_read", "error": str(error)}
