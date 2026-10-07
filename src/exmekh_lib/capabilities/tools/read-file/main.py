"""Expose inspection-gated reads with explicit inclusive physical line ranges."""

from pathlib import Path

from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.utils.tool_errors import missing_dependency


@deep_tool(
    name="read-file",
    description=(
        "Read an explicit inclusive physical line range inside the workroot. "
        "Always call inspect-file first and choose start_line/end_line from its "
        "report; a full analysis already in visible history can be reused only "
        "for unchanged content. path, start_line and end_line are all required; "
        "indices must be positive integers, numbered from 1. Never defaults to "
        "the whole file or adjusts indices. Missing/stale analysis returns an "
        "error followed by an automatic inspect-file call; choose new indices "
        "and call read-file again. Returns numbered text lines; binary formats "
        "and empty files are explicit errors. Reads run synchronously so the "
        "prerequisite error and automatic inspection precede the next inference."
    ),
    background_enabled=False,
    argument_guidance=(
        "Supply path, start_line and end_line; choose positive inclusive line "
        "indices from inspect-file and call read-file again."
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
