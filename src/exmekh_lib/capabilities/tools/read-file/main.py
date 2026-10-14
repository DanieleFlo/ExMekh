"""Expose direct image reads and inspection-gated text or PDF ranges."""

from pathlib import Path

from exmekh_lib.config.file_inspection import FILE_INSPECTION_TEXT
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.utils.tool_errors import missing_dependency, tool_error_result


@deep_tool(
    name="read-file",
    description=(FILE_INSPECTION_TEXT.read_description),
    background_enabled=False,
    argument_guidance=(FILE_INSPECTION_TEXT.read_guidance),
)
def run(
    work_root: Path, path: str, start_index: int, end_index: int
) -> dict[str, object]:
    try:
        from exmekh_lib.core.file_inspection.reading import read_file

        return read_file(work_root, path, start_index, end_index)
    except ModuleNotFoundError as error:
        return missing_dependency(error)
    except FileInspectionError as error:
        return tool_error_result(error, "invalid_file_read")
