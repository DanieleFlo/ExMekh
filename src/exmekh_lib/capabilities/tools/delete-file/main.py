"""Delete a workspace file through the standard tool lifecycle."""

from pathlib import Path

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.edit_filesystem import delete_file
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.utils.tool_errors import tool_error_result


@deep_tool(
    name="delete-file",
    description=FILE_EDITING.delete_file_description,
    background_enabled=False,
)
def run(work_root: Path, path: str) -> dict[str, object]:
    try:
        return delete_file(work_root, path)
    except (FileInspectionError, OSError, UnicodeError, ValueError) as error:
        return tool_error_result(error, "invalid_file_deletion")
