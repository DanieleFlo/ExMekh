"""Create a workspace file through the standard tool lifecycle."""

from pathlib import Path
from typing import Annotated

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.edit_filesystem import new_file
from exmekh_lib.core.file_inspection.errors import FileInspectionError


@deep_tool(
    name="new-file",
    description=FILE_EDITING.new_file_description,
    background_enabled=False,
)
def run(
    work_root: Path,
    path: str,
    content: Annotated[str, {"minLength": FILE_EDITING.min_new_file_characters}],
) -> dict[str, object]:
    try:
        return new_file(work_root, path, content)
    except (FileInspectionError, OSError, UnicodeError, ValueError) as error:
        return {"ok": False, "error_kind": "invalid_file_creation", "error": str(error)}
