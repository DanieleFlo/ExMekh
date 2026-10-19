"""Expose synchronous literal file editing through the standard tool lifecycle."""

from pathlib import Path
from typing import Annotated

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.config.file_editing import FILE_EDIT_FORMAT, FILE_EDITING
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.editing import edit_files
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.models.interactive import InteractiveDefinition
from exmekh_lib.utils.tool_errors import tool_error_result


@deep_tool(
    name=DEFAULT_CAPABILITIES.edit_file,
    description=FILE_EDITING.description,
    background_enabled=False,
    argument_guidance=FILE_EDITING.argument_guidance,
    interactive=InteractiveDefinition(FILE_EDIT_FORMAT.behavior, {"document": "edits"}),
)
def run(
    work_root: Path,
    edits: Annotated[str, FILE_EDITING.document_description],
) -> dict[str, object]:
    try:
        return edit_files(work_root, edits)
    except (FileInspectionError, OSError, UnicodeError, ValueError) as error:
        return tool_error_result(error, "invalid_file_edit")
