"""Expose synchronous XML file editing through the standard tool lifecycle."""

from pathlib import Path
from typing import Annotated

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.editing import edit_files
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.models.interactive import InteractiveDefinition


@deep_tool(
    name="edit-file",
    description=FILE_EDITING.description,
    background_enabled=False,
    argument_guidance="Supply one exact edits XML document. Inspect and read existing files first.",
    interactive=InteractiveDefinition("edit_file.v2", {"document": "edits"}),
)
def run(
    work_root: Path,
    edits: Annotated[str, FILE_EDITING.document_description],
) -> dict[str, object]:
    try:
        return edit_files(work_root, edits)
    except (FileInspectionError, OSError, UnicodeError, ValueError) as error:
        return {"ok": False, "error_kind": "invalid_file_edit", "error": str(error)}
