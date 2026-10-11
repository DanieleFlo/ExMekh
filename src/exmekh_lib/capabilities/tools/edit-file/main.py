"""Expose one synchronous file edit behavior through the standard tool lifecycle."""

from pathlib import Path
from typing import Annotated

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.editing import edit_file
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.models.interactive import InteractiveDefinition


@deep_tool(
    name="edit-file",
    description=FILE_EDITING.description,
    background_enabled=False,
    argument_guidance="Supply path, old_text and new_text as exact strings. Inspect and read existing files first.",
    interactive=InteractiveDefinition(
        "edit_file.v1",
        {"resource": "path", "selection": "old_text", "replacement": "new_text"},
        ("path",),
        {
            "line": FILE_EDITING.line_family,
            "all": FILE_EDITING.all_marker,
            "delete": FILE_EDITING.delete_marker,
        },
    ),
)
def run(
    work_root: Path,
    path: Annotated[str, FILE_EDITING.resource_description],
    old_text: Annotated[str, FILE_EDITING.selection_description],
    new_text: Annotated[str, FILE_EDITING.replacement_description],
) -> dict[str, object]:
    try:
        return edit_file(work_root, path, old_text, new_text)
    except (FileInspectionError, OSError, UnicodeError, ValueError) as error:
        return {"ok": False, "error_kind": "invalid_file_edit", "error": str(error)}
