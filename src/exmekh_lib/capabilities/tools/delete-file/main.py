"""Delete workspace files and directories with independent per-path outcomes."""

from pathlib import Path

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.deletion import delete_resources


@deep_tool(
    name="delete-file",
    description=FILE_EDITING.delete_file_description,
    background_enabled=False,
)
def run(work_root: Path, paths: list[str]) -> dict[str, object]:
    return delete_resources(work_root, paths)
