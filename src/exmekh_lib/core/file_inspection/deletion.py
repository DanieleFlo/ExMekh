"""Delete workspace resources in order without undoing earlier successes."""

from pathlib import Path

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.utils.tool_errors import tool_error_result

from .edit_filesystem import delete_file
from .errors import FileInspectionError


def delete_resources(work_root: Path, paths: list[str]) -> dict[str, object]:
    if not isinstance(paths, list) or not paths:
        return tool_error_result(
            FileInspectionError(FILE_EDITING.deletion_paths_required),
            "invalid_file_deletion",
        )
    results: list[dict[str, object]] = []
    for path in paths:
        try:
            result = delete_file(work_root, path)
        except (OSError, ValueError) as error:
            result = tool_error_result(error, "invalid_file_deletion")
        results.append({**result, "path": path})
    return {"ok": all(result["ok"] for result in results), "results": results}
