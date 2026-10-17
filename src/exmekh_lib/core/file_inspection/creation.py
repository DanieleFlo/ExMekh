"""Create ordered workspace resources with independent per-entry outcomes."""

from pathlib import Path

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.models.creation_document import CreationDocument, CreationElement
from exmekh_lib.utils.tool_errors import ToolInputError, tool_error_result

from .edit_filesystem import new_directory, new_file


def create_resources(work_root: Path, document: str) -> dict[str, object]:
    try:
        parsed = CreationDocument.parse(document)
    except ToolInputError as error:
        return tool_error_result(error, "invalid_file_creation")
    results: list[dict[str, object]] = []
    for index, element in enumerate(parsed.elements, 1):
        result = (
            tool_error_result(element, "invalid_file_creation")
            if isinstance(element, ToolInputError)
            else _create_resource(work_root, element)
        )
        results.append({"index": index, **result})
    return {"ok": all(result["ok"] for result in results), "results": results}


def _create_resource(work_root: Path, element: CreationElement) -> dict[str, object]:
    try:
        result = (
            new_file(work_root, element.path, element.content)
            if element.resource_type == "file"
            else new_directory(work_root, element.path)
        )
    except (OSError, ValueError) as error:
        result = tool_error_result(error, "invalid_file_creation")
    if element.resource_type == "directory" and element.content:
        result["warnings"] = [FILE_EDITING.directory_content_ignored]
    return {**result, "path": element.path, "resource_type": element.resource_type}
