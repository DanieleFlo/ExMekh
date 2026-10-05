"""Expose read-only workspace inspection through the standard tool runtime."""

from pathlib import Path

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.utils.tool_errors import missing_dependency


@deep_tool(
    name="inspect-file",
    description=(
        "Inspect one regular file inside the workroot before reading it. Returns "
        "metadata, physical line counts for decoded text, and bounded navigation "
        "regions with inclusive 1-based line ranges, without full file content. "
        "Supports Python symbols, Markdown sections/fences, structured data, CSV/TSV, "
        "archive indexes, PDF metadata and image headers. Other Unicode files get text statistics; "
        "unsupported formats and limits are explicit. max_items defaults to 50; "
        "use offset to paginate or query to rank the bounded index lexically. "
        "detailed adds technical metadata and applied limits. No extraction, OCR "
        "or model downloads. Standard background execution and output retrieval apply."
    ),
)
def run(
    work_root: Path,
    path: str,
    detailed: bool = False,
    max_items: int = FILE_INSPECTION.default_max_items,
    offset: int = 0,
    query: str | None = None,
) -> dict[str, object]:
    try:
        from exmekh_lib.core.file_inspection.service import inspect_file

        return inspect_file(
            work_root,
            path,
            detailed=detailed,
            max_items=max_items,
            offset=offset,
            query=query,
        )
    except ModuleNotFoundError as error:
        return missing_dependency(error)
    except FileInspectionError as error:
        return {
            "ok": False,
            "error_kind": "invalid_file_inspection",
            "error": str(error),
        }
