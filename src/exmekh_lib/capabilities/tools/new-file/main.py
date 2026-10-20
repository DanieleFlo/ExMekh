"""Create workspace resources in order with independent per-entry outcomes."""

from pathlib import Path
from typing import Annotated

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.file_inspection.creation import create_resources


@deep_tool(
    name=DEFAULT_CAPABILITIES.new_file,
    description=FILE_EDITING.new_file_description,
    background_enabled=False,
)
def run(
    work_root: Path, document: Annotated[str, FILE_EDITING.new_document_description]
) -> dict[str, object]:
    return create_resources(work_root, document)
