from pathlib import Path

from exmekh_lib.config.capabilities import CAPABILITY_TEXT
from exmekh_lib.core.agents.tools.decorator import deep_tool


@deep_tool(
    name="shared-tool",
    description=CAPABILITY_TEXT.shared_tool_description,
)
def run(work_root: Path, text: str) -> str:
    del work_root
    return CAPABILITY_TEXT.shared_tool_result.format(text=text)
