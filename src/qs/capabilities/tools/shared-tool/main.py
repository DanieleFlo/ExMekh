from pathlib import Path

from qs.core.agents.tools.decorator import deep_tool


@deep_tool(
    name="shared-tool",
    description="Restituisce un risultato condiviso tra mock skill e nested skill.",
)
def run(work_root: Path, text: str) -> str:
    del work_root
    return f"Shared tool result: {text}"
