from pathlib import Path

from qs.core.agents.tools.decorator import deep_tool


@deep_tool(
    name="mock-tool",
    description="Restituisce il testo ricevuto come risultato dimostrativo.",
)
def run(work_root: Path, text: str) -> str:
    del work_root
    return f"Mock tool result: {text}"
