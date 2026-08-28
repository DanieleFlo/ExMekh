from qs.core.agents.tools.decorator import deep_tool


@deep_tool(
    name="shared-tool",
    description="Restituisce un risultato condiviso tra mock skill e nested skill.",
)
def run(text: str) -> str:
    return f"Shared tool result: {text}"
