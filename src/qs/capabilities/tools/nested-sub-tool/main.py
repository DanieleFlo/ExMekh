from qs.core.agents.tools.decorator import deep_tool


@deep_tool(
    name="nested-sub-tool",
    description="Restituisce un risultato disponibile solo nella nested skill.",
)
def run(text: str) -> str:
    return f"Nested sub-tool result: {text}"
