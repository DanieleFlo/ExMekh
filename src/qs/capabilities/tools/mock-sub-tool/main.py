from qs.core.agents.tools.decorator import deep_tool


@deep_tool(
    name="mock-sub-tool",
    description="Restituisce un risultato disponibile solo nella mock skill.",
)
def run(text: str) -> str:
    return f"Mock sub-tool result: {text}"
