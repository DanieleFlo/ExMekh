from qs.core.agents.tools.decorator import deep_tool


@deep_tool(
    name="mock-tool",
    description="Restituisce il testo ricevuto come risultato dimostrativo.",
)
def run(text: str) -> str:
    return f"Mock tool result: {text}"
