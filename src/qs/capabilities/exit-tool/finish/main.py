from qs.core.agents.exit_tool.data import ExitToolData
from qs.core.agents.exit_tool.decorator import exit_tool


@exit_tool(
    name="exit_agent",
    description="Termina l'esecuzione restituendo la risposta finale.",
)
def exit_agent(text: str, items: list[str]) -> ExitToolData:
    return ExitToolData(text=text, items=items)
