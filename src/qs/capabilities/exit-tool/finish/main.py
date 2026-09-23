from qs.core.agents.exit_tool.data import ExitToolData
from qs.core.agents.exit_tool.decorator import exit_tool


@exit_tool(
    name="exit_agent",
    instructions_path="instructions.md",
    description="Finishes the current skill and returns its result to the caller.",
)
def exit_agent(text: str, items: list[str]) -> ExitToolData:
    return ExitToolData(text=text, items=items)
