from qs.config.capabilities import CAPABILITY_FORMAT, CAPABILITY_TEXT
from qs.config.runtime import RUNTIME
from qs.core.agents.exit_tool.data import ExitToolData
from qs.core.agents.exit_tool.decorator import exit_tool


@exit_tool(
    name=RUNTIME.exit_tool_name,
    instructions_path=CAPABILITY_FORMAT.exit_tool_instructions_filename,
    description=CAPABILITY_TEXT.exit_tool_description,
)
def exit_agent(text: str, items: list[str]) -> ExitToolData:
    return ExitToolData(text=text, items=items)
