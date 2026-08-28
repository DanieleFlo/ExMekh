from qs.core.agents.exit_tool.data import ExitToolData
from qs.core.agents.exit_tool.decorator import exit_tool as _exit_tool
from qs.core.agents.exit_tool.exit_tool import ExitTool

exit_tool = _exit_tool

__all__ = ["ExitTool", "ExitToolData", "exit_tool"]
