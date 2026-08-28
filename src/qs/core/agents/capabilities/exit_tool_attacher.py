from qs.core.agents.exit_skill.exit_skill import ExitSkill
from qs.core.agents.exit_tool.exit_tool import ExitTool


class ExitToolAttacher:
    def __init__(self, exit_tools: tuple[ExitTool, ...]) -> None:
        self._exit_tools = exit_tools

    def attach_to_all(self, exit_skills: tuple[ExitSkill, ...]) -> None:
        for exit_skill in exit_skills:
            tool_names = tuple(
                tool.name
                for tool in self._exit_tools
                if tool.get_exit_skill_name() == exit_skill.name
            )
            exit_skill.attach_tool_names(tool_names)
