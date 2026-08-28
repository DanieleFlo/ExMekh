from __future__ import annotations

from qs.core.agents.capabilities.resource_type import ResourceType
from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.models.assistant_message import ToolCall


class ExitSkill(DeepSkill):
    resource_type = ResourceType.EXIT_SKILL
    _TAG = "exit-skill"

    def get_body(self) -> str:
        return self.body.strip()

    def render_prompt_section(self) -> str:
        return f"<{self._TAG}>{self.get_body()}</{self._TAG}>"


class CalledExitSkill:
    def __init__(self, exit_skill: ExitSkill, tool_call: ToolCall) -> None:
        if not isinstance(exit_skill, ExitSkill):
            raise TypeError("exit_skill must be an ExitSkill")
        if not isinstance(tool_call, ToolCall):
            raise TypeError("tool_call must be a ToolCall")

        self.exit_skill = exit_skill
        self.tool_call = tool_call
