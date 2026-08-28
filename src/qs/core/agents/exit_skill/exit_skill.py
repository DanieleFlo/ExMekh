from __future__ import annotations

from pathlib import Path

from qs.core.agents.capabilities.resource_type import ResourceType
from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.deep_skill.metadata import DeepSkillMetadata
from qs.core.models.assistant_message import ToolCall


class ExitSkill(DeepSkill):
    resource_type = ResourceType.EXIT_SKILL
    _TAG = "exit-skill"

    def __init__(
        self,
        *,
        name: str,
        markdown_path: Path,
        metadata: DeepSkillMetadata,
        body: str,
        references: tuple[str, ...],
    ) -> None:
        super().__init__(
            name=name,
            markdown_path=markdown_path,
            metadata=metadata,
            body=body,
            references=references,
        )
        self._tool_names: tuple[str, ...] = ()
        self._tools_attached = False

    def get_body(self) -> str:
        return self.body.strip()

    def render_prompt_section(self) -> str:
        tools = ", ".join(self._tool_names)
        return f'<{self._TAG} tools="{tools}">{self.get_body()}</{self._TAG}>'

    def attach_tool_names(self, tool_names: tuple[str, ...]) -> None:
        if not isinstance(tool_names, tuple):
            raise TypeError("tool_names must be a tuple")
        if not all(isinstance(name, str) and name for name in tool_names):
            raise TypeError("tool_names must contain only non-empty strings")
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tool_names cannot contain duplicates")
        if self._tools_attached:
            raise RuntimeError(
                f"Exit tools already attached to exit skill '{self.name}'"
            )
        self._tool_names = tool_names
        self._tools_attached = True


class CalledExitSkill:
    def __init__(self, exit_skill: ExitSkill, tool_call: ToolCall) -> None:
        if not isinstance(exit_skill, ExitSkill):
            raise TypeError("exit_skill must be an ExitSkill")
        if not isinstance(tool_call, ToolCall):
            raise TypeError("tool_call must be a ToolCall")

        self._exit_skill = exit_skill
        self._tool_call = tool_call

    @property
    def name(self) -> str:
        return self._exit_skill.name

    @property
    def call_id(self) -> str:
        return self._tool_call.get_id()

    def render_instructions(self) -> str:
        return self._exit_skill.get_body()
