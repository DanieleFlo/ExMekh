from __future__ import annotations

from dataclasses import dataclass

from qs.core.agents.capability.markdown_capability import MarkdownCapability
from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.exit_skill.exit_skill import ExitSkill
from qs.core.agents.exit_tool.exit_tool import ExitTool
from qs.core.agents.hds.hds import HDS
from qs.core.models.tool import Tool


@dataclass(frozen=True, slots=True)
class CapabilitiesData:
    deep_skills: tuple[DeepSkill, ...]
    tools: tuple[Tool, ...]
    hds: tuple[HDS, ...] = ()
    exit_skills: tuple[ExitSkill, ...] = ()
    exit_tools: tuple[ExitTool, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.deep_skills, tuple):
            raise TypeError("deep_skills must be a tuple")

        if not all(
            isinstance(deep_skill, DeepSkill) for deep_skill in self.deep_skills
        ):
            raise TypeError("deep_skills must contain only DeepSkill instances")

        if not isinstance(self.tools, tuple):
            raise TypeError("tools must be a tuple")

        if not all(
            isinstance(tool, Tool)
            and not isinstance(tool, (MarkdownCapability, ExitSkill))
            for tool in self.tools
        ):
            raise TypeError("tools must contain only Tool instances")

        if not isinstance(self.hds, tuple):
            raise TypeError("hds must be a tuple")

        if not all(isinstance(hds, HDS) for hds in self.hds):
            raise TypeError("hds must contain only HDS instances")

        if not isinstance(self.exit_skills, tuple):
            raise TypeError("exit_skills must be a tuple")

        if not all(
            isinstance(exit_skill, ExitSkill) for exit_skill in self.exit_skills
        ):
            raise TypeError("exit_skills must contain only ExitSkill instances")

        if not isinstance(self.exit_tools, tuple):
            raise TypeError("exit_tools must be a tuple")

        if not all(isinstance(exit_tool, ExitTool) for exit_tool in self.exit_tools):
            raise TypeError("exit_tools must contain only ExitTool instances")

    def get_all_tools(self) -> tuple[DeepSkill | Tool | ExitSkill | ExitTool, ...]:
        return self.deep_skills + self.tools + self.exit_skills + self.exit_tools
