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
        self._validate_items(self.deep_skills, "deep_skills", DeepSkill)
        self._validate_items(self.tools, "tools", Tool)
        if any(
            isinstance(tool, (MarkdownCapability, ExitSkill)) for tool in self.tools
        ):
            raise TypeError("tools must contain only Tool instances")
        self._validate_items(self.hds, "hds", HDS)
        self._validate_items(self.exit_skills, "exit_skills", ExitSkill)
        self._validate_items(self.exit_tools, "exit_tools", ExitTool)

    @staticmethod
    def _validate_items(
        items: object,
        field_name: str,
        item_type: type[object],
    ) -> None:
        if not isinstance(items, tuple):
            raise TypeError(f"{field_name} must be a tuple")
        if not all(isinstance(item, item_type) for item in items):
            raise TypeError(
                f"{field_name} must contain only {item_type.__name__} instances"
            )

    def get_all_tools(self) -> tuple[DeepSkill | Tool | ExitSkill | ExitTool, ...]:
        return self.deep_skills + self.tools + self.exit_skills + self.exit_tools

    def get_skill_availability(self) -> dict[str, list[str]]:
        availability: dict[str, list[str]] = {}
        skills = self.deep_skills + self.exit_skills

        for skill in skills:
            for tool_name in skill.get_direct_tool_names():
                availability.setdefault(tool_name, []).append(skill.name)

        for exit_tool in self.exit_tools:
            availability.setdefault(exit_tool.name, []).append(
                exit_tool.get_exit_skill_name()
            )

        return availability
