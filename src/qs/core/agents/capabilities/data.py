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
    _SKILL_ONLY_AVAILABILITY = "Available in every skill."

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

    def get_contextual_tools(self, hds: HDS) -> tuple[Tool, ...]:
        direct_tool_names = hds.get_direct_tool_names()
        skill_availability = self.get_skill_availability()
        return tuple(
            self._prepare_contextual_tool(
                tool,
                direct_tool_names,
                skill_availability,
            )
            for tool in self.get_all_tools()
        )

    def _prepare_contextual_tool(
        self,
        tool: Tool,
        direct_tool_names: set[str],
        skill_availability: dict[str, tuple[str, ...]],
    ) -> Tool:
        if isinstance(tool, (ExitSkill, ExitTool)):
            return tool.with_description_suffix(self._SKILL_ONLY_AVAILABILITY)

        if tool.name in direct_tool_names:
            return tool

        skill_names = skill_availability.get(tool.name, [])
        if not skill_names:
            description = ""
        elif len(skill_names) == 1:
            description = f"Available in skill '{skill_names[0]}'."
        else:
            names = ", ".join(f"'{name}'" for name in skill_names)
            description = f"Available in skills {names}."

        return tool.with_description(description)

    def get_hds(self, name: str) -> HDS:
        for hds in self.hds:
            if hds.name == name:
                return hds
        available_hds = ", ".join(hds.name for hds in self.hds) or "none"
        raise ValueError(f"HDS '{name}' not found. Available HDS: {available_hds}")

    def get_exit_skill(self, name: str) -> ExitSkill:
        for exit_skill in self.exit_skills:
            if exit_skill.name == name:
                return exit_skill
        raise ValueError(f"Exit skill '{name}' not found")

    def get_exit_tool_names(self, exit_skill_name: str) -> tuple[str, ...]:
        return tuple(
            exit_tool.name
            for exit_tool in self.exit_tools
            if exit_tool.get_exit_skill_name() == exit_skill_name
        )

    def get_skill_availability(self) -> dict[str, tuple[str, ...]]:
        availability: dict[str, list[str]] = {}
        skills = self.deep_skills + self.exit_skills

        for skill in skills:
            for tool_name in skill.get_direct_tool_names():
                availability.setdefault(tool_name, []).append(skill.name)

        return {
            tool_name: tuple(skill_names)
            for tool_name, skill_names in availability.items()
        }
