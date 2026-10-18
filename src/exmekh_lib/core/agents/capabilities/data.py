from __future__ import annotations

import logging
from dataclasses import dataclass

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.capabilities import CAPABILITY_TEXT
from exmekh_lib.core.agents.capability.markdown_capability import MarkdownCapability
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.models.tool import Tool

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CapabilitiesData:
    _SKILL_ONLY_AVAILABILITY = CAPABILITY_TEXT.skill_only_availability

    deep_skills: tuple[DeepSkill, ...]
    tools: tuple[Tool, ...]
    hds: tuple[HDS, ...] = ()
    exit_tools: tuple[ExitTool, ...] = ()

    def __post_init__(self) -> None:
        self._validate_items(self.deep_skills, "deep_skills", DeepSkill)
        self._validate_items(self.tools, "tools", Tool)
        if any(isinstance(tool, (MarkdownCapability, ExitTool)) for tool in self.tools):
            raise TypeError(AGENT_TEXT.tools_must_contain_only_tool_instances)
        self._validate_items(self.hds, "hds", HDS)
        self._validate_items(self.exit_tools, "exit_tools", ExitTool)

    @staticmethod
    def _validate_items(
        items: object,
        field_name: str,
        item_type: type[object],
    ) -> None:
        if not isinstance(items, tuple):
            raise TypeError(AGENT_TEXT.must_be_a_tuple.format(field_name=field_name))
        if not all(isinstance(item, item_type) for item in items):
            raise TypeError(
                AGENT_TEXT.must_contain_only_instances.format(
                    field_name=field_name, item_type_name=item_type.__name__
                )
            )

    def get_all_tools(self) -> tuple[Tool, ...]:
        all_tools = self.deep_skills + self.tools + self.exit_tools
        logger.debug(AGENT_TEXT.available_tools, all_tools)
        return all_tools

    def get_contextual_tools(self, hds: HDS) -> tuple[Tool, ...]:
        direct_tool_names = hds.get_direct_tool_names()
        skill_availability = self.get_skill_availability()
        tools = self.get_all_tools()
        contextual_tools = tuple(
            self._prepare_contextual_tool(
                tool,
                direct_tool_names,
                skill_availability,
            )
            for tool in tools
        )
        for original, contextual in zip(tools, contextual_tools, strict=True):
            logger.debug(
                AGENT_TEXT.contextual_capability_hds_type_name_description,
                hds.get_name(),
                type(original).__name__,
                contextual.get_name(),
                contextual.get_description(),
            )
        return contextual_tools

    def _prepare_contextual_tool(
        self,
        tool: Tool,
        direct_tool_names: set[str],
        skill_availability: dict[str, tuple[str, ...]],
    ) -> Tool:
        if isinstance(tool, ExitTool):
            return tool.with_description_suffix(self._SKILL_ONLY_AVAILABILITY)

        if tool.get_name() in direct_tool_names:
            return tool

        skill_names = skill_availability.get(tool.get_name(), [])
        if not skill_names:
            description = ""
        elif len(skill_names) == 1:
            description = AGENT_TEXT.available_in_skill.format(
                skill_names=skill_names[0]
            )
        else:
            names = ", ".join(f"'{name}'" for name in skill_names)
            description = AGENT_TEXT.available_in_skills.format(names=names)
        contextual = tool.with_description(description)
        if skill_names and not isinstance(tool, DeepSkill):
            return contextual.without_parameter_descriptions()
        return contextual

    def get_hds(self, name: str) -> HDS:
        for hds in self.hds:
            if hds.get_name() == name:
                return hds
        available_hds = ", ".join(hds.get_name() for hds in self.hds) or "none"
        raise ValueError(
            AGENT_TEXT.hds_not_found_available_hds.format(
                name=name, available_hds=available_hds
            )
        )

    def get_exit_tool(self, name: str) -> ExitTool:
        for tool in self.exit_tools:
            if tool.get_name() == name:
                return tool
        raise ValueError(AGENT_TEXT.exit_tool_not_found.format(name=name))

    def get_skill_availability(self) -> dict[str, tuple[str, ...]]:
        availability: dict[str, list[str]] = {}
        for skill in self.deep_skills:
            for tool_name in skill.get_direct_tool_names():
                availability.setdefault(tool_name, []).append(skill.get_name())

        return {
            tool_name: tuple(skill_names)
            for tool_name, skill_names in availability.items()
        }

    def __str__(self) -> str:
        description = AGENT_TEXT.hds_deepskills_tools_exittools.format(
            hds=self.hds,
            deep_skills=self.deep_skills,
            tools=self.tools,
            exit_tools=self.exit_tools,
        )
        return description
