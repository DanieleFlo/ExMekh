from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar

from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.models.tool import Tool

Resource = TypeVar("Resource", bound=Tool)


@dataclass(frozen=True, slots=True)
class CapabilitiesIndex:
    deep_skills: MappingProxyType[str, DeepSkill]
    tools: MappingProxyType[str, Tool]
    hds: MappingProxyType[str, HDS]
    exit_tools: MappingProxyType[str, ExitTool]
    reference_names: frozenset[str]

    @classmethod
    def build(cls, data: CapabilitiesData) -> CapabilitiesIndex:
        if not isinstance(data, CapabilitiesData):
            raise TypeError("data must be CapabilitiesData")

        deep_skills = cls._index_by_name(data.deep_skills, "deep-skill")
        tools = cls._index_by_name(data.tools, "tool")
        hds = cls._index_by_name(data.hds, "HDS")
        exit_tools = cls._index_by_name(data.exit_tools, "exit-tool")
        cls._validate_distinct_names(deep_skills, tools, hds, exit_tools)

        return cls(
            deep_skills=MappingProxyType(deep_skills),
            tools=MappingProxyType(tools),
            hds=MappingProxyType(hds),
            exit_tools=MappingProxyType(exit_tools),
            reference_names=frozenset(deep_skills | tools | hds | exit_tools),
        )

    @staticmethod
    def _index_by_name(
        resources: tuple[Resource, ...],
        resource_type: str,
    ) -> dict[str, Resource]:
        indexed_resources: dict[str, Resource] = {}

        for resource in resources:
            name = resource.get_name()
            if name in indexed_resources:
                raise ValueError(f"Duplicate {resource_type} name: '{name}'")

            indexed_resources[name] = resource

        return indexed_resources

    @staticmethod
    def _validate_distinct_names(
        deep_skills: dict[str, DeepSkill],
        tools: dict[str, Tool],
        hds: dict[str, HDS],
        exit_tools: dict[str, ExitTool],
    ) -> None:
        duplicate_names = set(deep_skills) & set(tools)

        if duplicate_names:
            duplicates = ", ".join(sorted(duplicate_names))
            raise ValueError(f"Names shared by deep skills and tools: {duplicates}")

        duplicate_names = set(hds) & (set(deep_skills) | set(tools))

        if duplicate_names:
            duplicates = ", ".join(sorted(duplicate_names))
            raise ValueError(
                f"Names shared by HDS and other capabilities: {duplicates}"
            )

        duplicate_names = set(exit_tools) & (set(deep_skills) | set(tools) | set(hds))

        if duplicate_names:
            duplicates = ", ".join(sorted(duplicate_names))
            raise ValueError(
                f"Names shared by exit tools and other capabilities: {duplicates}"
            )
