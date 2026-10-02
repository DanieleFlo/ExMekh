from __future__ import annotations

from collections.abc import Mapping

from qs.core.agents.capabilities.index import CapabilitiesIndex
from qs.core.agents.deep_skill.catalog import SkillCatalog, SkillCatalogEntry
from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.models.tool import Tool


class SkillCatalogAttacher:
    def __init__(self, index: CapabilitiesIndex) -> None:
        self._index = index

    def attach_to_all(self, deep_skills: tuple[DeepSkill, ...]) -> None:
        for deep_skill in deep_skills:
            deep_skill.attach_catalog(self._build_catalog(deep_skill))

    def _build_catalog(self, deep_skill: DeepSkill) -> SkillCatalog:
        return SkillCatalog(
            sub_skills=self._build_entries(
                deep_skill.get_sub_skill_names(),
                self._index.deep_skills,
            ),
            sub_tools=self._build_entries(
                deep_skill.get_sub_tool_names(),
                self._index.tools,
            ),
        )

    @staticmethod
    def _build_entries(
        names: tuple[str, ...],
        resources: Mapping[str, Tool],
    ) -> tuple[SkillCatalogEntry, ...]:
        return tuple(
            SkillCatalogEntry(
                name=name,
                description=resources[name].get_description(),
            )
            for name in names
        )
