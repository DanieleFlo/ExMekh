from __future__ import annotations

import inspect
from collections.abc import Mapping

from exmekh_lib.core.agents.capabilities.index import CapabilitiesIndex
from exmekh_lib.core.agents.deep_skill.catalog import SkillCatalog, SkillCatalogEntry
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.models.annotation_schema import AnnotationSchemaBuilder
from exmekh_lib.core.models.json_schema import (
    Schema,
    SchemaDescription,
    schema_descriptions,
)
from exmekh_lib.core.models.tool import Tool


class SkillCatalogAttacher:
    def __init__(self, index: CapabilitiesIndex) -> None:
        self._index = index
        self._schema_builder = AnnotationSchemaBuilder()

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

    def _build_entries(
        self,
        names: tuple[str, ...],
        resources: Mapping[str, Tool],
    ) -> tuple[SkillCatalogEntry, ...]:
        return tuple(
            SkillCatalogEntry(
                name=name,
                description=resources[name].get_description(),
                parameters=self._parameter_descriptions(resources[name]),
            )
            for name in names
        )

    def _parameter_descriptions(self, tool: Tool) -> tuple[SchemaDescription, ...]:
        hints = tool.get_type_hints()
        properties: Schema = {}
        for name, parameter in tool.get_signature().parameters.items():
            if tool.is_execution_argument(name):
                continue
            annotation = hints.get(name, parameter.annotation)
            if annotation is not inspect.Parameter.empty:
                properties[name] = self._schema_builder.build(annotation)
        return schema_descriptions({"properties": properties})
