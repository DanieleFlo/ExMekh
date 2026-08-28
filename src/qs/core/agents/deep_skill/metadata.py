from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from qs.core.agents.capability.metadata import (
    CapabilityMetadata,
    CapabilityMetadataParser,
    MetadataFieldSpec,
)


@dataclass(frozen=True, slots=True)
class DeepSkillMetadata(CapabilityMetadata):
    inherits_parent_tools: bool


class DeepSkillMetadataParser(CapabilityMetadataParser):
    INHERITS_PARENT_TOOLS_FIELD = MetadataFieldSpec("inherits-parent-tools")

    @property
    def field_specs(self) -> tuple[MetadataFieldSpec, ...]:
        return (
            *super().field_specs,
            self.INHERITS_PARENT_TOOLS_FIELD,
        )

    def parse(self, data: dict[str, Any]) -> DeepSkillMetadata:
        return cast(DeepSkillMetadata, super().parse(data))

    def _build_metadata(self, data: dict[str, Any]) -> DeepSkillMetadata:
        common = super()._build_metadata(data)
        return DeepSkillMetadata(
            description=common.description,
            sub_skills=common.sub_skills,
            sub_tools=common.sub_tools,
            inherits_parent_tools=self._parse_boolean(
                data,
                self.INHERITS_PARENT_TOOLS_FIELD,
                default=False,
            ),
        )
