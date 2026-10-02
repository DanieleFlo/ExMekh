from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.agents.capability.metadata import (
    CapabilityMetadata,
    CapabilityMetadataParser,
    MetadataFieldSpec,
)


@dataclass(frozen=True, slots=True)
class DeepSkillMetadata(CapabilityMetadata):
    inherits_parent_tools: bool

    def get_inherits_parent_tools(self) -> bool:
        return self.inherits_parent_tools


class DeepSkillMetadataParser(CapabilityMetadataParser):
    INHERITS_PARENT_TOOLS_FIELD = MetadataFieldSpec(
        CAPABILITY_FORMAT.inherits_parent_tools_field
    )

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
            description=common.get_description(),
            sub_skills=common.get_sub_skill_names(),
            sub_tools=common.get_sub_tool_names(),
            inherits_parent_tools=self._parse_boolean(
                data,
                self.INHERITS_PARENT_TOOLS_FIELD,
                default=False,
            ),
        )
