from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from qs.core.agents.capability.metadata import (
    CapabilityMetadata,
    CapabilityMetadataParser,
    MetadataFieldSpec,
)


@dataclass(frozen=True, slots=True)
class HDSMetadata(CapabilityMetadata):
    exit_skill: str

    def get_direct_tool_names(self) -> set[str]:
        direct_tool_names = CapabilityMetadata.get_direct_tool_names(self)
        return {*direct_tool_names, self.exit_skill}

    def get_exit_skill_name(self) -> str:
        return self.exit_skill


EXIT_SKILL_FIELD = MetadataFieldSpec("exit-skill", required=True)


class HDSMetadataParser(CapabilityMetadataParser):
    @property
    def field_specs(self) -> tuple[MetadataFieldSpec, ...]:
        return (*super().field_specs, EXIT_SKILL_FIELD)

    def parse(self, data: dict[str, Any]) -> HDSMetadata:
        return cast(HDSMetadata, super().parse(data))

    def _build_metadata(self, data: dict[str, Any]) -> HDSMetadata:
        common = super()._build_metadata(data)
        return HDSMetadata(
            description=common.get_description(),
            sub_skills=common.get_sub_skill_names(),
            sub_tools=common.get_sub_tool_names(),
            exit_skill=self._parse_string(data, EXIT_SKILL_FIELD),
        )
