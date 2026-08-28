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
            description=common.description,
            sub_skills=common.sub_skills,
            sub_tools=common.sub_tools,
            exit_skill=self._parse_string(data, EXIT_SKILL_FIELD),
        )
