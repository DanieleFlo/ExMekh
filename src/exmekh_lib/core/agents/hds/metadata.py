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
class HDSMetadata(CapabilityMetadata):
    exit_tool: str
    agent_context_id: str | None = None

    def get_exit_tool_name(self) -> str:
        return self.exit_tool

    def get_agent_context_id(self) -> str | None:
        return self.agent_context_id


EXIT_TOOL_FIELD = MetadataFieldSpec(CAPABILITY_FORMAT.exit_tool_field, required=True)
AGENT_CONTEXT_ID_FIELD = MetadataFieldSpec(CAPABILITY_FORMAT.agent_context_id_field)


class HDSMetadataParser(CapabilityMetadataParser):
    @property
    def field_specs(self) -> tuple[MetadataFieldSpec, ...]:
        return (*super().field_specs, EXIT_TOOL_FIELD, AGENT_CONTEXT_ID_FIELD)

    def parse(self, data: dict[str, Any]) -> HDSMetadata:
        return cast(HDSMetadata, super().parse(data))

    def _build_metadata(self, data: dict[str, Any]) -> HDSMetadata:
        common = super()._build_metadata(data)
        return HDSMetadata(
            description=common.get_description(),
            sub_skills=common.get_sub_skill_names(),
            sub_tools=common.get_sub_tool_names(),
            exit_tool=self._parse_string(data, EXIT_TOOL_FIELD),
            agent_context_id=self._parse_optional_string(data, AGENT_CONTEXT_ID_FIELD),
        )
