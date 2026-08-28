from __future__ import annotations

from dataclasses import dataclass

from qs.core.agents.capability.metadata import CapabilityMetadata


@dataclass(frozen=True, slots=True)
class CapabilityMarkdown[MetadataType: CapabilityMetadata]:
    metadata: MetadataType
    body: str
    references: tuple[str, ...]
