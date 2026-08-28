from __future__ import annotations

from dataclasses import dataclass

from qs.core.agents.capability.markdown import CapabilityMarkdown
from qs.core.agents.hds.metadata import HDSMetadata


@dataclass(frozen=True, slots=True)
class HDSMarkdown(CapabilityMarkdown[HDSMetadata]):
    pass
