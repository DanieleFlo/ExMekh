from __future__ import annotations

from dataclasses import dataclass

from exmekh_lib.core.agents.capability.markdown import CapabilityMarkdown
from exmekh_lib.core.agents.hds.metadata import HDSMetadata


@dataclass(frozen=True, slots=True)
class HDSMarkdown(CapabilityMarkdown[HDSMetadata]):
    pass
