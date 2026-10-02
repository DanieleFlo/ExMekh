from __future__ import annotations

from dataclasses import dataclass

from exmekh_lib.core.agents.capability.markdown import CapabilityMarkdown
from exmekh_lib.core.agents.deep_skill.metadata import DeepSkillMetadata


@dataclass(frozen=True, slots=True)
class SkillMarkdown(CapabilityMarkdown[DeepSkillMetadata]):
    pass
