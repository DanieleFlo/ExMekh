from __future__ import annotations

from dataclasses import dataclass

from qs.core.agents.capability.markdown import CapabilityMarkdown
from qs.core.agents.deep_skill.metadata import DeepSkillMetadata


@dataclass(frozen=True, slots=True)
class SkillMarkdown(CapabilityMarkdown[DeepSkillMetadata]):
    pass
