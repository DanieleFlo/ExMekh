from __future__ import annotations

from collections.abc import Sequence

from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.capabilities.defaults import (
    DeepSkillOverride,
    DefaultCapabilities,
    ToolOverride,
)
from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.exit_tool.exit_tool import ExitTool
from qs.core.agents.hds.hds import HDS
from qs.core.models.tool import Tool


class _NamedOverrideResolver:
    """Resolve named replacements against resources loaded for one node."""

    def __init__(self, defaults: DefaultCapabilities) -> None:
        self._defaults = defaults

    def resolve(self, loaded: CapabilitiesData) -> DefaultCapabilities:
        if not any(
            isinstance(value, (ToolOverride, DeepSkillOverride))
            for value in self._defaults.overrides.values()
        ):
            return self._defaults

        resolved: dict[str, HDS | DeepSkill | ExitTool | Tool] = {}
        for target, replacement in self._defaults.overrides.items():
            if isinstance(replacement, ToolOverride):
                source = self._find_source(
                    loaded.tools, replacement.replacement_name, "tool"
                )
                resolved[target] = Tool(
                    source.tool,
                    name=target,
                    description=source.description,
                    cache_result=source.cache_result,
                )
            elif isinstance(replacement, DeepSkillOverride):
                source = self._find_source(
                    loaded.deep_skills, replacement.replacement_name, "deep skill"
                )
                resolved[target] = DeepSkill(
                    name=target,
                    markdown_path=source.markdown_path,
                    metadata=source.metadata,
                    body=source.body,
                    references=source.references,
                )
            else:
                resolved[target] = replacement

        return self._defaults._with_overrides(resolved)

    @staticmethod
    def _find_source[Resource: Tool](
        resources: Sequence[Resource], name: str, label: str
    ) -> Resource:
        matches = [resource for resource in resources if resource.name == name]
        if not matches:
            raise ValueError(
                f"Override {label} not found in capabilities_paths: {name}"
            )
        if len(matches) > 1:
            raise ValueError(f"Override {label} name is ambiguous: {name}")
        return matches[0]
