from __future__ import annotations

from collections.abc import Sequence

from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.defaults import (
    DeepSkillOverride,
    DefaultCapabilities,
    ToolOverride,
)
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.models.tool import Tool


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
                resolved[target] = source.with_name(target)
            elif isinstance(replacement, DeepSkillOverride):
                source = self._find_source(
                    loaded.deep_skills, replacement.replacement_name, "deep skill"
                )
                resolved[target] = source.copy(name=target)
            else:
                resolved[target] = replacement

        return self._defaults.with_overrides(resolved)

    @staticmethod
    def _find_source[Resource: Tool](
        resources: Sequence[Resource], name: str, label: str
    ) -> Resource:
        matches = [resource for resource in resources if resource.get_name() == name]
        if not matches:
            raise ValueError(
                f"Override {label} not found in capabilities_paths: {name}"
            )
        if len(matches) > 1:
            raise ValueError(f"Override {label} name is ambiguous: {name}")
        return matches[0]
