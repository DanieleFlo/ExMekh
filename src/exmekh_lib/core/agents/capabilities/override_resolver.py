from __future__ import annotations

from collections.abc import Sequence

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT, CAPABILITY_TEXT
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
                    loaded.tools,
                    replacement.replacement_name,
                    CAPABILITY_TEXT.tool_label,
                )
                resolved[target] = source.with_name(target)
            elif isinstance(replacement, DeepSkillOverride):
                source = self._find_source(
                    loaded.deep_skills,
                    replacement.replacement_name,
                    CAPABILITY_FORMAT.deep_skill_resource_type.lower(),
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
                AGENT_TEXT.override_not_found_in_capabilities_paths.format(
                    label=label, name=name
                )
            )
        if len(matches) > 1:
            raise ValueError(
                AGENT_TEXT.override_name_is_ambiguous.format(label=label, name=name)
            )
        return matches[0]
