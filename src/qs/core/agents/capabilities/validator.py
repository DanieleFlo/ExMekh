from __future__ import annotations

from collections.abc import Set as AbstractSet

from qs.core.agents.capabilities.index import CapabilitiesIndex
from qs.core.agents.capability.markdown_capability import MarkdownCapability
from qs.core.agents.hds.hds import HDS


class CapabilitiesValidator:
    def __init__(self, index: CapabilitiesIndex) -> None:
        self._index = index

    def validate(self, resources: tuple[MarkdownCapability, ...]) -> None:
        for tool in self._index.exit_tools.values():
            self._validate_reference_group(
                f"Exit tool '{tool.name}'",
                tool.references,
                self._index.reference_names,
                "references",
            )
        for resource in resources:
            self._validate_resource(resource)

    def _validate_resource(self, resource: MarkdownCapability) -> None:
        label = f"{resource.resource_type} '{resource.name}'"
        self._validate_reference_group(
            label,
            resource.references,
            self._index.reference_names,
            "references",
        )
        self._validate_reference_group(
            label,
            resource.get_sub_skill_names(),
            self._index.deep_skills.keys(),
            "sub-skills",
        )
        self._validate_reference_group(
            label,
            resource.get_sub_tool_names(),
            self._index.tools.keys(),
            "sub-tools",
        )

        if isinstance(resource, HDS):
            self._validate_reference_group(
                label,
                (resource.get_exit_tool_name(),),
                self._index.exit_tools.keys(),
                "exit-tool",
            )

    @staticmethod
    def _validate_reference_group(
        label: str,
        names: tuple[str, ...],
        available_names: AbstractSet[str],
        group: str,
    ) -> None:
        invalid_names = set(names) - available_names

        if invalid_names:
            invalid = ", ".join(sorted(invalid_names))
            raise ValueError(f"{label} has invalid {group}: {invalid}")
