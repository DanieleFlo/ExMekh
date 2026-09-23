from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from types import MappingProxyType

from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.exit_tool.exit_tool import ExitTool
from qs.core.agents.hds.hds import HDS
from qs.core.models.tool import Tool

_MOCK_RESOURCE_TYPES: dict[str, type[Tool]] = {
    "main": HDS,
    "mock-sub-agent": HDS,
    "mock-skill": DeepSkill,
    "nested-skill": DeepSkill,
    "mock-tool": Tool,
    "mock-sub-tool": Tool,
    "nested-sub-tool": Tool,
    "shared-tool": Tool,
}
_HARNESS_RESOURCE_TYPES: dict[str, type[Tool]] = {
    "exit_agent": ExitTool,
    "get_execution_history": Tool,
    "search_saved_tool_outputs": Tool,
    "read_saved_tool_output": Tool,
    "call_sub_agent": Tool,
}
_RESOURCE_TYPES = MappingProxyType(_MOCK_RESOURCE_TYPES | _HARNESS_RESOURCE_TYPES)
MOCK_NAMES = frozenset(_MOCK_RESOURCE_TYPES)
HARNESS_NAMES = frozenset(_HARNESS_RESOURCE_TYPES)
RUNTIME_NAMES = HARNESS_NAMES - {"exit_agent"}
BUILTIN_CAPABILITIES_PATH = Path(__file__).resolve().parents[3] / "capabilities"


class DefaultCapabilities:
    """Select and replace the capabilities bundled with qs for one node."""

    def __init__(
        self,
        *,
        mocks: bool = False,
        harness: bool = True,
        include: Iterable[str] = (),
        exclude: Iterable[str] = (),
        overrides: Mapping[str, HDS | DeepSkill | ExitTool | Tool] | None = None,
    ) -> None:
        if not isinstance(mocks, bool) or not isinstance(harness, bool):
            raise TypeError("mocks and harness must be bool")
        included = self._validate_names(include, "include")
        excluded = self._validate_names(exclude, "exclude")
        replacements = self._validate_overrides(overrides)
        conflict = (included & excluded) | (replacements.keys() & excluded)
        if conflict:
            raise ValueError(
                "Capability cannot be included or overridden and excluded: "
                + ", ".join(sorted(conflict))
            )

        selected = (MOCK_NAMES if mocks else frozenset()) | (
            HARNESS_NAMES if harness else frozenset()
        )
        self._mocks = mocks
        self._harness = harness
        self._include = included
        self._exclude = excluded
        self._selected_names = frozenset(
            (selected | included | replacements.keys()) - excluded
        )
        self._overrides = MappingProxyType(replacements)

    @property
    def mocks(self) -> bool:
        return self._mocks

    @property
    def harness(self) -> bool:
        return self._harness

    @property
    def include(self) -> frozenset[str]:
        return self._include

    @property
    def exclude(self) -> frozenset[str]:
        return self._exclude

    @property
    def selected_names(self) -> frozenset[str]:
        return self._selected_names

    @property
    def overrides(self) -> Mapping[str, HDS | DeepSkill | ExitTool | Tool]:
        return self._overrides

    def includes(self, name: str) -> bool:
        return name in self._selected_names

    @staticmethod
    def _validate_names(names: Iterable[str], label: str) -> frozenset[str]:
        if isinstance(names, (str, bytes)) or not isinstance(names, Iterable):
            raise TypeError(f"{label} must be an iterable of capability names")
        values = tuple(names)
        if not all(isinstance(name, str) for name in values):
            raise TypeError(f"{label} must contain only strings")
        unknown = set(values) - MOCK_NAMES - HARNESS_NAMES
        if unknown:
            raise ValueError(
                f"Unknown default capabilities: {', '.join(sorted(unknown))}"
            )
        return frozenset(values)

    @classmethod
    def _validate_overrides(
        cls,
        overrides: Mapping[str, HDS | DeepSkill | ExitTool | Tool] | None,
    ) -> dict[str, HDS | DeepSkill | ExitTool | Tool]:
        if overrides is None:
            return {}
        if not isinstance(overrides, Mapping):
            raise TypeError("overrides must be a mapping")
        names = cls._validate_names(overrides.keys(), "overrides")
        replacements = dict(overrides)
        for name in names:
            resource = replacements[name]
            expected = _RESOURCE_TYPES[name]
            if not isinstance(resource, expected) or (
                expected is Tool and isinstance(resource, (HDS, DeepSkill, ExitTool))
            ):
                raise TypeError(f"Override for '{name}' must be a {expected.__name__}")
            if resource.name != name:
                raise ValueError(f"Override for '{name}' must have name '{name}'")
        return replacements
