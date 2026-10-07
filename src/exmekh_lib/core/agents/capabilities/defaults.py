from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_hooks import ToolHooks

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
    "deep-thinking": DeepSkill,
    "file-processing": DeepSkill,
    "search-web": Tool,
    "read-web-page": Tool,
    "interact-web-page": Tool,
    "calculate": Tool,
    "date-time": Tool,
    "inspect-file": Tool,
    "read-file": Tool,
    "logic-check": Tool,
    "logic-consistency": Tool,
    "logic-counterexample": Tool,
    RUNTIME.exit_tool_name: ExitTool,
    RUNTIME.history_tool_name: Tool,
    RUNTIME.search_tool_name: Tool,
    RUNTIME.read_tool_name: Tool,
    RUNTIME.execution_tool_name: Tool,
    RUNTIME.bash_tool_name: Tool,
    RUNTIME.sub_agent_tool_name: Tool,
}
_RESOURCE_TYPES = MappingProxyType(_MOCK_RESOURCE_TYPES | _HARNESS_RESOURCE_TYPES)
MOCK_NAMES = frozenset(_MOCK_RESOURCE_TYPES)
HARNESS_NAMES = frozenset(_HARNESS_RESOURCE_TYPES)
RUNTIME_NAMES = RUNTIME.runtime_tool_names
_MAIN_HDS_DEPENDENCIES = frozenset(
    {
        RUNTIME.exit_tool_name,
        "deep-thinking",
        "file-processing",
        "search-web",
        "read-web-page",
        "interact-web-page",
        "calculate",
        "date-time",
        "inspect-file",
        "read-file",
        "logic-check",
        "logic-consistency",
        "logic-counterexample",
    }
)


@dataclass(frozen=True, slots=True)
class _NamedOverride:
    target_name: str
    replacement_name: str

    def __post_init__(self) -> None:
        for label, name in (
            ("target_name", self.target_name),
            ("replacement_name", self.replacement_name),
        ):
            if not isinstance(name, str):
                raise TypeError(f"{label} must be a string")
            if not name:
                raise ValueError(f"{label} cannot be empty")


@dataclass(frozen=True, slots=True)
class ToolOverride(_NamedOverride):
    """Replace a tool by the name of another loaded tool."""


@dataclass(frozen=True, slots=True)
class DeepSkillOverride(_NamedOverride):
    """Replace a deep skill by the name of another loaded deep skill."""


NamedOverride = ToolOverride | DeepSkillOverride
Override = HDS | DeepSkill | ExitTool | Tool | NamedOverride
OverrideInput = Mapping[str, Override] | Iterable[NamedOverride] | NamedOverride | None


class DefaultCapabilities:
    """Select bundled capabilities and customize loaded resources for one node."""

    CAPABILITIES_PATH = CAPABILITY_FORMAT.bundled_root

    def __init__(
        self,
        *,
        mocks: bool = False,
        harness: bool = True,
        include: Iterable[str] = (),
        exclude: Iterable[str] = (),
        overrides: OverrideInput = None,
        tool_hooks: Mapping[str, ToolHooks] | None = None,
    ) -> None:
        if not isinstance(mocks, bool) or not isinstance(harness, bool):
            raise TypeError("mocks and harness must be bool")
        included = self._validate_default_names(include, "include")
        excluded = self._validate_default_names(exclude, "exclude")
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
        if mocks:
            selected |= _MAIN_HDS_DEPENDENCIES
        selected_names = frozenset(
            (selected | included | replacements.keys()) - excluded
        )
        hooks = self._validate_tool_hooks(tool_hooks)
        unselected = hooks.keys() - selected_names
        if unselected:
            raise ValueError(
                "Tool hooks require selected capabilities: "
                + ", ".join(sorted(unselected))
            )
        self._mocks = mocks
        self._harness = harness
        self._include = included
        self._exclude = excluded
        self._selected_names = selected_names
        self._overrides = MappingProxyType(replacements)
        self._tool_hooks = MappingProxyType(hooks)

    @property
    def mocks(self) -> bool:
        return self._mocks

    @property
    def harness(self) -> bool:
        return self._harness

    @property
    def capabilities_path(self) -> Path:
        return self.CAPABILITIES_PATH

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
    def overrides(self) -> Mapping[str, Override]:
        return self._overrides

    @property
    def tool_hooks(self) -> Mapping[str, ToolHooks]:
        return self._tool_hooks

    def includes(self, name: str) -> bool:
        return name in self._selected_names

    def tool_override(self, name: str) -> Tool | None:
        replacement = self._overrides.get(name)
        if replacement is None:
            return None
        if not isinstance(replacement, Tool) or isinstance(
            replacement, (HDS, DeepSkill, ExitTool)
        ):
            raise TypeError(f"Override for '{name}' is not a resolved Tool")
        return replacement

    def tool_hooks_for(self, name: str) -> ToolHooks | None:
        return self._tool_hooks.get(name)

    def customizes_tool(self, name: str) -> bool:
        return name in self._overrides or name in self._tool_hooks

    def with_overrides(
        self, resolved: Mapping[str, HDS | DeepSkill | ExitTool | Tool]
    ) -> DefaultCapabilities:
        """Return the same selection with a new set of resource overrides."""
        return DefaultCapabilities(
            mocks=self._mocks,
            harness=self._harness,
            include=self._include,
            exclude=self._exclude,
            overrides=resolved,
            tool_hooks=self._tool_hooks,
        )

    @classmethod
    def _validate_tool_hooks(
        cls, tool_hooks: Mapping[str, ToolHooks] | None
    ) -> dict[str, ToolHooks]:
        if tool_hooks is None:
            return {}
        if not isinstance(tool_hooks, Mapping):
            raise TypeError("tool_hooks must be a mapping")
        cls._validate_default_names(tool_hooks, "tool_hooks")
        hooks: dict[str, ToolHooks] = {}
        for name, configuration in tool_hooks.items():
            if _RESOURCE_TYPES[name] not in (Tool, ExitTool):
                raise TypeError(f"Tool hooks for '{name}' require a tool or exit tool")
            if not isinstance(configuration, ToolHooks):
                raise TypeError(f"Tool hooks for '{name}' must be a ToolHooks instance")
            hooks[name] = configuration
        return hooks

    @staticmethod
    def _normalize_names(names: Iterable[str], label: str) -> frozenset[str]:
        if isinstance(names, (str, bytes)) or not isinstance(names, Iterable):
            raise TypeError(f"{label} must be an iterable of capability names")
        values = tuple(names)
        if not all(isinstance(name, str) for name in values):
            raise TypeError(f"{label} must contain only strings")
        return frozenset(values)

    @classmethod
    def _validate_default_names(
        cls, names: Iterable[str], label: str
    ) -> frozenset[str]:
        values = cls._normalize_names(names, label)
        unknown = values - MOCK_NAMES - HARNESS_NAMES
        if unknown:
            raise ValueError(
                f"Unknown default capabilities: {', '.join(sorted(unknown))}"
            )
        return values

    @classmethod
    def _validate_overrides(cls, overrides: OverrideInput) -> dict[str, Override]:
        if overrides is None:
            return {}
        entries = cls._override_entries(overrides)
        cls._normalize_names((name for name, _ in entries), "overrides")
        replacements: dict[str, Override] = {}
        for name, resource in entries:
            if name in replacements:
                raise ValueError(f"Duplicate override for '{name}'")
            if isinstance(resource, _NamedOverride):
                cls._validate_named_override(name, resource)
            else:
                cls._validate_object_override(name, resource)
            replacements[name] = resource
        return replacements

    @staticmethod
    def _validate_named_override(name: str, resource: _NamedOverride) -> None:
        if resource.target_name != name:
            raise ValueError(f"Override for '{name}' must target '{name}'")
        expected = _RESOURCE_TYPES.get(name)
        resource_type = Tool if isinstance(resource, ToolOverride) else DeepSkill
        if expected is not None and expected is not resource_type:
            raise TypeError(f"Override for '{name}' must match {expected.__name__}")

    @staticmethod
    def _validate_object_override(
        name: str, resource: HDS | DeepSkill | ExitTool | Tool
    ) -> None:
        expected = _RESOURCE_TYPES.get(
            name, DeepSkill if isinstance(resource, DeepSkill) else Tool
        )
        if not isinstance(resource, expected) or (
            expected is Tool and isinstance(resource, (HDS, DeepSkill, ExitTool))
        ):
            raise TypeError(f"Override for '{name}' must be a {expected.__name__}")
        if resource.get_name() != name:
            raise ValueError(f"Override for '{name}' must have name '{name}'")

    @staticmethod
    def _override_entries(
        overrides: Mapping[str, Override] | Iterable[NamedOverride] | NamedOverride,
    ) -> tuple[tuple[str, Override], ...]:
        if isinstance(overrides, _NamedOverride):
            return ((overrides.target_name, overrides),)
        if isinstance(overrides, Mapping):
            return tuple(cast(Mapping[str, Override], overrides).items())
        if isinstance(overrides, Iterable) and not isinstance(overrides, (str, bytes)):
            entries: list[tuple[str, Override]] = []
            for item in overrides:
                if not isinstance(item, _NamedOverride):
                    raise TypeError("overrides must contain only override descriptors")
                entries.append((item.target_name, item))
            return tuple(entries)
        raise TypeError("overrides must be a mapping or iterable of overrides")


BUILTIN_CAPABILITIES_PATH = DefaultCapabilities.CAPABILITIES_PATH
