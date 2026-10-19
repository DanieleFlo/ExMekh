from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.capability_prompts import CAPABILITY_PROMPTS
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.capabilities.contracts import apply_runtime_contract
from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.default_catalog import (
    DEFAULT_CAPABILITY_CATALOG,
)
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.capabilities.tool_composer import _DefaultToolComposer
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.models.tool import Tool


class _DefaultCapabilitiesComposer:
    """Compose selected bundled resources with application resources."""

    def __init__(
        self,
        defaults: DefaultCapabilities,
        builtin_root: Path,
        tool_paths: Sequence[Path],
    ) -> None:
        self._defaults = defaults
        self._tool_composer = _DefaultToolComposer(defaults)
        self._builtin_root = builtin_root.resolve()
        self._tool_paths = tuple(tool_paths)

    def compose(self, loaded: CapabilitiesData) -> CapabilitiesData:
        self._validate_override_targets(loaded)
        selected = self._select_loaded(loaded)
        composed = self._apply_overrides(selected, loaded)
        self._validate_selected_names(composed)
        self._validate_dependencies(composed)
        return self._apply_tool_hooks(composed)

    def _validate_dependencies(self, data: CapabilitiesData) -> None:
        names = {tool.get_name() for tool in data.get_all_tools()}
        for tool in data.tools:
            name = tool.get_name()
            if name in self._defaults.overrides:
                continue
            entry = DEFAULT_CAPABILITY_CATALOG.get(name)
            if entry is None:
                continue
            missing = set(entry.dependencies) - names
            if missing:
                raise ValueError(
                    CAPABILITY_PROMPTS.dependency_missing.format(
                        name=name, dependencies=", ".join(sorted(missing))
                    )
                )

    def _apply_tool_hooks(self, composed: CapabilitiesData) -> CapabilitiesData:
        return replace(
            composed,
            tools=tuple(
                self._tool_composer.apply_hooks(tool) for tool in composed.tools
            ),
            exit_tools=tuple(
                self._tool_composer.apply_hooks(tool) for tool in composed.exit_tools
            ),
        )

    def _validate_override_targets(self, loaded: CapabilitiesData) -> None:
        tool_names = {tool.get_name() for tool in loaded.tools}
        skill_names = {skill.get_name() for skill in loaded.deep_skills}
        missing_tools = {
            name
            for name, replacement in self._defaults.overrides.items()
            if name not in RUNTIME.runtime_tool_names
            and not isinstance(replacement, (HDS, DeepSkill, ExitTool))
            and name not in tool_names
        }
        if missing_tools:
            raise ValueError(
                AGENT_TEXT.override_tools_not_found_in_capabilities_paths
                + ", ".join(sorted(missing_tools))
            )
        missing_skills = {
            name
            for name, replacement in self._defaults.overrides.items()
            if isinstance(replacement, DeepSkill) and name not in skill_names
        }
        if missing_skills:
            raise ValueError(
                AGENT_TEXT.override_deep_skills_not_found_in_capabilities_paths
                + ", ".join(sorted(missing_skills))
            )

    def _select_loaded(self, loaded: CapabilitiesData) -> CapabilitiesData:
        return CapabilitiesData(
            deep_skills=tuple(
                skill
                for skill in loaded.deep_skills
                if self._should_keep_loaded_resource(
                    skill.get_name(), skill.markdown_path
                )
            ),
            tools=tuple(
                tool
                for path, tool in zip(self._tool_paths, loaded.tools, strict=True)
                if self._should_keep_loaded_resource(tool.get_name(), path)
            ),
            hds=tuple(
                hds
                for hds in loaded.hds
                if self._should_keep_loaded_resource(hds.get_name(), hds.markdown_path)
            ),
            exit_tools=tuple(
                tool
                for tool in loaded.exit_tools
                if self._should_keep_loaded_resource(tool.get_name(), tool.folder_path)
            ),
        )

    def _should_keep_loaded_resource(self, name: str, path: Path) -> bool:
        if name in self._defaults.overrides:
            return False
        return not self._is_bundled_resource(path) or self._defaults.includes(name)

    def _is_bundled_resource(self, path: Path) -> bool:
        return path.resolve().is_relative_to(self._builtin_root)

    def _original_tools(self, loaded: CapabilitiesData) -> dict[str, Tool]:
        originals = {tool.get_name(): tool for tool in loaded.tools}
        for path, tool in zip(self._tool_paths, loaded.tools, strict=True):
            # An application collision must not replace the bundled target's contract.
            if self._is_bundled_resource(path):
                originals[tool.get_name()] = tool
        return originals

    def _apply_overrides(
        self, selected: CapabilitiesData, loaded: CapabilitiesData
    ) -> CapabilitiesData:
        skills = list(selected.deep_skills)
        tools = list(selected.tools)
        hds = list(selected.hds)
        exit_tools = list(selected.exit_tools)
        original_tools = self._original_tools(loaded)

        for name, replacement in self._defaults.overrides.items():
            if name in RUNTIME.runtime_tool_names:
                continue
            if isinstance(replacement, DeepSkill):
                skills.append(replacement.copy())
            elif isinstance(replacement, HDS):
                hds.append(replacement.copy())
            elif isinstance(replacement, ExitTool):
                exit_tools.append(replacement.copy())
            elif isinstance(replacement, Tool):
                tools.append(
                    apply_runtime_contract(replacement).with_execution_policy_of(
                        original_tools[name]
                    )
                )
            else:
                raise TypeError(
                    AGENT_TEXT.override_for_must_be_resolved_before_composition.format(
                        name=name
                    )
                )

        return CapabilitiesData(
            tuple(skills), tuple(tools), tuple(hds), tuple(exit_tools)
        )

    def _validate_selected_names(self, data: CapabilitiesData) -> None:
        required_names = self._defaults.selected_names - RUNTIME.runtime_tool_names
        loaded_names = {
            resource.get_name()
            for resource in (
                *data.deep_skills,
                *data.tools,
                *data.exit_tools,
                *data.hds,
            )
        }
        missing = required_names - loaded_names
        if missing:
            raise ValueError(
                AGENT_TEXT.selected_default_capabilities_are_missing
                + ", ".join(sorted(missing))
            )
