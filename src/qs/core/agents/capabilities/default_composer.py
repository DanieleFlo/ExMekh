from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.capabilities.defaults import RUNTIME_NAMES, DefaultCapabilities
from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.exit_tool.exit_tool import ExitTool
from qs.core.agents.hds.hds import HDS
from qs.core.models.tool import Tool


class _DefaultCapabilitiesComposer:
    """Compose selected bundled resources with application resources."""

    def __init__(
        self,
        defaults: DefaultCapabilities,
        builtin_root: Path,
        tool_paths: Sequence[Path],
    ) -> None:
        self._defaults = defaults
        self._builtin_root = builtin_root.resolve()
        self._tool_paths = tuple(tool_paths)

    def compose(self, loaded: CapabilitiesData) -> CapabilitiesData:
        self._validate_override_targets(loaded)
        selected = self._select_loaded(loaded)
        composed = self._apply_overrides(selected)
        self._validate_selected_names(composed)
        return composed

    def _validate_override_targets(self, loaded: CapabilitiesData) -> None:
        tool_names = {tool.name for tool in loaded.tools}
        skill_names = {skill.name for skill in loaded.deep_skills}
        missing_tools = {
            name
            for name, replacement in self._defaults.overrides.items()
            if name not in RUNTIME_NAMES
            and not isinstance(replacement, (HDS, DeepSkill, ExitTool))
            and name not in tool_names
        }
        if missing_tools:
            raise ValueError(
                "Override tools not found in capabilities_paths: "
                + ", ".join(sorted(missing_tools))
            )
        missing_skills = {
            name
            for name, replacement in self._defaults.overrides.items()
            if isinstance(replacement, DeepSkill) and name not in skill_names
        }
        if missing_skills:
            raise ValueError(
                "Override deep skills not found in capabilities_paths: "
                + ", ".join(sorted(missing_skills))
            )

    def _select_loaded(self, loaded: CapabilitiesData) -> CapabilitiesData:
        return CapabilitiesData(
            deep_skills=tuple(
                skill
                for skill in loaded.deep_skills
                if self._should_keep_loaded_resource(skill.name, skill.markdown_path)
            ),
            tools=tuple(
                tool
                for path, tool in zip(self._tool_paths, loaded.tools, strict=True)
                if self._should_keep_loaded_resource(tool.name, path)
            ),
            hds=tuple(
                hds
                for hds in loaded.hds
                if self._should_keep_loaded_resource(hds.name, hds.markdown_path)
            ),
            exit_tools=tuple(
                tool
                for tool in loaded.exit_tools
                if self._should_keep_loaded_resource(tool.name, tool.folder_path)
            ),
        )

    def _should_keep_loaded_resource(self, name: str, path: Path) -> bool:
        if name in self._defaults.overrides:
            return False
        return not path.resolve().is_relative_to(
            self._builtin_root
        ) or self._defaults.includes(name)

    def _apply_overrides(self, selected: CapabilitiesData) -> CapabilitiesData:
        skills = list(selected.deep_skills)
        tools = list(selected.tools)
        hds = list(selected.hds)
        exit_tools = list(selected.exit_tools)

        for name, replacement in self._defaults.overrides.items():
            if name in RUNTIME_NAMES:
                continue
            if isinstance(replacement, DeepSkill):
                skills.append(self._copy_skill(replacement))
            elif isinstance(replacement, HDS):
                hds.append(self._copy_hds(replacement))
            elif isinstance(replacement, ExitTool):
                exit_tools.append(replacement)
            elif isinstance(replacement, Tool):
                tools.append(replacement)
            else:
                raise TypeError(
                    f"Override for '{name}' must be resolved before composition"
                )

        return CapabilitiesData(
            tuple(skills), tuple(tools), tuple(hds), tuple(exit_tools)
        )

    def _validate_selected_names(self, data: CapabilitiesData) -> None:
        required_names = self._defaults.selected_names - RUNTIME_NAMES
        loaded_names = {
            resource.name
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
                "Selected default capabilities are missing: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _copy_skill(skill: DeepSkill) -> DeepSkill:
        return DeepSkill(
            name=skill.name,
            markdown_path=skill.markdown_path,
            metadata=skill.metadata,
            body=skill.body,
            references=skill.references,
        )

    @staticmethod
    def _copy_hds(hds: HDS) -> HDS:
        return HDS(
            name=hds.name,
            markdown_path=hds.markdown_path,
            metadata=hds.metadata,
            body=hds.body,
            references=hds.references,
        )
