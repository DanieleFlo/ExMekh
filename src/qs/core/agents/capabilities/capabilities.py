from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from qs.core.agents.capabilities.catalog_attacher import SkillCatalogAttacher
from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.capabilities.default_composer import _DefaultCapabilitiesComposer
from qs.core.agents.capabilities.defaults import (
    BUILTIN_CAPABILITIES_PATH,
    DefaultCapabilities,
)
from qs.core.agents.capabilities.index import CapabilitiesIndex
from qs.core.agents.capabilities.layout import CapabilitiesLayout
from qs.core.agents.capabilities.validator import CapabilitiesValidator
from qs.core.agents.deep_skill.deep_skills import DeepSkills
from qs.core.agents.exit_tool.exit_tools import ExitTools
from qs.core.agents.hds.hds_collection import HDSCollection
from qs.core.agents.tools.deep_tools import DeepTools


class Capabilities:
    CAPABILITIES_FOLDER_NAME: Final[str] = CapabilitiesLayout.CAPABILITIES_FOLDER_NAME

    def __init__(
        self,
        capabilities_paths: Sequence[Path],
        *,
        default_capabilities: DefaultCapabilities | None = None,
    ) -> None:
        if default_capabilities is not None and not isinstance(
            default_capabilities, DefaultCapabilities
        ):
            raise TypeError("default_capabilities must be DefaultCapabilities or None")
        self._default_capabilities = default_capabilities
        if default_capabilities is None:
            self._layout = CapabilitiesLayout(capabilities_paths)
            return

        if isinstance(capabilities_paths, (str, bytes)) or not isinstance(
            capabilities_paths, Sequence
        ):
            raise TypeError("capabilities_paths must be a sequence of Path")
        paths = tuple(capabilities_paths)
        if not all(isinstance(path, Path) for path in paths):
            raise TypeError("capabilities_paths must contain only Path instances")
        if BUILTIN_CAPABILITIES_PATH.resolve() not in {
            path.resolve() for path in paths
        }:
            paths = (*paths, BUILTIN_CAPABILITIES_PATH)
        self._layout = CapabilitiesLayout(paths)

    def load(self, *, work_root: Path) -> CapabilitiesData:
        self.format_capabilities_folders()
        tool_paths = self.get_tool_paths()
        deep_tools = DeepTools(
            tool_paths,
            work_root=work_root,
        )
        exit_tools = ExitTools(self.get_exit_tool_paths())

        try:
            data = self._load_data(deep_tools, exit_tools)
            defaults = self._default_capabilities
            if defaults is not None:
                data = _DefaultCapabilitiesComposer(
                    defaults, BUILTIN_CAPABILITIES_PATH, tool_paths
                ).compose(data)
            index = CapabilitiesIndex.build(data)
            CapabilitiesValidator(index).validate((*data.hds, *data.deep_skills))
            SkillCatalogAttacher(index).attach_to_all(data.deep_skills)
        except Exception:
            exit_tools.unload()
            deep_tools.unload()
            raise

        return data

    def _load_data(
        self,
        deep_tools: DeepTools,
        exit_tools: ExitTools,
    ) -> CapabilitiesData:
        return CapabilitiesData(
            deep_skills=DeepSkills(self.get_deep_skill_folder_paths()).load(),
            tools=deep_tools.load(),
            hds=HDSCollection(self.get_hds_folder_paths()).load(),
            exit_tools=exit_tools.load_exit_tools(),
        )

    def format_capabilities_folders(self) -> None:
        self._layout.prepare()

    def get_deep_skill_folder_paths(self) -> tuple[Path, ...]:
        return self._layout.deep_skill_folder_paths()

    def get_tool_paths(self) -> list[Path]:
        return list(self._layout.tool_paths())

    def get_exit_tool_paths(self) -> list[Path]:
        return list(self._layout.exit_tool_paths())

    def get_hds_paths(self) -> list[Path]:
        return list(self._layout.hds_paths())

    def get_hds_folder_paths(self) -> tuple[Path, ...]:
        return self._layout.hds_folder_paths()
