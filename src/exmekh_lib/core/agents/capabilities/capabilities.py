from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from exmekh_lib.core.agents.capabilities.catalog_attacher import SkillCatalogAttacher
from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.default_composer import (
    _DefaultCapabilitiesComposer,
)
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.capabilities.index import CapabilitiesIndex
from exmekh_lib.core.agents.capabilities.layout import CapabilitiesLayout
from exmekh_lib.core.agents.capabilities.override_resolver import _NamedOverrideResolver
from exmekh_lib.core.agents.capabilities.validator import CapabilitiesValidator
from exmekh_lib.core.agents.deep_skill.deep_skills import DeepSkills
from exmekh_lib.core.agents.exit_tool.exit_tools import ExitTools
from exmekh_lib.core.agents.hds.hds_collection import HDSCollection
from exmekh_lib.core.agents.tools.deep_tools import DeepTools


class Capabilities:
    CAPABILITIES_FOLDER_NAME: Final[str] = CapabilitiesLayout.CAPABILITIES_FOLDER_NAME

    def __init__(
        self,
        capabilities_paths: Sequence[Path] | None,
        *,
        default_capabilities: DefaultCapabilities | None = None,
    ) -> None:
        if default_capabilities is not None and not isinstance(
            default_capabilities, DefaultCapabilities
        ):
            raise TypeError("default_capabilities must be DefaultCapabilities or None")
        self._default_capabilities = default_capabilities
        if default_capabilities is not None:
            default_path = default_capabilities.capabilities_path
        elif capabilities_paths is None:
            default_path = DefaultCapabilities.CAPABILITIES_PATH
        else:
            default_path = None
        self._layout = CapabilitiesLayout(capabilities_paths, default_path=default_path)

    def load(self, *, work_root: Path) -> CapabilitiesData:
        data, _ = self.load_with_defaults(work_root=work_root)
        return data

    def load_with_defaults(
        self, *, work_root: Path
    ) -> tuple[CapabilitiesData, DefaultCapabilities | None]:
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
                defaults = _NamedOverrideResolver(defaults).resolve(data)
                data = _DefaultCapabilitiesComposer(
                    defaults, defaults.capabilities_path, tool_paths
                ).compose(data)
            index = CapabilitiesIndex.build(data)
            CapabilitiesValidator(index).validate((*data.hds, *data.deep_skills))
            SkillCatalogAttacher(index).attach_to_all(data.deep_skills)
        except Exception:
            exit_tools.unload()
            deep_tools.unload()
            raise

        return data, defaults

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
