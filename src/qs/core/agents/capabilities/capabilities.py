from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from qs.core.agents.capabilities.catalog_attacher import SkillCatalogAttacher
from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.capabilities.exit_tool_attacher import ExitToolAttacher
from qs.core.agents.capabilities.index import CapabilitiesIndex
from qs.core.agents.capabilities.layout import CapabilitiesLayout
from qs.core.agents.capabilities.validator import CapabilitiesValidator
from qs.core.agents.deep_skill.deep_skills import DeepSkills
from qs.core.agents.exit_skill.exit_skills import ExitSkills
from qs.core.agents.exit_tool.exit_tools import ExitTools
from qs.core.agents.hds.hds_collection import HDSCollection
from qs.core.agents.tools.deep_tools import DeepTools


class Capabilities:
    CAPABILITIES_FOLDER_NAME: Final[str] = CapabilitiesLayout.CAPABILITIES_FOLDER_NAME

    def __init__(
        self,
        capabilities_paths: Sequence[Path],
    ) -> None:
        self._layout = CapabilitiesLayout(capabilities_paths)

    def load(self) -> CapabilitiesData:
        self.format_capabilities_folders()
        deep_tools = DeepTools(self.get_tool_paths())
        exit_skills = ExitSkills(self.get_exit_skill_folder_paths())
        exit_tools = ExitTools(self.get_exit_tool_paths())

        try:
            data = self._load_data(deep_tools, exit_skills, exit_tools)
            index = CapabilitiesIndex.build(data)
            CapabilitiesValidator(index).validate((*data.hds, *data.deep_skills))
            SkillCatalogAttacher(index).attach_to_all(data.deep_skills)
            ExitToolAttacher(data.exit_tools).attach_to_all(data.exit_skills)
        except Exception:
            exit_tools.unload()
            deep_tools.unload()
            raise

        return data

    def _load_data(
        self,
        deep_tools: DeepTools,
        exit_skills: ExitSkills,
        exit_tools: ExitTools,
    ) -> CapabilitiesData:
        return CapabilitiesData(
            deep_skills=DeepSkills(self.get_deep_skill_folder_paths()).load(),
            tools=deep_tools.load(),
            hds=HDSCollection(self.get_hds_folder_paths()).load(),
            exit_skills=exit_skills.load(),
            exit_tools=exit_tools.load_exit_tools(),
        )

    def format_capabilities_folders(self) -> None:
        self._layout.prepare()

    def get_deep_skill_folder_paths(self) -> tuple[Path, ...]:
        return self._layout.deep_skill_folder_paths()

    def get_tool_paths(self) -> list[Path]:
        return list(self._layout.tool_paths())

    def get_exit_skill_folder_paths(self) -> tuple[Path, ...]:
        return self._layout.exit_skill_folder_paths()

    def get_exit_tool_paths(self) -> list[Path]:
        return list(self._layout.exit_tool_paths())

    def get_hds_paths(self) -> list[Path]:
        return list(self._layout.hds_paths())

    def get_hds_folder_paths(self) -> tuple[Path, ...]:
        return self._layout.hds_folder_paths()
