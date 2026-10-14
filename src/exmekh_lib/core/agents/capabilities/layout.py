from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Final

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.config.runtime import RUNTIME


class CapabilityType(StrEnum):
    DEEP_SKILL = CAPABILITY_FORMAT.deep_skill_folder
    TOOLS = CAPABILITY_FORMAT.tools_folder
    EXIT_TOOL = CAPABILITY_FORMAT.exit_tool_folder
    HDS = CAPABILITY_FORMAT.hds_folder


class CapabilitiesLayout:
    CAPABILITIES_FOLDER_NAME: Final[str] = CAPABILITY_FORMAT.root_folder

    def __init__(
        self,
        capabilities_paths: Sequence[Path] | None,
        *,
        default_path: Path | None = None,
    ) -> None:
        self._capabilities_paths = self._validate_capabilities_paths(
            capabilities_paths, default_path
        )

    @staticmethod
    def _validate_capabilities_paths(
        capabilities_paths: Sequence[Path] | None,
        default_path: Path | None,
    ) -> tuple[Path, ...]:
        if capabilities_paths is None:
            paths = ()
        elif isinstance(capabilities_paths, (str, bytes)) or not isinstance(
            capabilities_paths, Sequence
        ):
            raise TypeError(AGENT_TEXT.capabilities_paths_must_be_a_sequence_of_path)
        else:
            paths = tuple(capabilities_paths)

        if not all(isinstance(path, Path) for path in paths):
            raise TypeError(
                AGENT_TEXT.capabilities_paths_must_contain_only_path_instances
            )

        canonical_paths = tuple(path.resolve() for path in paths)

        if default_path is not None and default_path.resolve() not in canonical_paths:
            paths = (*paths, default_path)
            canonical_paths = (*canonical_paths, default_path.resolve())

        if not paths:
            raise ValueError(AGENT_TEXT.capabilities_paths_cannot_be_empty)

        if len(canonical_paths) != len(set(canonical_paths)):
            raise ValueError(AGENT_TEXT.duplicate_capabilities_path)

        return paths

    def prepare(self) -> None:
        for capabilities_path in self._capabilities_paths:
            self._validate_capabilities_folder(capabilities_path)

        for capabilities_path in self._capabilities_paths:
            self._create_missing_type_folders(capabilities_path)

    def deep_skill_folder_paths(self) -> tuple[Path, ...]:
        return self._type_folder_paths(CapabilityType.DEEP_SKILL)

    def tool_paths(self) -> tuple[Path, ...]:
        return tuple(
            path
            for path in self._child_paths(CapabilityType.TOOLS)
            if not self._is_bundled_runtime_tool(path)
        )

    @staticmethod
    def _is_bundled_runtime_tool(path: Path) -> bool:
        return (
            path.name in RUNTIME.runtime_tool_names
            and path.parent.parent.resolve() == CAPABILITY_FORMAT.bundled_root.resolve()
        )

    def exit_tool_paths(self) -> tuple[Path, ...]:
        return self._child_paths(CapabilityType.EXIT_TOOL)

    def hds_paths(self) -> tuple[Path, ...]:
        return self._child_paths(CapabilityType.HDS)

    def hds_folder_paths(self) -> tuple[Path, ...]:
        return self._type_folder_paths(CapabilityType.HDS)

    def _type_folder_paths(
        self,
        capability_type: CapabilityType,
    ) -> tuple[Path, ...]:
        return tuple(
            self._type_folder_path(capabilities_path, capability_type)
            for capabilities_path in self._capabilities_paths
        )

    def _child_paths(
        self,
        capability_type: CapabilityType,
    ) -> tuple[Path, ...]:
        children = (
            child
            for folder_path in self._type_folder_paths(capability_type)
            for child in folder_path.iterdir()
        )
        return tuple(sorted(children, key=str))

    def _validate_capabilities_folder(self, capabilities_path: Path) -> None:
        self._validate_capabilities_path(capabilities_path)

        for entry in capabilities_path.iterdir():
            self._validate_capabilities_entry(entry)

    def _validate_capabilities_path(self, capabilities_path: Path) -> None:
        if capabilities_path.is_symlink():
            raise ValueError(
                AGENT_TEXT.capabilities_path_must_not_be_a_symbolic_link.format(
                    capabilities_path=capabilities_path
                )
            )

        if not capabilities_path.exists():
            raise FileNotFoundError(
                AGENT_TEXT.capabilities_path_does_not_exist.format(
                    capabilities_path=capabilities_path
                )
            )

        if not capabilities_path.is_dir():
            raise ValueError(
                AGENT_TEXT.capabilities_path_must_be_a_directory.format(
                    capabilities_path=capabilities_path
                )
            )

        if capabilities_path.name != self.CAPABILITIES_FOLDER_NAME:
            raise ValueError(
                AGENT_TEXT.capabilities_folder_must_be_named.format(
                    capabilities_folder_name=self.CAPABILITIES_FOLDER_NAME,
                    capabilities_path=capabilities_path,
                )
            )

    def _validate_capabilities_entry(self, entry: Path) -> None:
        capability_type = self._capability_type(entry)

        if entry.is_symlink():
            raise ValueError(
                AGENT_TEXT.capabilities_entry_must_not_be_a_symbolic_link.format(
                    capability_type_value=capability_type.value, entry=entry
                )
            )

        if not entry.is_dir():
            raise ValueError(
                AGENT_TEXT.capabilities_entry_must_be_a_directory.format(
                    capability_type_value=capability_type.value, entry=entry
                )
            )

    def _capability_type(self, entry: Path) -> CapabilityType:
        if entry.name == "exit-skill":
            raise ValueError(
                AGENT_TEXT.the_exit_skill_directory_is_no_longer_supported_move
            )
        try:
            return CapabilityType(entry.name)
        except ValueError:
            allowed = ", ".join(
                capability_type.value for capability_type in CapabilityType
            )
            raise ValueError(
                AGENT_TEXT.unexpected_entry_inside_capabilities_folder_allowed_directories.format(
                    entry=entry, allowed=allowed
                )
            ) from None

    def _create_missing_type_folders(self, capabilities_path: Path) -> None:
        for capability_type in CapabilityType:
            self._type_folder_path(
                capabilities_path,
                capability_type,
            ).mkdir(exist_ok=True)

    @staticmethod
    def _type_folder_path(
        capabilities_path: Path,
        capability_type: CapabilityType,
    ) -> Path:
        return capabilities_path / capability_type
