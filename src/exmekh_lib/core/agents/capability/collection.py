from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.agents.capability.markdown_capability import MarkdownCapability
from exmekh_lib.core.agents.capability.parser import CapabilityMarkdownParser


class MarkdownCapabilityCollection[Resource: MarkdownCapability[Any]](ABC):
    MARKDOWN_SUFFIX = CAPABILITY_FORMAT.markdown_suffix
    resource_label = "Capability"
    duplicate_label = "capability"

    def __init__(self, folder_paths: Sequence[Path]) -> None:
        self._parser = self.parser_factory()
        self._folder_paths = self._validate_folder_paths(folder_paths)

    @property
    @abstractmethod
    def resource_class(self) -> type[Resource]: ...

    @property
    @abstractmethod
    def parser_factory(self) -> Callable[[], CapabilityMarkdownParser[Any, Any]]: ...

    def load(self) -> tuple[Resource, ...]:
        return tuple(
            self.resource_class.from_markdown(markdown_path, self._parser)
            for markdown_path in self._get_markdown_paths()
        )

    def _validate_folder_paths(
        self,
        folder_paths: Sequence[Path],
    ) -> tuple[Path, ...]:
        if isinstance(folder_paths, (str, bytes)) or not isinstance(
            folder_paths,
            Sequence,
        ):
            raise TypeError(AGENT_TEXT.folder_paths_must_be_a_sequence_of_path)

        paths = tuple(folder_paths)

        if not all(isinstance(path, Path) for path in paths):
            raise TypeError(AGENT_TEXT.folder_paths_must_contain_only_path_instances)

        canonical_paths = tuple(path.resolve() for path in paths)

        if len(canonical_paths) != len(set(canonical_paths)):
            raise ValueError(
                AGENT_TEXT.duplicate_folder_path.format(
                    duplicate_label=self.duplicate_label
                )
            )

        return paths

    def _get_markdown_paths(self) -> tuple[Path, ...]:
        markdown_paths: list[Path] = []

        for folder_path in self._folder_paths:
            self._validate_folder(folder_path)
            markdown_paths.extend(self._get_folder_markdown_paths(folder_path))

        return tuple(markdown_paths)

    def _get_folder_markdown_paths(self, folder_path: Path) -> tuple[Path, ...]:
        entries = tuple(sorted(folder_path.iterdir(), key=lambda path: path.name))

        for entry in entries:
            self._validate_markdown_entry(entry)

        return entries

    def _validate_folder(self, folder_path: Path) -> None:
        if folder_path.is_symlink():
            raise ValueError(
                AGENT_TEXT.folder_must_not_be_a_symbolic_link.format(
                    resource_label=self.resource_label, folder_path=folder_path
                )
            )

        if not folder_path.exists():
            raise FileNotFoundError(
                AGENT_TEXT.folder_does_not_exist.format(
                    resource_label=self.resource_label, folder_path=folder_path
                )
            )

        if not folder_path.is_dir():
            raise ValueError(
                AGENT_TEXT.path_must_be_a_directory.format(
                    resource_label=self.resource_label, folder_path=folder_path
                )
            )

    def _validate_markdown_entry(self, entry: Path) -> None:
        if entry.is_symlink():
            raise ValueError(
                AGENT_TEXT.file_must_not_be_a_symbolic_link.format(
                    resource_label=self.resource_label, entry=entry
                )
            )

        if not entry.is_file():
            raise ValueError(
                AGENT_TEXT.folder_can_contain_only_markdown_files.format(
                    resource_label=self.resource_label, entry=entry
                )
            )

        if entry.suffix.lower() != self.MARKDOWN_SUFFIX:
            raise ValueError(
                AGENT_TEXT.file_must_have_extension.format(
                    resource_label=self.resource_label,
                    markdown_suffix=self.MARKDOWN_SUFFIX,
                    entry=entry,
                )
            )
