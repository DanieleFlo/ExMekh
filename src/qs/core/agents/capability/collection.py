from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from qs.core.agents.capability.markdown_capability import MarkdownCapability
from qs.core.agents.capability.parser import CapabilityMarkdownParser


class MarkdownCapabilityCollection[Resource: MarkdownCapability]:
    MARKDOWN_SUFFIX = ".md"
    resource_class: type[Resource]
    parser_factory: Callable[[], CapabilityMarkdownParser]
    resource_label = "Capability"
    duplicate_label = "capability"

    def __init__(self, folder_paths: Sequence[Path]) -> None:
        self._parser = self.parser_factory()
        self._folder_paths = self._validate_folder_paths(folder_paths)

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
            raise TypeError("folder_paths must be a sequence of Path")

        paths = tuple(folder_paths)

        if not all(isinstance(path, Path) for path in paths):
            raise TypeError("folder_paths must contain only Path instances")

        canonical_paths = tuple(path.resolve() for path in paths)

        if len(canonical_paths) != len(set(canonical_paths)):
            raise ValueError(f"Duplicate {self.duplicate_label} folder path")

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
                f"{self.resource_label} folder must not be a symbolic link: "
                f"{folder_path}"
            )

        if not folder_path.exists():
            raise FileNotFoundError(
                f"{self.resource_label} folder does not exist: {folder_path}"
            )

        if not folder_path.is_dir():
            raise ValueError(
                f"{self.resource_label} path must be a directory: {folder_path}"
            )

    def _validate_markdown_entry(self, entry: Path) -> None:
        if entry.is_symlink():
            raise ValueError(
                f"{self.resource_label} file must not be a symbolic link: {entry}"
            )

        if not entry.is_file():
            raise ValueError(
                f"{self.resource_label} folder can contain only Markdown files: {entry}"
            )

        if entry.suffix.lower() != self.MARKDOWN_SUFFIX:
            raise ValueError(
                f"{self.resource_label} file must have "
                f"'{self.MARKDOWN_SUFFIX}' extension: {entry}"
            )
