from __future__ import annotations

from logging import getLogger
from pathlib import Path

from exmekh_lib.utils.file import read_file_from_path

logger = getLogger(__name__)


class MarkdownLoader:
    def __init__(self, markdown_path: Path) -> None:
        if not isinstance(markdown_path, Path):
            raise TypeError("markdown_path must be a Path")

        self._markdown_path = markdown_path

    def load(self) -> str:
        self._validate_path()

        try:
            return read_file_from_path(self._markdown_path)
        except Exception:
            logger.exception("Failed to load Markdown: %s", self._markdown_path)
            raise

    def _validate_path(self) -> None:
        if self._markdown_path.is_symlink():
            raise ValueError(
                f"Markdown path must not be a symbolic link: {self._markdown_path}"
            )

        if not self._markdown_path.exists():
            raise FileNotFoundError(
                f"Markdown file does not exist: {self._markdown_path}"
            )

        if not self._markdown_path.is_file():
            raise ValueError(f"Markdown path must be a file: {self._markdown_path}")

        if self._markdown_path.suffix.lower() != ".md":
            raise ValueError(
                f"Capability file must be a Markdown file: {self._markdown_path}"
            )
