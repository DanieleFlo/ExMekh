from __future__ import annotations

from logging import getLogger
from pathlib import Path

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.core.agents.capability.config_references import render_config_references
from exmekh_lib.utils.file import read_file_from_path

logger = getLogger(__name__)


class MarkdownLoader:
    def __init__(self, markdown_path: Path) -> None:
        if not isinstance(markdown_path, Path):
            raise TypeError(AGENT_TEXT.markdown_path_must_be_a_path)

        self._markdown_path = markdown_path

    def load(self) -> str:
        self._validate_path()

        try:
            return render_config_references(read_file_from_path(self._markdown_path))
        except Exception:
            logger.exception(AGENT_TEXT.failed_to_load_markdown, self._markdown_path)
            raise

    def _validate_path(self) -> None:
        if self._markdown_path.is_symlink():
            raise ValueError(
                AGENT_TEXT.markdown_path_must_not_be_a_symbolic_link.format(
                    markdown_path=self._markdown_path
                )
            )

        if not self._markdown_path.exists():
            raise FileNotFoundError(
                AGENT_TEXT.markdown_file_does_not_exist.format(
                    markdown_path=self._markdown_path
                )
            )

        if not self._markdown_path.is_file():
            raise ValueError(
                AGENT_TEXT.markdown_path_must_be_a_file.format(
                    markdown_path=self._markdown_path
                )
            )

        if self._markdown_path.suffix.lower() != ".md":
            raise ValueError(
                AGENT_TEXT.capability_file_must_be_a_markdown_file.format(
                    markdown_path=self._markdown_path
                )
            )
