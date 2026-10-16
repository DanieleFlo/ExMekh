"""Resolve and authorize output files through the public node-scoped cache API."""

from collections.abc import Callable
from pathlib import Path

from exmekh_lib.config.cache_text import CACHE_TEXT
from exmekh_lib.config.tool_output import ToolOutputSettings
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.output_file_store import OutputFileSession
from exmekh_lib.core.models.external_tool_output import ExternalToolOutput


class OutputFileWorkspace:
    def __init__(
        self,
        cache_provider: Callable[[], NodeCache],
        work_root: Path,
        settings: ToolOutputSettings | None = None,
    ) -> None:
        self._cache_provider = cache_provider
        self._work_root = work_root.resolve()
        self.settings = settings or ToolOutputSettings.from_environment()
        self.root = self._work_root / self.settings.directory
        if self.root.resolve() != self.root:
            raise ValueError(
                CACHE_TEXT.output_root_must_not_redirect_to_another_location
            )
        OutputFileSession.clean_abandoned(self.root)

    def write(self, key: str, parts: tuple[str, ...]) -> tuple[str, ...]:
        paths = self._cache_provider().write_output_files(self.root, key, parts)
        return tuple(path.relative_to(self._work_root).as_posix() for path in paths)

    def read_text(self, reference: ExternalToolOutput) -> str:
        """Read registered, unchanged output parts in their original order."""
        cache = self._cache_provider()
        return "".join(
            cache.read_output_file(self.root, self._work_root / item.path)
            for item in reference.files
        )

    def read_limit(self, path: Path, default: int) -> int:
        if not path.is_relative_to(self.root):
            return default
        return self._cache_provider().output_file_read_limit(self.root, path, default)
