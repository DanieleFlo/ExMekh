from __future__ import annotations

from pathlib import Path

from qs.config.cache import CACHE_FORMAT
from qs.core.cache.tool_call_store import ToolCallStore
from qs.core.tool_calls.directory import (
    TokenizerCacheIdentity,
    resolve_tool_call_directory,
)
from qs.core.tool_calls.tool_call_data import ToolCallData


class ToolCallReader:
    """Reads tool calling metadata from .qs/tokenizers/<hash>/temp/<folder_name>/."""

    def __init__(
        self,
        directory: Path | None = None,
        *,
        tokenizer: TokenizerCacheIdentity | None = None,
        tokenizer_identity: str | None = None,
        cache_directory: Path | None = None,
        folder_name: str = CACHE_FORMAT.tool_calls_folder,
    ) -> None:
        self._directory = resolve_tool_call_directory(
            directory,
            tokenizer=tokenizer,
            tokenizer_identity=tokenizer_identity,
            cache_directory=cache_directory,
            folder_name=folder_name,
        )
        self._store = ToolCallStore(self._directory)

    @property
    def directory(self) -> Path:
        return self._directory

    def exists(self, call_id: str) -> bool:
        return self.read(call_id) is not None

    def read(self, call_id: str) -> ToolCallData | None:
        return self._store.get(call_id)

    def get(self, call_id: str) -> ToolCallData:
        """Retrieve a ToolCallData record by ID, raising FileNotFoundError if absent."""
        record = self.read(call_id)
        if record is None:
            raise FileNotFoundError(
                f"No tool call recorded for ID '{call_id}' at {self._directory}"
            )
        return record

    def list_ids(self) -> list[str]:
        return self._store.list_ids()

    def read_all(self) -> list[ToolCallData]:
        return self._store.get_all()
