from __future__ import annotations

from pathlib import Path
from typing import Any

from qs.core.cache.tool_call_record import CalledToolMetadata
from qs.core.cache.tool_call_store import ToolCallStore
from qs.core.models.assistant_message import ToolCall
from qs.core.tool_calls.directory import (
    TokenizerCacheIdentity,
    resolve_tool_call_directory,
)
from qs.core.tool_calls.tool_call_data import ToolCallData


class ToolCallSaver:
    """Saves tool calling metadata to .qs/tokenizers/<hash>/temp/<folder_name>/."""

    def __init__(
        self,
        directory: Path | None = None,
        *,
        tokenizer: TokenizerCacheIdentity | None = None,
        tokenizer_identity: str | None = None,
        cache_directory: Path | None = None,
        folder_name: str = "tool_calls",
    ) -> None:
        self._directory = resolve_tool_call_directory(
            directory,
            tokenizer=tokenizer,
            tokenizer_identity=tokenizer_identity,
            cache_directory=cache_directory,
            folder_name=folder_name,
        )
        self._store = ToolCallStore(self._directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._directory

    def save(self, data: ToolCallData) -> Path:
        """Atomically persist a ToolCallData record to disk."""
        return self._store.save(data)

    def record(
        self,
        name: str,
        id: str,
        args: dict[str, Any],
        result: Any,
        *,
        skill_path: str = "/",
        thinking: str | None = None,
    ) -> ToolCallData:
        """Group tool call information into a ToolCallData record and save it."""
        data = ToolCallData(
            name=name,
            id=id,
            args=args,
            result=result,
            skill_path=skill_path,
            thinking=thinking,
        )
        self.save(data)
        return data

    def record_call(
        self,
        tool_call: CalledToolMetadata | ToolCall,
        result: Any,
        id: str | None = None,
        *,
        skill_path: str = "/",
        thinking: str | None = None,
    ) -> ToolCallData:
        data = ToolCallData.from_call(
            tool_call, result, id, skill_path=skill_path, thinking=thinking
        )
        self.save(data)
        return data
