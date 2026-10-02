from __future__ import annotations

from pathlib import Path
from typing import Protocol

from exmekh_lib.core.cache.cache_manager import CacheManager
from exmekh_lib.core.cache.paths import cache_component


class TokenizerCacheIdentity(Protocol):
    @property
    def cache_identity(self) -> str: ...


def resolve_tool_call_directory(
    directory: Path | None,
    *,
    tokenizer: TokenizerCacheIdentity | None,
    tokenizer_identity: str | None,
    cache_directory: Path | None,
    folder_name: str,
) -> Path:
    """Resolve standalone archives; live node caches use their session directory."""
    if not isinstance(folder_name, str) or not folder_name.strip():
        raise ValueError("folder_name must be a non-empty string")
    if cache_component(folder_name) != folder_name:
        raise ValueError("folder_name must be a portable directory name")
    if directory is not None:
        if not isinstance(directory, Path):
            raise TypeError("directory must be a Path or None")
        return directory.resolve()
    identity = tokenizer_identity
    if identity is None and tokenizer is not None:
        identity = tokenizer.cache_identity
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError(
            "Either 'directory', 'tokenizer_identity', or 'tokenizer' must be provided"
        )
    return CacheManager.resolve_temp_directory(identity, cache_directory) / folder_name
