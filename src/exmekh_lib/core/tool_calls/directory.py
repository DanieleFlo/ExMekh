from __future__ import annotations

from pathlib import Path
from typing import Protocol

from exmekh_lib.config.runtime_text import RUNTIME_TEXT
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
        raise ValueError(RUNTIME_TEXT.folder_name_must_be_a_non_empty_string)
    if cache_component(folder_name) != folder_name:
        raise ValueError(RUNTIME_TEXT.folder_name_must_be_a_portable_directory_name)
    if directory is not None:
        if not isinstance(directory, Path):
            raise TypeError(RUNTIME_TEXT.directory_must_be_a_path_or_none)
        return directory.resolve()
    identity = tokenizer_identity
    if identity is None and tokenizer is not None:
        identity = tokenizer.cache_identity
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError(
            RUNTIME_TEXT.either_directory_tokenizer_identity_or_tokenizer_must_be_provided
        )
    return CacheManager.resolve_temp_directory(identity, cache_directory) / folder_name
