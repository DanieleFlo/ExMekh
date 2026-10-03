from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StorageSettings:
    """User-facing location for tokenizer and session caches."""

    tokenizer_cache_directory: Path = Path(".exmekh_lib/tokenizers")

    @classmethod
    def from_environment(cls) -> StorageSettings:
        default = cls()
        path = os.getenv("EXMEKH_LIB_TOKENIZER_CACHE_DIR")
        return cls(Path(path) if path else default.tokenizer_cache_directory)


DEFAULT_STORAGE_SETTINGS = StorageSettings()
