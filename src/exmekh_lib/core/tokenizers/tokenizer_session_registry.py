from __future__ import annotations

from pathlib import Path

from exmekh_lib.core.cache.cache_manager import CacheManager
from exmekh_lib.core.cache.session_cache import SessionCache


class TokenizerSessionRegistry:
    """Share sessions by process, tokenizer identity and vocabulary version.

    Delegates to CacheManager for centralized session lifecycle and storage.
    """

    @classmethod
    def get_session(
        cls,
        tokenizer_identity: str,
        vocabulary_fingerprint: str,
        directory: Path,
    ) -> SessionCache:
        return CacheManager.get_session(
            tokenizer_identity,
            vocabulary_fingerprint,
            directory,
        )

    @classmethod
    def close_all(cls) -> None:
        """Close all active tokenizer sessions."""
        CacheManager.close_all()

    @classmethod
    def reset(cls) -> None:
        """Reset all active tokenizer sessions."""
        CacheManager.reset()
