import os
from pathlib import Path
from threading import Lock
from typing import ClassVar

from qs.core.tokenizers.tokenizer_session import TokenizerSession


class TokenizerSessionRegistry:
    """Share sessions by process, tokenizer identity and vocabulary version."""

    _lock = Lock()
    _sessions: ClassVar[dict[tuple[int, str, str], TokenizerSession]] = {}

    @classmethod
    def get_session(
        cls,
        tokenizer_identity: str,
        vocabulary_fingerprint: str,
        directory: Path,
    ) -> TokenizerSession:
        key = (os.getpid(), tokenizer_identity, vocabulary_fingerprint)
        with cls._lock:
            if key not in cls._sessions:
                cls._sessions[key] = TokenizerSession(directory)
            return cls._sessions[key]
