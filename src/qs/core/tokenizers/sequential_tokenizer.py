from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from threading import Lock

from qs.core.tokenizers.tokenizer_session_registry import TokenizerSessionRegistry
from qs.core.tokenizers.vocabulary_cache import VocabularyCache
from qs.interfaces.tokenizer import Tokenizer


class SequentialTokenizer(Tokenizer):
    """Reusable filtering, caching and process-local sequential ID generation."""

    _initialization_lock = Lock()

    def __init__(
        self,
        identity: str,
        cache_directory: Path | None = None,
    ) -> None:
        if not isinstance(identity, str):
            raise TypeError("Tokenizer identity must be a string")
        if not identity.strip():
            raise ValueError("Tokenizer identity must not be empty")
        if cache_directory is not None and not isinstance(cache_directory, Path):
            raise TypeError("Cache directory must be a Path or None")
        self._identity = identity
        namespace = sha256(identity.encode("utf-8")).hexdigest()
        root = (cache_directory or Path(".qs/tokenizers")).resolve()
        self._directory = root / namespace
        self._cache = VocabularyCache(self._directory, identity)
        self._tokens: tuple[str, ...] | None = None
        self._vocabulary_fingerprint: str | None = None

    @abstractmethod
    def _load_vocabulary_tokens(self) -> Iterable[str]:  # pragma: no cover
        """Load raw vocabulary entries; do not strip or normalize them."""

    def initialize(self) -> None:
        """Load/build the vocabulary once; no network access on a cache hit."""
        with self._initialization_lock:
            if self._tokens is None:
                tokens = self._cache.load_or_build(self._load_vocabulary_tokens)
                self._vocabulary_fingerprint = sha256(
                    "\n".join(tokens).encode("utf-8")
                ).hexdigest()
                self._tokens = tokens

    def generate_id(self) -> str:
        if self._tokens is None:
            self.initialize()
        tokens = self._tokens
        fingerprint = self._vocabulary_fingerprint
        assert tokens is not None and fingerprint is not None
        session = TokenizerSessionRegistry.get_session(
            self._identity,
            fingerprint,
            self._directory / "temp",
        )
        size = len(tokens)
        index = session.reserve_next_index(size**3)
        prefix, third = divmod(index, size)
        first, second = divmod(prefix, size)
        return f"{tokens[first]}_{tokens[second]}_{tokens[third]}"
