from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from exmekh_lib.config.tokenizer_text import TOKENIZER_TEXT

if TYPE_CHECKING:
    from exmekh_lib.core.cache.session_cache import SessionCache


class Tokenizer(ABC):
    def split_text(self, text: str, max_tokens: int) -> tuple[str, ...]:
        """Split without losing characters; tokenizers can optimize this fallback."""
        self._validate_token_limit(max_tokens, "max_tokens")
        if max_tokens == 0:
            raise ValueError(TOKENIZER_TEXT.max_tokens_must_be_positive)
        parts: list[str] = []
        remaining = text
        while self.count_tokens(remaining) > max_tokens:
            prefix = self.truncate_text(remaining, max_tokens)
            if (
                not prefix
                or not remaining.startswith(prefix)
                or self.count_tokens(prefix) > max_tokens
            ):
                raise ValueError(TOKENIZER_TEXT.lossless_split_required)
            parts.append(prefix)
            remaining = remaining[len(prefix) :]
        if remaining:
            parts.append(remaining)
        return tuple(parts)

    @staticmethod
    def _validate_token_limit(count: int, name: str) -> None:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(TOKENIZER_TEXT.must_be_an_integer.format(name=name))
        if count < 0:
            raise ValueError(TOKENIZER_TEXT.cannot_be_negative.format(name=name))

    @abstractmethod
    def count_tokens(self, text: str) -> int:  # pragma: no cover
        """Count model tokens in text, without automatically added special tokens."""

    @abstractmethod
    def truncate_text(
        self, text: str, start_tokens: int, end_tokens: int = 0
    ) -> str:  # pragma: no cover
        """Keep up to the requested initial and final tokens of text."""

    @abstractmethod
    def initialize(self) -> None:  # pragma: no cover
        """Prepare the tokenizer for ID generation; repeated calls are safe."""

    @abstractmethod
    def generate_id(self) -> str:  # pragma: no cover
        """Return a new three-component ID in the current session."""

    @abstractmethod
    def get_session(self) -> SessionCache:  # pragma: no cover
        """Return the active runtime session cache."""
