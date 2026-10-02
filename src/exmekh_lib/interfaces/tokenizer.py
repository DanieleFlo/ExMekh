from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from exmekh_lib.core.cache.session_cache import SessionCache


class Tokenizer(ABC):
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
