from __future__ import annotations

from abc import ABC, abstractmethod


class Tokenizer(ABC):
    @abstractmethod
    def initialize(self) -> None:  # pragma: no cover
        """Prepare the tokenizer for ID generation; repeated calls are safe."""

    @abstractmethod
    def generate_id(self) -> str:  # pragma: no cover
        """Return a new three-component ID in the current session."""
