from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any


class Message[MessageContent](ABC):
    @abstractmethod
    def get_message(self) -> MessageContent:
        pass  # pragma: no cover

    @abstractmethod
    def get_role_name(self) -> str:
        pass  # pragma: no cover

    @abstractmethod
    def message_dump(self) -> dict[str, str]:
        pass  # pragma: no cover

    def set_metadata(self, name: str, value: Any) -> None:
        """Store local metadata independently of the caller and shallow copies."""
        metadata = dict(getattr(self, "_metadata", {}))
        metadata[name] = deepcopy(value)
        self._metadata = metadata

    def get_metadata(self) -> dict[str, Any]:
        """Return an independent snapshot; metadata is not sent to the provider."""
        return deepcopy(getattr(self, "_metadata", {}))

    def copy(self) -> Message[MessageContent]:
        return deepcopy(self)
