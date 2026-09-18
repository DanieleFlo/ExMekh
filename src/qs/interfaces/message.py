from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy


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

    def copy(self) -> Message[MessageContent]:
        return deepcopy(self)
