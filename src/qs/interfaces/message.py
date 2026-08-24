from __future__ import annotations

from abc import ABC, abstractmethod


class Message[MessageContent](ABC):
    @abstractmethod
    def get_message(self) -> MessageContent:
        pass  # pragma: no cover

    @abstractmethod
    def get_role_name(self) -> str:
        pass  # pragma: no cover

    @abstractmethod
    def message_dump(self) -> dict:
        pass  # pragma: no cover
