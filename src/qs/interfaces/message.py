from __future__ import annotations

from abc import ABC, abstractmethod


class Message[MessageContent](ABC):
    @abstractmethod
    def get_message(self) -> MessageContent:
        pass

    @abstractmethod
    def get_role_name(self) -> str:
        pass

    @abstractmethod
    def message_dump(self) -> dict:
        pass
