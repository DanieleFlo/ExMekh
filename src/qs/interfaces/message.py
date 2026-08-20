from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qs.core.models.user_message import (
        FilesUserMessage,
        ImagesUserMessage,
        TextsUserMessage,
    )


class Message(ABC):
    @abstractmethod
    def get_message(
        self,
    ) -> str | TextsUserMessage | ImagesUserMessage | FilesUserMessage | None:
        pass

    @abstractmethod
    def get_role_name(self) -> str:
        pass

    @abstractmethod
    def message_dump(self) -> dict:
        pass
