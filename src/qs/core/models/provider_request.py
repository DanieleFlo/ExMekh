from dataclasses import dataclass

from qs.core.models.history import MessageHistory
from qs.interfaces.message import Message


@dataclass(frozen=True)
class ProviderRequest:
    messages: Message | list[Message]
    history: MessageHistory

    def get_request(self) -> MessageHistory:
        if not self.messages:
            raise ValueError("Messages cannot be empty")

        history = MessageHistory(self.history.get_history())

        if isinstance(self.messages, Message):
            new_history = history.add(self.messages)
            return new_history

        if isinstance(self.messages, list):
            if not all(isinstance(message, Message) for message in self.messages):
                raise TypeError("All items in messages must be instances of Message")
            new_history = history.add(self.messages)
            return new_history

        raise TypeError(
            "Messages must be an instance of Message or a list of Message instances"
        )
