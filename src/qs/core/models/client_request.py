from dataclasses import dataclass

from qs.interfaces.message import Message


@dataclass(frozen=True)
class ClientRequest:
    message: Message
    extra: dict | None = None
    tool_choice: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, Message):
            raise TypeError("Expected a Message instance")
        if self.extra is not None and not isinstance(self.extra, dict):
            raise TypeError("Extra provider data must be a dictionary")
        if self.tool_choice is not None and not isinstance(self.tool_choice, str):
            raise TypeError("Tool choice must be a string")
        

        if self.extra is not None:
            object.__setattr__(self, "extra", self.extra.copy())
