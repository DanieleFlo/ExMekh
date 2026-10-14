from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from exmekh_lib.config.agent_text import AGENT_TEXT


class ExitToolResult(TypedDict):
    text: str
    items: list[str]


@dataclass(frozen=True, slots=True)
class ExitToolData:
    text: str
    items: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError(AGENT_TEXT.text_must_be_a_string)

        text = self.text.strip()
        if not text:
            raise ValueError(AGENT_TEXT.text_cannot_be_blank)

        if not isinstance(self.items, list):
            raise TypeError(AGENT_TEXT.items_must_be_a_list)
        if not all(isinstance(item, str) for item in self.items):
            raise TypeError(AGENT_TEXT.items_must_contain_only_strings)

        items = [item.strip() for item in self.items]
        if not all(items):
            raise ValueError(AGENT_TEXT.items_cannot_contain_blank_strings)

        object.__setattr__(self, "text", text)
        object.__setattr__(self, "items", items)

    def dump(self) -> ExitToolResult:
        return {"text": self.text, "items": self.items}
