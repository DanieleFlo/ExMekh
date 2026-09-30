from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict


class ExitToolResult(TypedDict):
    text: str
    items: list[str]


@dataclass(frozen=True, slots=True)
class ExitToolData:
    text: str
    items: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")

        text = self.text.strip()
        if not text:
            raise ValueError("text cannot be blank")

        if not isinstance(self.items, list):
            raise TypeError("items must be a list")
        if not all(isinstance(item, str) for item in self.items):
            raise TypeError("items must contain only strings")

        items = [item.strip() for item in self.items]
        if not all(items):
            raise ValueError("items cannot contain blank strings")

        object.__setattr__(self, "text", text)
        object.__setattr__(self, "items", items)

    def dump(self) -> ExitToolResult:
        return {"text": self.text, "items": self.items}
