from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolOutputReference:
    """An immutable reference to one tool's canonical output."""

    name: str
    id: str

    def __post_init__(self) -> None:
        for field_name, value in (("name", self.name), ("id", self.id)):
            if not isinstance(value, str):
                raise TypeError(f"Tool output reference {field_name} must be a string")
            if not value.strip():
                raise ValueError(f"Tool output reference {field_name} cannot be empty")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "id": self.id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolOutputReference:
        if not isinstance(data, dict):
            raise TypeError("Tool output reference must be a dictionary")
        if set(data) != {"name", "id"}:
            raise ValueError("Tool output reference must contain only name and id")
        return cls(name=data["name"], id=data["id"])
