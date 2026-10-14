from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exmekh_lib.config.model_text import MODEL_TEXT


@dataclass(frozen=True, slots=True)
class ToolOutputReference:
    """An immutable reference to one tool's canonical output."""

    name: str
    id: str

    def __post_init__(self) -> None:
        for field_name, value in (("name", self.name), ("id", self.id)):
            if not isinstance(value, str):
                raise TypeError(
                    MODEL_TEXT.tool_output_reference_must_be_a_string.format(
                        field_name=field_name
                    )
                )
            if not value.strip():
                raise ValueError(
                    MODEL_TEXT.tool_output_reference_cannot_be_empty.format(
                        field_name=field_name
                    )
                )

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "id": self.id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolOutputReference:
        if not isinstance(data, dict):
            raise TypeError(MODEL_TEXT.tool_output_reference_must_be_a_dictionary)
        if set(data) != {"name", "id"}:
            raise ValueError(
                MODEL_TEXT.tool_output_reference_must_contain_only_name_and_id
            )
        return cls(name=data["name"], id=data["id"])
