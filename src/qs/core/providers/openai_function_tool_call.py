from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAIFunctionToolCall:
    name: str
    arguments: str

    def get_name(self) -> str:
        return self.name

    def get_arguments(self) -> str:
        return self.arguments
