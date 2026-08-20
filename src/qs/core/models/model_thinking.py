from dataclasses import dataclass


@dataclass(frozen=True)
class ModelThinking:
    thoughts: str | None = None
