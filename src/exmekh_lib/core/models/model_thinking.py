from dataclasses import dataclass


@dataclass(frozen=True)
class ModelThinking:
    thoughts: str | None = None

    def __post_init__(self):
        if self.thoughts is not None and not isinstance(self.thoughts, str):
            raise TypeError("ModelThinking thoughts must be a string or None")
