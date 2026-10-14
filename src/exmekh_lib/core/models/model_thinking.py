from dataclasses import dataclass

from exmekh_lib.config.model_text import MODEL_TEXT


@dataclass(frozen=True)
class ModelThinking:
    thoughts: str | None = None

    def __post_init__(self):
        if self.thoughts is not None and not isinstance(self.thoughts, str):
            raise TypeError(MODEL_TEXT.modelthinking_thoughts_must_be_a_string_or_none)
