from dataclasses import dataclass

from exmekh_lib.config.model_text import MODEL_TEXT


@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self):
        if self.prompt_tokens is not None and not isinstance(self.prompt_tokens, int):
            raise TypeError(MODEL_TEXT.modelusage_prompt_tokens_must_be_an_int_or_none)
        if self.completion_tokens is not None and not isinstance(
            self.completion_tokens, int
        ):
            raise TypeError(
                MODEL_TEXT.modelusage_completion_tokens_must_be_an_int_or_none
            )
        if self.total_tokens is not None and not isinstance(self.total_tokens, int):
            raise TypeError(MODEL_TEXT.modelusage_total_tokens_must_be_an_int_or_none)
