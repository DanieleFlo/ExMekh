from dataclasses import dataclass

from exmekh_lib.config.model_text import MODEL_TEXT


@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        counts = (
            (
                self.prompt_tokens,
                MODEL_TEXT.modelusage_prompt_tokens_must_be_an_int_or_none,
            ),
            (
                self.completion_tokens,
                MODEL_TEXT.modelusage_completion_tokens_must_be_an_int_or_none,
            ),
            (
                self.total_tokens,
                MODEL_TEXT.modelusage_total_tokens_must_be_an_int_or_none,
            ),
        )
        for value, type_error in counts:
            if value is None:
                continue
            if type(value) is not int:
                raise TypeError(type_error)
            if value < 0:
                raise ValueError(MODEL_TEXT.modelusage_token_counts_must_be_nonnegative)

    def get_total_tokens(self) -> int | None:
        """Use the reported total, deriving it only when both components are known."""
        if self.total_tokens is not None:
            return self.total_tokens
        if self.prompt_tokens is not None and self.completion_tokens is not None:
            return self.prompt_tokens + self.completion_tokens
        return None
