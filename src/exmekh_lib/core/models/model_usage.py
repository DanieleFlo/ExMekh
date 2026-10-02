from dataclasses import dataclass


@dataclass(frozen=True)
class ModelUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self):
        if self.prompt_tokens is not None and not isinstance(self.prompt_tokens, int):
            raise TypeError("ModelUsage prompt_tokens must be an int or None")
        if self.completion_tokens is not None and not isinstance(
            self.completion_tokens, int
        ):
            raise TypeError("ModelUsage completion_tokens must be an int or None")
        if self.total_tokens is not None and not isinstance(self.total_tokens, int):
            raise TypeError("ModelUsage total_tokens must be an int or None")
