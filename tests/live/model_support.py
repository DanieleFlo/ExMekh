"""Shared provider and tokenizer infrastructure for live model tests."""

import math
import os

from openai import OpenAI

from exmekh_lib.core.tokenizers.sequential_tokenizer import SequentialTokenizer
from exmekh_lib.providers.local import LocalProvider

DEFAULT_LIVE_REQUEST_TIMEOUT_SECONDS = 900.0


class LiveModelProvider(LocalProvider):
    def _build_client(self) -> OpenAI:
        value = os.getenv(
            "QS_LIVE_REQUEST_TIMEOUT_SECONDS",
            str(DEFAULT_LIVE_REQUEST_TIMEOUT_SECONDS),
        )
        message = (
            "QS_LIVE_REQUEST_TIMEOUT_SECONDS must be a finite number of seconds "
            "greater than zero"
        )
        try:
            timeout = float(value)
        except ValueError as error:
            raise ValueError(message) from error
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(message)
        return super()._build_client().with_options(timeout=timeout, max_retries=0)


class CacheTestTokenizer(SequentialTokenizer):
    """Exercise real session/ID storage without downloading a vocabulary."""

    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

    def _load_vocabulary_tokens(self) -> list[str]:
        return [f"token{index:02d}" for index in range(32)]
