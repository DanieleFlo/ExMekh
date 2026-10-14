"""Resolve provider thinking for direct deep-thinking tool headers."""

from __future__ import annotations

import logging

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.runtime_text import RUNTIME_TEXT
from exmekh_lib.core.models.text_summarizer import TextSummarizer
from exmekh_lib.interfaces.provider import Provider

logger = logging.getLogger(__name__)


class ThinkingAnnotationResolver:
    """Keep short traces and summarize longer ones once per node turn."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self._text_summarizer = TextSummarizer(provider)
        self._annotations: dict[str, str | None] = {}

    def begin_turn(self) -> None:
        self._annotations.clear()

    def resolve(self, thinking: str | None) -> str | None:
        if not thinking or not thinking.strip():
            return None
        if thinking not in self._annotations:
            try:
                token_count = self._provider.get_tokenizer().count_tokens(thinking)
            except Exception:
                logger.exception(RUNTIME_TEXT.deep_thinking_token_count_unavailable)
                self._annotations[thinking] = None
            else:
                self._annotations[thinking] = (
                    thinking
                    if token_count <= RUNTIME.deep_thinking_token_limit
                    else self._summarize(thinking)
                )
        return self._annotations[thinking]

    def _summarize(self, thinking: str) -> str | None:
        try:
            return self._text_summarizer.summarize(
                thinking, RUNTIME.thinking_summary_prompt
            )
        except Exception:
            logger.exception(RUNTIME_TEXT.deep_thinking_summary_unavailable)
            return None
