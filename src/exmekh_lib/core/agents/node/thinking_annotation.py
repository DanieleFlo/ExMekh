"""Resolve provider thinking for direct deep-thinking tool headers."""

from __future__ import annotations

import logging

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.provider import Provider

logger = logging.getLogger(__name__)


class ThinkingAnnotationResolver:
    """Keep short traces and summarize longer ones once per node turn."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
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
                logger.exception("Deep-thinking token count unavailable")
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
            response = self._provider.call(
                ProviderRequest(
                    messages=[
                        SystemMessage(RUNTIME.thinking_summary_prompt),
                        UserMessage(thinking),
                    ],
                    reasoning_effort_override="none",
                )
            )
            summary = response.get_message()
            return (
                summary.strip()
                if isinstance(summary, str) and summary.strip()
                else None
            )
        except Exception:
            logger.exception("Deep-thinking summary unavailable")
            return None
