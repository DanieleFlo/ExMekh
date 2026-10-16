from __future__ import annotations

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.request_budget import (
    ContextBudgetExceededError,
    RequestBudget,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.provider import Provider


class TextSummarizer:
    """Request plain text using an isolated client without reasoning or tools."""

    def __init__(
        self, provider: Provider, *, budget: RequestBudget | None = None
    ) -> None:
        self._provider = provider
        self._budget = budget

    def summarize(self, text: str, prompt: str) -> str:
        if self._budget is not None:
            tokenizer = self._provider.get_tokenizer()
            while not self._budget.fits(self._request(text, prompt)):
                parts = self._budget.split_text(
                    text, tokenizer, lambda part: self._request(part, prompt)
                )
                reduced = "\n\n".join(
                    self._summarize_once(part, prompt) for part in parts
                )
                if tokenizer.count_tokens(reduced) >= tokenizer.count_tokens(text):
                    raise ContextBudgetExceededError(
                        HISTORY_COMPACTION.summary_no_progress
                    )
                text = reduced
        return self._summarize_once(text, prompt)

    @staticmethod
    def _request(text: str, prompt: str) -> ProviderRequest:
        return ProviderRequest(
            [SystemMessage(prompt), UserMessage(text)],
            tool_choice="none",
            reasoning_effort_override="none",
        )

    def _summarize_once(self, text: str, prompt: str) -> str:
        if self._budget is not None:
            self._budget.require_fits(self._request(text, prompt))
        client = Client(self._provider, SystemMessage(prompt))
        response = client.call(
            ClientRequest(
                UserMessage(text), tool_choice="none", reasoning_effort_override="none"
            )
        )
        if response.get_tool_calls():
            raise ValueError(
                MODEL_TEXT.the_summary_response_must_not_contain_tool_calls
            )
        summary = response.get_message()
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(MODEL_TEXT.the_summary_response_contains_no_text)
        return summary.strip()
