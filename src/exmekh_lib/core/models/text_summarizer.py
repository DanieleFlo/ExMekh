from __future__ import annotations

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.provider import Provider


class TextSummarizer:
    """Request plain text using an isolated client without reasoning or tools."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def summarize(self, text: str, prompt: str) -> str:
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
