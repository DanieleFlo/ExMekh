from __future__ import annotations

import logging
from dataclasses import replace
from typing import Literal

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.models.text_summarizer import TextSummarizer
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.interfaces.provider import Provider

SummaryMode = Literal["none", "brief", "detailed"]

logger = logging.getLogger(__name__)


class ToolOutputSummarizer:
    """Summarize a read's body without changing its metadata or stored output."""

    def __init__(self, provider: Provider) -> None:
        self._provider = provider
        self._text_summarizer = TextSummarizer(provider)

    @staticmethod
    def validate_mode(mode: SummaryMode) -> None:
        if mode not in ("none", "brief", "detailed"):
            raise ValueError(RUNTIME.invalid_summary_mode)

    def summarize(
        self, result: PreformattedToolResult, mode: SummaryMode
    ) -> PreformattedToolResult:
        self.validate_mode(mode)
        if mode == "none":
            return result
        body = XmlHeader.body_from_content(result.content)
        if body is None:
            return result
        try:
            if (
                self._provider.get_tokenizer().count_tokens(body)
                < RUNTIME.tool_output_summary_min_tokens
            ):
                return result
            prompt = (
                RUNTIME.tool_output_brief_summary_prompt
                if mode == "brief"
                else RUNTIME.tool_output_detailed_summary_prompt
            )
            summary = self._text_summarizer.summarize(body, prompt)
        except Exception:
            logger.exception("Tool output summary unavailable; returning full output")
            return result
        content = XmlHeader.add_field_to_content(
            XmlHeader.replace_body_in_content(result.content, summary),
            "summary-mode",
            mode,
        )
        return replace(result, content=content)
