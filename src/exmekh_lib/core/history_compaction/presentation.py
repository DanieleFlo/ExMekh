"""Summarize selected context and render synthetic compaction exchanges."""

from uuid import uuid4

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.history import HistoryMessage
from exmekh_lib.core.models.text_summarizer import TextSummarizer
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage, UserMessagePart
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.interfaces.provider import Provider
from exmekh_lib.utils.xml import stringify_tool_result

from .outputs import OutputCandidate
from .policy import summary_prompt


class CompactionPresenter:
    def __init__(self, provider: Provider) -> None:
        self._summarizer = TextSummarizer(provider)

    def create_summary(
        self,
        messages: list[HistoryMessage],
        candidates: tuple[OutputCandidate, ...],
        scores: dict[str, int],
    ) -> list[HistoryMessage]:
        outputs = [
            self._summarize_output(candidate, scores[candidate.entry_id])
            for candidate in candidates
        ]
        assistant_text = [
            content
            for message in messages
            if not isinstance(message, ToolMessage)
            and (content := message.message_dump().get("content"))
        ]
        state = (
            self._summarizer.summarize(
                stringify_tool_result(assistant_text),
                HISTORY_COMPACTION.assistant_prompt,
            )
            if assistant_text
            else None
        )
        call_id = uuid4().hex
        name = HISTORY_COMPACTION.tool_name
        body = stringify_tool_result({"assistant": state, "outputs": outputs})
        content = XmlHeader(
            source="Tool",
            name=name,
            id=call_id,
            fields={"response-type": "final", "compacted": "true"},
        ).prepend_to(body)
        parts: list[UserMessagePart] = []
        for message in messages:
            if (
                isinstance(message, ToolMessage)
                and (attachment := message.get_attachments()) is not None
            ):
                original = attachment.get_message()
                parts.extend(original if isinstance(original, list) else [original])
        return [
            AssistantMessage(
                None, tool_calls=[ToolCall(call_id, FunctionCall(name, {}))]
            ),
            ToolMessage(
                content,
                call_id,
                name,
                attachments=UserMessage(parts) if parts else None,
            ),
        ]

    def _summarize_output(
        self, candidate: OutputCandidate, score: int
    ) -> dict[str, object]:
        prompt = (
            summary_prompt(score, candidate.token_count) if candidate.available else ""
        )
        result: dict[str, object] = {
            "name": candidate.name,
            "id": candidate.output_id,
            "tool-call-id": candidate.message.get_tool_call_id(),
            "score": score,
            "recoverable": candidate.recoverable,
        }
        if candidate.used_tools:
            result["used-tools"] = [
                reference.to_dict() for reference in candidate.used_tools
            ]
        if candidate.external is not None:
            result["files"] = candidate.external.to_dict()
        if prompt is None:
            result["representation"] = "original"
            # A file manifest stays a reference, never a large inline copy.
            result["content"] = (
                candidate.external.render()
                if candidate.external is not None
                else candidate.body
            )
        elif prompt:
            result["representation"] = "summary"
            result["content"] = self._summarizer.summarize(candidate.body, prompt)
        else:
            result["representation"] = "reference"
        for field in ("type", "status", "response-type"):
            value = XmlHeader.field_from_content(candidate.message.get_message(), field)
            if value is not None:
                result[field] = value
        return result
