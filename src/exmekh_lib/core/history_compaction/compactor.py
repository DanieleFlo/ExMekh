"""Prepare a validated replacement conversation; application belongs to its owner."""

from collections.abc import Callable, Sequence

from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.history import HistoryMessage, MessageHistory
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.interfaces.provider import Provider

from .models import CompactionResult
from .outputs import CompactionOutputs
from .presentation import CompactionPresenter
from .projection import compaction_spans, complete_batches
from .scoring import CompactionScorer


class HistoryCompactor:
    def __init__(
        self,
        provider: Provider,
        *,
        output_lookup: Callable[[str], ToolCallRecord | None] | None = None,
        files: OutputFileWorkspace | None = None,
    ) -> None:
        self._scorer = CompactionScorer(provider)
        self._presenter = CompactionPresenter(provider)
        self._outputs = CompactionOutputs(
            provider.get_tokenizer(), output_lookup, files
        )

    def compact(
        self,
        history: MessageHistory,
        *,
        tools: Tools | None = None,
        protected_indices: Sequence[int] = (),
    ) -> CompactionResult:
        """Return a new history; never persist summaries or alter the input."""
        history.check_history()
        original = history.get_history()
        messages = history.get_copy_of_history()
        spans = compaction_spans(messages, protected_indices)
        candidates = {
            index: self._outputs.candidates(index, message)
            for span in spans
            for index in range(span.start, span.end)
            if isinstance(message := messages[index], ToolMessage)
        }
        entries = tuple(item for group in candidates.values() for item in group)
        scores = self._scorer.score(messages, entries, tools) if candidates else {}
        replacements = {}
        for span in spans:
            selected_outputs = tuple(
                item
                for index in range(span.start, span.end)
                for item in candidates.get(index, ())
            )
            summary = self._presenter.create_summary(
                messages[span.start : span.end], selected_outputs, scores
            )
            replacements[span.start] = (span, summary)
        compacted: list[HistoryMessage] = []
        index_map = [0] * (len(messages) + 1)
        index = 0
        while index < len(messages):
            index_map[index] = len(compacted)
            replacement = replacements.get(index)
            if replacement is None:
                # Preserve original objects, including user attachments and metadata.
                compacted.append(original[index])
                index += 1
                continue
            span, pair = replacement
            for removed in range(span.start, span.end):
                index_map[removed] = len(compacted)
            compacted.extend(pair)
            index = span.end
        index_map[-1] = len(compacted)
        result = MessageHistory(compacted)
        result.check_history()
        complete_batches(compacted)
        return CompactionResult(result, tuple(index_map), len(entries))
