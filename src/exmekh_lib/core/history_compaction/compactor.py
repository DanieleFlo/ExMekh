"""Prepare a validated replacement conversation; application belongs to its owner."""

from collections.abc import Callable, Sequence

from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.history import HistoryMessage, MessageHistory
from exmekh_lib.core.models.request_budget import RequestBudget
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.interfaces.provider import Provider

from .context import CompactionContext
from .models import CompactionResult
from .outputs import CompactionOutputs
from .policy import normalize_scores
from .presentation import CompactionPresenter
from .progress import CompactionProgress, CompactionProgressReporter
from .projection import compaction_spans, complete_batches
from .scoring import CompactionScorer


class HistoryCompactor:
    def __init__(
        self,
        provider: Provider,
        *,
        output_lookup: Callable[[str], ToolCallRecord | None] | None = None,
        files: OutputFileWorkspace | None = None,
        budget: RequestBudget | None = None,
        on_progress: Callable[[CompactionProgress], None] | None = None,
    ) -> None:
        self._scorer = CompactionScorer(provider, budget)
        self._presenter = CompactionPresenter(provider, budget)
        self._tokenizer = provider.get_tokenizer()
        self._files = files
        self._on_progress = on_progress
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
        progress = CompactionProgressReporter(self._on_progress)
        succeeded = False
        try:
            progress.start(0)
            result = self._compact(history, tools, protected_indices, progress)
            succeeded = True
            return result
        finally:
            progress.finish(failed=not succeeded)

    def _compact(
        self,
        history: MessageHistory,
        tools: Tools | None,
        protected_indices: Sequence[int],
        progress: CompactionProgressReporter,
    ) -> CompactionResult:
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
        context = CompactionContext(messages, candidates, self._tokenizer, self._files)
        progress.start(len(context.entries()))
        scores = (
            normalize_scores(self._scorer.score(context, tools)) if candidates else {}
        )
        messages = context.messages
        progress.plan(
            sum(
                self._presenter.summary_count(
                    messages[span.start : span.end],
                    context.entries(range(span.start, span.end)),
                    scores,
                )
                for span in spans
            )
        )
        replacements = {}
        for span in spans:
            selected_outputs = context.entries(range(span.start, span.end))
            summary = self._presenter.create_summary(
                messages[span.start : span.end], selected_outputs, scores, progress
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
        replaced_history = {
            pair[0].get_tool_calls()[0].id: [
                message.message_dump() for message in original[span.start : span.end]
            ]
            for span, pair in replacements.values()
        }
        return CompactionResult(
            result, tuple(index_map), len(context.entries()), replaced_history
        )
