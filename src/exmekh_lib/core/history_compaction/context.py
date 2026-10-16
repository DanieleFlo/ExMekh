"""Shorten temporary scoring context while retaining the complete source outputs."""

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.models.history import HistoryMessage
from exmekh_lib.core.models.request_budget import ContextBudgetExceededError
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.tool_calls.output_preparer import ToolOutputPreparer
from exmekh_lib.interfaces.tokenizer import Tokenizer

from .outputs import OutputCandidate


class CompactionContext:
    def __init__(
        self,
        messages: list[HistoryMessage],
        candidates: dict[int, tuple[OutputCandidate, ...]],
        tokenizer: Tokenizer,
        files: OutputFileWorkspace | None,
    ) -> None:
        self._messages = list(messages)
        self._candidates = dict(candidates)
        self._preparer = ToolOutputPreparer(tokenizer, files) if files else None
        self._pending = sorted(
            candidates, key=lambda index: candidates[index][0].token_count
        )

    @property
    def messages(self) -> list[HistoryMessage]:
        return list(self._messages)

    def entries(self, indices: range | None = None) -> tuple[OutputCandidate, ...]:
        return tuple(
            item
            for index, group in self._candidates.items()
            if indices is None or index in indices
            for item in group
        )

    def externalize_next(self) -> bool:
        """Replace the largest eligible body only if its file reference is smaller."""
        if self._preparer is None:
            return False
        while self._pending:
            index = self._pending.pop()
            candidate, *descendants = self._candidates[index]
            if candidate.external is not None:
                continue
            message = candidate.message
            original = PreformattedToolResult(
                message.get_message(), attachments=message.get_attachments()
            )
            prepared = self._preparer.shorten(
                ToolOutputPresentation(
                    ToolOutputReference(candidate.name, candidate.output_id), original
                )
            )
            if prepared.failure is not None:
                raise ContextBudgetExceededError(HISTORY_COMPACTION.export_failure)
            if not prepared.changed:
                continue
            shortened = candidate.with_presentation(prepared.as_result(original))
            self._messages[index] = shortened.message
            self._candidates[index] = (shortened, *descendants)
            return True
        return False
