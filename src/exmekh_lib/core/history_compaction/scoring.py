"""Rate exactly the output candidates supplied by a compaction plan."""

from pydantic import ValidationError

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.models.history import HistoryMessage
from exmekh_lib.core.models.request_budget import RequestBudget
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.provider import Provider
from exmekh_lib.utils.json import serialize_json

from .context import CompactionContext
from .models import CompactionScores, compaction_scores_model, score_field
from .outputs import OutputCandidate


class CompactionScorer:
    def __init__(self, provider: Provider, budget: RequestBudget | None = None) -> None:
        self._provider = provider
        self._budget = budget

    def score(
        self,
        context: CompactionContext,
        tools: Tools | None,
    ) -> dict[str, int]:
        pending = [tuple(item.entry_id for item in context.entries())]
        scores: dict[str, int] = {}
        while pending:
            entry_ids = pending.pop()
            candidates = tuple(
                item for item in context.entries() if item.entry_id in entry_ids
            )
            request = self._request(context.messages, candidates, tools)
            if self._budget is not None and not self._budget.fits(request):
                if context.externalize_next():
                    pending.append(entry_ids)
                    continue
                if len(entry_ids) > 1:
                    middle = len(entry_ids) // 2
                    pending.extend((entry_ids[middle:], entry_ids[:middle]))
                    continue
                self._budget.require_fits(request)
            scores.update(self._call(request, candidates))
        return scores

    def _request(
        self,
        messages: list[HistoryMessage],
        candidates: tuple[OutputCandidate, ...],
        tools: Tools | None,
    ) -> StructuredProviderRequest[CompactionScores]:
        # This is temporary model input, not a tool result. Compact JSON avoids
        # repeating XML type tags for every field when the history is nearly full.
        catalog = serialize_json(
            {
                "expected_count": len(candidates),
                "expected_entry_ids": [item.entry_id for item in candidates],
                "outputs": [
                    {
                        **item.preview(self._provider.get_tokenizer()),
                        "score_field": score_field(item.entry_id),
                    }
                    for item in candidates
                ],
            }
        )
        response_model = compaction_scores_model([item.entry_id for item in candidates])
        return StructuredProviderRequest(
            [*messages, UserMessage(HISTORY_COMPACTION.scoring_prompt + catalog)],
            response_model,
            tools=tools,
            tool_choice="none",
            reasoning_effort_override="none",
        )

    def _call(
        self,
        request: StructuredProviderRequest[CompactionScores],
        candidates: tuple[OutputCandidate, ...],
    ) -> dict[str, int]:
        response = self._provider.call_structured(request)
        if response.get_tool_calls():
            raise ValueError(HISTORY_COMPACTION.unexpected_calls)
        try:
            result = request.get_response_model().model_validate(
                response.get_message().model_dump()
            )
        except ValidationError as error:
            raise ValueError(HISTORY_COMPACTION.invalid_scores) from error
        fields = result.model_dump()
        return {
            item.entry_id: fields[score_field(item.entry_id)] for item in candidates
        }
