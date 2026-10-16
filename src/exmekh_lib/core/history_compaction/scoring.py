"""Rate exactly the output candidates supplied by a compaction plan."""

from pydantic import ValidationError

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.models.history import HistoryMessage
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.provider import Provider
from exmekh_lib.utils.xml import stringify_tool_result

from .models import compaction_scores_model, score_field
from .outputs import OutputCandidate


class CompactionScorer:
    def __init__(self, provider: Provider) -> None:
        self._provider = provider

    def score(
        self,
        messages: list[HistoryMessage],
        candidates: tuple[OutputCandidate, ...],
        tools: Tools | None,
    ) -> dict[str, int]:
        catalog = stringify_tool_result(
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
        request = StructuredProviderRequest(
            [*messages, UserMessage(HISTORY_COMPACTION.scoring_prompt + catalog)],
            response_model,
            tools=tools,
            tool_choice="none",
            reasoning_effort_override="none",
        )
        response = self._provider.call_structured(request)
        if response.get_tool_calls():
            raise ValueError(HISTORY_COMPACTION.unexpected_calls)
        try:
            result = response_model.model_validate(response.get_message().model_dump())
        except ValidationError as error:
            raise ValueError(HISTORY_COMPACTION.invalid_scores) from error
        fields = result.model_dump()
        return {
            item.entry_id: fields[score_field(item.entry_id)] for item in candidates
        }
