from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.models.history import HistoryMessage, MessageHistory


class CompactionScores(BaseModel):
    model_config = ConfigDict(extra="forbid")


def score_field(entry_id: str) -> str:
    return f"entry_{entry_id}"


def compaction_scores_model(entry_ids: Sequence[str]) -> type[CompactionScores]:
    """Make every history entry a required, independently constrained score field."""
    fields: dict[str, Any] = {
        score_field(entry_id): (
            int,
            Field(
                ge=1,
                le=100,
                strict=True,
                description=HISTORY_COMPACTION.score_description.format(
                    entry_id=entry_id
                ),
            ),
        )
        for entry_id in entry_ids
    }
    return create_model("CompactionScores", __base__=CompactionScores, **fields)


@dataclass(frozen=True, slots=True)
class CompactionResult:
    history: MessageHistory
    index_map: tuple[int, ...]
    output_count: int
    replaced_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def replacement_for(
        self, original: Sequence[HistoryMessage]
    ) -> list[HistoryMessage]:
        """Keep retained inputs by identity, so their owner can still reject them."""
        if len(original) != self.original_message_count:
            raise ValueError(HISTORY_COMPACTION.changed_history)
        replacement = self.history.get_history()
        for index, message in enumerate(original):
            if self.index_map[index + 1] == self.index_map[index] + 1:
                replacement[self.index_map[index]] = message
        return replacement

    @property
    def original_message_count(self) -> int:
        return len(self.index_map) - 1

    @property
    def compacted_message_count(self) -> int:
        return len(self.history.get_history())
