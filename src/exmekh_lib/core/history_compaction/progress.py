"""Transient compaction observations; estimate work from actual summaries only."""

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Literal
from uuid import uuid4

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CompactionProgress:
    operation_id: str
    started_at: float
    phase: Literal["scoring", "summarizing", "completed", "failed"] = "scoring"
    total_outputs: int = 0
    completed_outputs: int = 0
    total_summaries: int = 0
    completed_summaries: int = 0
    summary_seconds: float = 0
    active_summary_started_at: float | None = None

    @property
    def finished(self) -> bool:
        return self.phase in ("completed", "failed")

    def elapsed_seconds(self, now: float | None = None) -> float:
        return max(0, (time.monotonic() if now is None else now) - self.started_at)

    def remaining_seconds(self, now: float | None = None) -> float | None:
        if self.phase != "summarizing" or not self.completed_summaries:
            return None
        remaining = self.total_summaries - self.completed_summaries
        estimate = self.summary_seconds / self.completed_summaries * remaining
        if self.active_summary_started_at is not None:
            estimate -= max(
                0,
                (time.monotonic() if now is None else now)
                - self.active_summary_started_at,
            )
        return max(0, estimate)


class CompactionProgressReporter:
    def __init__(self, callback: Callable[[CompactionProgress], None] | None) -> None:
        self._callback = callback
        self._state = CompactionProgress(uuid4().hex, time.monotonic())

    def _publish(self, **changes) -> None:
        self._state = replace(self._state, **changes)
        if self._callback is not None:
            try:
                self._callback(self._state)
            except Exception:
                logger.exception(HISTORY_COMPACTION.progress_observer_failure)

    def start(self, total_outputs: int) -> None:
        self._publish(total_outputs=total_outputs)

    def plan(self, total_summaries: int) -> None:
        self._publish(phase="summarizing", total_summaries=total_summaries)

    @contextmanager
    def summary(self) -> Iterator[None]:
        started = time.monotonic()
        self._publish(active_summary_started_at=started)
        yield
        self._publish(
            active_summary_started_at=None,
            completed_summaries=self._state.completed_summaries + 1,
            summary_seconds=self._state.summary_seconds + time.monotonic() - started,
        )

    def output_completed(self) -> None:
        self._publish(completed_outputs=self._state.completed_outputs + 1)

    def finish(self, *, failed: bool) -> None:
        self._publish(
            phase="failed" if failed else "completed", active_summary_started_at=None
        )
