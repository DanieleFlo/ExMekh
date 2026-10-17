from dataclasses import replace

import pytest

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.history_compaction import HistoryCompactor, policy
from exmekh_lib.core.history_compaction.policy import normalize_scores
from exmekh_lib.core.history_compaction.progress import CompactionProgressReporter
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.history import MessageHistory
from exmekh_lib.core.models.request_budget import RequestBudget
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import UserMessage
from tests.core.history_compaction.support import (
    CompactionProvider,
    compacted_payload,
    count_request,
    output_batch,
)


@pytest.mark.parametrize(
    "scores", [{}, {"a": 80}, {"a": 50, "b": 60}, {"a": 1, "b": 9, "c": 60}]
)
def test_small_summary_sets_keep_their_original_scores(scores):
    assert normalize_scores(scores) == scores


@pytest.mark.parametrize("count,retained", [(20, 2), (21, 3), (100, 10)])
def test_normalization_lowers_every_score_by_the_same_offset(count, retained):
    original = {str(index): 100 - index for index in range(count)}
    normalized = normalize_scores(original)
    assert sum(score >= 10 for score in normalized.values()) == retained
    offsets = {original[key] - value for key, value in normalized.items() if value > 1}
    assert len(offsets) == 1
    assert original["0"] == 100


def test_cutoff_ties_are_all_retained_without_lowering_an_all_tied_set():
    original = {"first": 90, "second": 80, "tied": 80, "low": 79}
    assert normalize_scores(original) == {
        "first": 20,
        "second": 10,
        "tied": 10,
        "low": 9,
    }
    tied = {str(index): 60 for index in range(30)}
    assert normalize_scores(tied) == tied


def test_summary_share_is_owned_by_system_configuration(monkeypatch):
    monkeypatch.setattr(
        policy, "HISTORY_COMPACTION", replace(HISTORY_COMPACTION, summary_fraction=0.25)
    )
    assert (
        sum(
            score >= 10
            for score in normalize_scores({str(i): 100 - i for i in range(20)}).values()
        )
        == 5
    )


def test_elapsed_includes_scoring_but_eta_samples_only_real_summaries(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(
        "exmekh_lib.core.history_compaction.progress.time.monotonic", lambda: clock[0]
    )
    events = []
    reporter = CompactionProgressReporter(events.append)
    reporter.start(5)
    assert events[-1].remaining_seconds() is None
    clock[0] = 120
    reporter.plan(2)
    reporter.output_completed()
    assert events[-1].remaining_seconds() is None
    with reporter.summary():
        clock[0] = 126
    assert events[-1].elapsed_seconds() == 26
    assert events[-1].remaining_seconds() == 6
    reporter.output_completed()
    reporter.output_completed()
    assert events[-1].remaining_seconds() == 6
    with reporter.summary():
        clock[0] = 128
        assert events[-1].remaining_seconds() == 4
        assert events[-1].remaining_seconds(127) == 5
        assert events[-1].elapsed_seconds(127) == 27
        clock[0] = 136
        assert events[-1].remaining_seconds() == 0
    reporter.finish(failed=False)
    assert events[-1].finished
    assert events[-1].remaining_seconds() is None


def test_real_compaction_publishes_total_outputs_and_actual_summary_work():
    provider = CompactionProvider(score=30)
    history = MessageHistory(
        [
            SystemMessage("rules"),
            UserMessage("task"),
            *output_batch("first", call_id="a"),
            *output_batch("second", call_id="b"),
            AssistantMessage("pending decision"),
        ]
    )
    events = []

    def scoring_started(_):
        assert events[-1].phase == "scoring"
        assert events[-1].total_outputs == 2

    provider.on_score = scoring_started
    result = HistoryCompactor(provider, on_progress=events.append).compact(history)
    assert events[0].total_outputs == 0
    final = events[-1]
    assert final.phase == "completed"
    assert final.completed_outputs == final.total_outputs == result.output_count == 2
    assert (
        final.completed_summaries
        == final.total_summaries
        == len(provider.summary_requests)
        == 3
    )


@pytest.mark.parametrize("error", [RuntimeError("summary failed"), KeyboardInterrupt()])
def test_failed_or_interrupted_compaction_always_ends_progress(error):
    provider = CompactionProvider(score=30)
    provider.summary_error = error
    history = MessageHistory(
        [SystemMessage("rules"), UserMessage("task"), *output_batch("data")]
    )
    events = []
    with pytest.raises(type(error)):
        HistoryCompactor(provider, on_progress=events.append).compact(history)
    assert events[-1].phase == "failed"
    assert events[-1].active_summary_started_at is None


def test_optional_progress_observer_cannot_break_compaction():
    def broken(_):
        raise RuntimeError("disconnected observer")

    history = MessageHistory(
        [SystemMessage("rules"), UserMessage("task"), *output_batch("data")]
    )
    result = HistoryCompactor(CompactionProvider(score=9), on_progress=broken).compact(
        history
    )
    assert (
        compacted_payload(result.history.get_history()[-1])["outputs"][0][
            "representation"
        ]
        == "reference"
    )


def test_summary_quota_is_global_after_all_scoring_batches():
    provider = CompactionProvider()
    provider.score_transform = lambda fields: {
        field: 100 - int(field.rsplit("_", 1)[1]) // 2 for field in fields
    }
    history = MessageHistory([SystemMessage("rules"), UserMessage("task")])
    for index in range(24):
        history.add(
            output_batch(
                "small evidence", call_id=f"call-{index}", output_id=f"id-{index}"
            )
        )
    events = []
    result = HistoryCompactor(
        provider, budget=RequestBudget(13000, count_request), on_progress=events.append
    ).compact(history)
    assert len(provider.score_requests) > 1
    outputs = compacted_payload(result.history.get_history()[-1])["outputs"]
    assert sum(output["representation"] == "summary" for output in outputs) == 3
    assert len(provider.summary_requests) == events[-1].completed_summaries == 3
    assert events[-1].completed_outputs == 24
