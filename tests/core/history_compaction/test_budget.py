from unittest.mock import patch

import pytest

from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.cache.session_cache import SessionCache
from exmekh_lib.core.history_compaction import HistoryCompactor
from exmekh_lib.core.models.history import MessageHistory
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.request_budget import (
    ContextBudgetExceededError,
    RequestBudget,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.text_summarizer import TextSummarizer
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.tool_calls.output_preparer import ToolOutputPreparer
from tests.core.history_compaction.support import (
    CompactionProvider,
    compacted_payload,
    count_request,
    output_batch,
)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_capacity_is_rejected(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        RequestBudget(capacity, count_request)


def test_capacity_reserves_generation_space_and_rejects_an_oversized_prompt():
    budget = RequestBudget(1000, count_request)
    assert budget.prompt_limit == 875
    assert budget.fits(ProviderRequest([UserMessage("small")]))
    with pytest.raises(ContextBudgetExceededError, match="protected inputs"):
        budget.require_fits(ProviderRequest([UserMessage("x" * 1000)]))


def test_output_below_static_limit_is_exported_before_scoring_and_preserved(tmp_path):
    provider = CompactionProvider(score=9)
    body = "evidence line\n" * 300
    history = MessageHistory(
        [
            SystemMessage("rules"),
            UserMessage("user constraint " * 220),
            *output_batch(body),
        ]
    )
    original = history.dump_history()
    budget = RequestBudget(8000, count_request)
    provider.on_score = budget.require_fits
    with SessionCache(tmp_path / "cache") as session:
        cache = session.node("node")
        files = OutputFileWorkspace(lambda: cache, tmp_path)
        result = HistoryCompactor(provider, files=files, budget=budget).compact(history)
        assert len(provider.score_requests) == 1
        scoring_history = provider.score_requests[0].get_messages()[:-1]
        assert not any(
            str(item.get_message()).endswith(body) for item in scoring_history
        )
        output = compacted_payload(result.history.get_history()[-1])["outputs"][0]
        assert output["recoverable"]
        reference = output["files"]["files"][0]["path"]
        assert (tmp_path / reference).read_text() == body
        assert cache.get_all() == []
        assert next(iter(result.replaced_history.values()))[-1]["content"].endswith(
            body
        )
    assert history.dump_history() == original


def test_large_score_catalog_is_split_without_losing_or_duplicating_entries():
    provider = CompactionProvider(score=9)
    history = MessageHistory([SystemMessage("rules"), UserMessage("task")])
    for index in range(24):
        history.add(
            output_batch(
                "small evidence", call_id=f"call-{index}", output_id=f"id-{index}"
            )
        )
    budget = RequestBudget(11000, count_request)
    provider.on_score = budget.require_fits
    result = HistoryCompactor(provider, budget=budget).compact(history)
    assert len(provider.score_requests) > 1
    fields = [
        field
        for request in provider.score_requests
        for field in request.get_response_model().model_fields
    ]
    assert len(fields) == len(set(fields)) == result.output_count == 24


def test_protected_input_cannot_be_discarded_to_fit_scoring():
    provider = CompactionProvider(score=9)
    history = MessageHistory(
        [SystemMessage("rules"), UserMessage("x" * 3000), *output_batch("body")]
    )
    before = history.dump_history()
    with pytest.raises(ContextBudgetExceededError):
        HistoryCompactor(provider, budget=RequestBudget(2000, count_request)).compact(
            history
        )
    assert not provider.score_requests
    assert history.dump_history() == before


def test_export_failure_does_not_replace_output_or_call_the_provider(tmp_path):
    provider = CompactionProvider(score=9)
    history = MessageHistory(
        [
            SystemMessage("rules"),
            UserMessage("x" * 2000),
            *output_batch("evidence\n" * 1000),
        ]
    )
    before = history.dump_history()
    with SessionCache(tmp_path / "cache") as session:
        files = OutputFileWorkspace(lambda: session.node("node"), tmp_path)
        with patch.object(files, "write", side_effect=OSError("disk full")):
            with pytest.raises(ContextBudgetExceededError, match="externalize"):
                HistoryCompactor(
                    provider, files=files, budget=RequestBudget(5000, count_request)
                ).compact(history)
    assert not provider.score_requests
    assert history.dump_history() == before


def test_summary_of_oversized_output_uses_bounded_chunks_and_keeps_essential_result():
    provider = CompactionProvider(summary="Essential result retained.")
    budget = RequestBudget(2000, count_request)
    summarizer = TextSummarizer(provider, budget=budget)
    text = "source evidence and pending actions\n" * 200
    assert summarizer.summarize(text, "Summarize evidence") == provider.summary
    assert len(provider.summary_requests) > 2
    for request in provider.summary_requests:
        budget.require_fits(request)
    chunks = [
        request.get_messages()[-1].get_message()
        for request in provider.summary_requests[:-1]
    ]
    assert "".join(chunks) == text


def test_summary_failure_to_reduce_stops_instead_of_looping():
    provider = CompactionProvider(summary="x" * 3000)
    with pytest.raises(ContextBudgetExceededError, match="did not reduce"):
        TextSummarizer(provider, budget=RequestBudget(1000, count_request)).summarize(
            "y" * 2000, "Summarize"
        )


def test_oversized_summary_prompt_is_rejected_before_inference():
    provider = CompactionProvider()
    with pytest.raises(ContextBudgetExceededError):
        TextSummarizer(provider, budget=RequestBudget(1000, count_request)).summarize(
            "body", "rules" * 1000
        )
    assert not provider.summary_requests


@pytest.mark.parametrize("body", ["", "tiny"])
def test_tiny_output_cannot_be_exported_into_a_larger_reference_to_fit_user_input(
    tmp_path, body
):
    provider = CompactionProvider(score=9)
    history = MessageHistory(
        [SystemMessage("rules"), UserMessage("x" * 5000), *output_batch(body)]
    )
    before = history.dump_history()
    with SessionCache(tmp_path / "cache") as session:
        files = OutputFileWorkspace(lambda: session.node("node"), tmp_path)
        with pytest.raises(ContextBudgetExceededError):
            HistoryCompactor(
                provider, files=files, budget=RequestBudget(4000, count_request)
            ).compact(history)
    assert not provider.score_requests
    assert history.dump_history() == before


def test_existing_file_reference_is_not_exported_recursively(tmp_path):
    provider = CompactionProvider(score=9)
    with SessionCache(tmp_path / "cache") as session:
        files = OutputFileWorkspace(lambda: session.node("node"), tmp_path)
        prepared = ToolOutputPreparer(provider.get_tokenizer(), files).externalize(
            ToolOutputPresentation(
                ToolOutputReference("work", "output"),
                PreformattedToolResult("source\n" * 1000),
            ),
        )
        reference = prepared.external_output
        history = MessageHistory(
            [
                SystemMessage("rules"),
                UserMessage("x" * 5000),
                *output_batch(reference.render(), external_output=reference),
            ]
        )
        with patch.object(
            files,
            "write",
            side_effect=AssertionError("recursive export"),
        ):
            with pytest.raises(ContextBudgetExceededError):
                HistoryCompactor(
                    provider, files=files, budget=RequestBudget(4000, count_request)
                ).compact(history)
    assert not provider.score_requests


def test_summary_prompt_that_occupies_the_entire_budget_leaves_no_room_for_source():
    provider = CompactionProvider()
    budget = RequestBudget(
        800, lambda request: 700 + len(str(request.get_messages()[-1].get_message()))
    )
    with pytest.raises(ContextBudgetExceededError):
        TextSummarizer(provider, budget=budget).summarize("body", "rules")
    assert not provider.summary_requests


def test_summary_split_checks_wrapped_cost_instead_of_assuming_additive_tokens():
    provider = CompactionProvider(summary="Retained evidence.")
    budget = RequestBudget(
        2000,
        lambda request: 700 + 4 * len(str(request.get_messages()[-1].get_message())),
    )
    text = "source evidence " * 70
    assert (
        TextSummarizer(provider, budget=budget).summarize(text, "Summarize")
        == provider.summary
    )
    assert len(provider.summary_requests) > 2
    for request in provider.summary_requests:
        budget.require_fits(request)
    assert (
        "".join(
            request.get_messages()[-1].get_message()
            for request in provider.summary_requests[:-1]
        )
        == text
    )
