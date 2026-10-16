"""Live workflow invariants; scores and summary wording are intentionally variable."""

from pathlib import Path

import pytest

from exmekh_lib.config.tool_output import ToolOutputSettings
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.history_compaction import HistoryCompactor
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.tool_calls.output_preparer import ToolOutputPreparer
from tests.history_validation import validate_request_history
from tests.live.history_compaction_support import (
    PendingWorkflow,
    RecordingCompactionProvider,
    compaction_outputs,
    fixture_tools,
    history_tokens,
    synthetic_history,
    write_observation,
)

pytestmark = pytest.mark.live


@pytest.fixture(params=["chat_completions", "responses"])
def compaction_provider(request):
    provider = RecordingCompactionProvider(request.param)
    try:
        yield provider
    finally:
        provider.close()


@pytest.mark.parametrize(
    "trial", [0, 1, 2], ids=["repeat-1", "repeat-2", "uncached-files"]
)
def test_live_synthetic_history_compacts_and_workflow_remains_usable(
    compaction_provider, tmp_path: Path, trial
):
    provider = compaction_provider
    cache = NodeCache("live-compaction", provider.get_session())
    files = OutputFileWorkspace(
        lambda: cache, tmp_path, ToolOutputSettings(max_tokens=5000)
    )
    preparer = (
        ToolOutputPreparer(provider.get_tokenizer(), files) if trial == 2 else None
    )
    history = synthetic_history(preparer)
    before = history.get_history()
    original_dump = history.dump_history()
    tools = fixture_tools()
    compactor = HistoryCompactor(provider, files=files)
    result = compactor.compact(history, tools=tools)
    after = result.history.get_history()

    assert history.dump_history() == original_dump, "CLIENT: source history was mutated"
    assert after[0] is before[0], "CLIENT: system prompt was replaced"
    assert [message for message in after if message.get_role_name() == "user"] == [
        message for message in before if message.get_role_name() == "user"
    ], "CLIENT: user messages or order changed"
    validate_request_history(ProviderRequest(after, tools=tools))
    assert result.output_count == 3
    entries = compaction_outputs(after)
    assert {entry["id"] for entry in entries} == {
        "runtime-report",
        "archived-log",
        "dispatch-plan",
    }, "CLIENT: output identities were lost"
    assert all(1 <= entry["score"] <= 100 for entry in entries)
    assert all(
        isinstance(entry.get("content"), str) and entry["content"].strip()
        for entry in entries
        if entry["representation"] == "summary"
    )
    assert any(entry["representation"] == "summary" for entry in entries), (
        "MODEL: no long output was summarized in this compaction"
    )
    if trial != 2:
        assert history_tokens(provider, after) < history_tokens(provider, before), (
            "MODEL: synthetic repetitive history was not reduced"
        )
    for request in provider.requests:
        assert request.get_reasoning_effort_override() == "none"
    scoring_request = provider.scoring_requests[0]
    assert scoring_request.get_tools() is tools
    assert [
        message.message_dump() for message in scoring_request.get_messages()[:-1]
    ] == original_dump
    assert cache.get_all() == [], "CLIENT: uncached synthetic output was persisted"

    probe = provider.call_structured(
        StructuredProviderRequest(
            [
                *after,
                UserMessage(
                    "Using only the retained conversation, report the batch, target queue, "
                    "approval checkpoint, whether anything was dispatched, and whether "
                    "approval is still required. Use the requested fields; do not invent facts."
                ),
            ],
            PendingWorkflow,
            tools=tools,
            tool_choice="none",
            reasoning_effort_override="none",
        )
    ).get_message()
    assert probe.batch == "DELTA-73"
    assert probe.queue.casefold() == "staging"
    assert probe.checkpoint == "CP-618"
    assert probe.dispatched is False and probe.requires_approval is True, (
        "MODEL: global pending/side-effect state changed after compaction"
    )

    calls_before = len(provider.requests)
    repeated = compactor.compact(result.history, tools=tools)
    assert repeated.history.dump_history() == result.history.dump_history()
    assert len(provider.requests) == calls_before, (
        "CLIENT: unchanged summaries were regenerated"
    )
    if trial == 2:
        assert next(entry for entry in entries if entry["id"] == "runtime-report")[
            "files"
        ]["files"]

    write_observation(
        tmp_path / "live-compaction-observation.json",
        provider,
        before,
        after,
        trial=trial,
        memory_probe=probe.model_dump(),
    )
