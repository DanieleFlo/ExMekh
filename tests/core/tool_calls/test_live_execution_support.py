"""Validate live subprocess fixtures and request settings without a model endpoint."""

from contextlib import ExitStack

import pytest
from openai import DEFAULT_TIMEOUT

from qs.core.models.tool import ExecutionOptions, ToolExecutionOperation
from qs.core.tool_calls.execution_manager import ToolExecutionManager
from qs.providers.local import LocalProvider
from tests.live.tool_execution_support import (
    ProcessFixture,
    ProcessTool,
    RecordingExecutionProvider,
)


@pytest.fixture
def recording_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("API_KEY", "offline-test-key")
    monkeypatch.setenv("QS_LIVE_MODEL", "offline-test-model")
    monkeypatch.setattr(
        LocalProvider, "_get_available_models", lambda self: {"offline-test-model"}
    )
    with ExitStack() as cleanup:

        def build():
            provider = RecordingExecutionProvider(tmp_path / "cache")
            cleanup.callback(provider.client.close)
            cleanup.callback(provider.get_session().close)
            return provider

        yield build


def test_live_provider_keeps_default_http_timeout_independent_of_tool_wait(
    recording_provider, monkeypatch
):
    monkeypatch.delenv("QS_LIVE_REQUEST_TIMEOUT_SECONDS", raising=False)
    provider = recording_provider()
    assert provider.client.timeout == DEFAULT_TIMEOUT
    assert provider.client.max_retries == 0
    assert ExecutionOptions().wait_seconds == 60


@pytest.mark.parametrize("seconds", [120, 2.5])
def test_live_provider_accepts_configured_request_timeout(
    recording_provider, monkeypatch, seconds
):
    monkeypatch.setenv("QS_LIVE_REQUEST_TIMEOUT_SECONDS", str(seconds))
    provider = recording_provider()
    assert provider.client.timeout == seconds
    assert provider.client.max_retries == 0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "", "invalid"])
def test_live_provider_rejects_invalid_request_timeout(
    recording_provider, monkeypatch, value
):
    monkeypatch.setenv("QS_LIVE_REQUEST_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="QS_LIVE_REQUEST_TIMEOUT_SECONDS"):
        recording_provider()


def test_real_process_wait_expiry_keeps_progress_and_original_process(tmp_path):
    fixture = ProcessFixture(tmp_path)
    manager = ToolExecutionManager()
    tool = ProcessTool(fixture)
    operation = tool.create_execution_with_hooks(
        {"label": "offline-fixture"}, call_id="origin", work_root=tmp_path, history=()
    )
    try:
        initial = manager.start(
            "origin",
            tool.name,
            {"label": "offline-fixture"},
            operation,
            ("scope",),
            ExecutionOptions(wait_seconds=0.01),
            output_id="output",
        )
        job = fixture.job("origin")
        assert initial.status == "running"
        assert manager.wait("origin", 0.01).status == "running"
        assert job.process.poll() is None
        content, cursor = manager.partial_output("output", 0)
        assert '"phase": "started"' in content
        job.send("progress")
        assert job.advanced.wait(5)
        content, next_cursor = manager.partial_output("output", cursor)
        assert '"phase": "advanced"' in content
        assert '"phase": "started"' not in content
        assert next_cursor > cursor
        job.send("finish")
        completed = manager.wait("origin", 5)
        assert completed.status == "completed"
        assert completed.result["sha256"] == fixture.sha256
        assert completed.result["pid"] == job.process.pid
        assert job.process.poll() == 0
    finally:
        fixture.close()


def test_real_process_cleanup_stops_unfinished_work(tmp_path):
    fixture = ProcessFixture(tmp_path)
    manager = ToolExecutionManager()
    job = fixture.launch("origin", "cleanup", "gated")
    manager.start(
        "origin",
        "process",
        {},
        ToolExecutionOperation(job.run),
        ("scope",),
        ExecutionOptions(mode="background"),
    )
    fixture.job("origin")
    fixture.close()
    assert job.process.poll() is not None
    assert job.reader_done.is_set()
    assert manager.wait("origin", 5).is_finished


def test_cleanup_prevents_late_workers_from_launching_an_orphan_process(tmp_path):
    fixture = ProcessFixture(tmp_path)
    fixture.close()
    with pytest.raises(RuntimeError, match="already closed"):
        fixture.launch("late", "cleanup", "gated")
    assert not fixture.jobs
