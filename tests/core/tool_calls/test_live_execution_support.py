"""Validate live subprocess fixtures and request settings without a model endpoint."""

import json
from contextlib import ExitStack
from threading import Condition

import pytest

from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import ExecutionOptions, Tool, ToolExecutionOperation
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.tool_calls import execution_manager
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.core.web_navigation import crawler, search
from exmekh_lib.providers.local import LocalProvider
from tests.history_validation import validate_request_history
from tests.live import test_deep_thinking_skill_live as macro_live
from tests.live.model_support import (
    DEFAULT_LIVE_REQUEST_TIMEOUT_SECONDS,
    CacheTestTokenizer,
)
from tests.live.test_deep_thinking_skill_live import DeepThinkingLiveProvider
from tests.live.test_tool_cache_live import RecordingLocalProvider
from tests.live.tool_execution_support import (
    FailedProcessTool,
    NegativeProcessTool,
    ProcessFixture,
    ProcessTool,
    RecordingExecutionProvider,
    UncancellableProcessTool,
)
from tests.tokenizer_support import StubTokenizer


@pytest.mark.parametrize(
    "text, head, tail, expected",
    [
        ("abc", 2, 2, "abc"),
        ("à界😀", 2, 2, "à界😀"),
        ("abcdef", 2, 2, "abef"),
        ("abc", 0, 0, ""),
        ("abc", 0, 2, "bc"),
        ("abc", 2, 0, "ab"),
        ("", 2, 2, ""),
    ],
)
def test_live_tokenizer_truncation_matches_the_standalone_double(
    tmp_path, text, head, tail, expected
):
    tokenizer = CacheTestTokenizer("truncation-test", tmp_path / "cache")

    assert tokenizer.truncate_text(text, head, tail) == expected
    assert tokenizer.truncate_text(text, head, tail) == StubTokenizer().truncate_text(
        text, head, tail
    )


@pytest.fixture
def recording_provider(tmp_path, monkeypatch):
    monkeypatch.setenv(
        DEFAULT_PROVIDER_SETTINGS.base_url_env_var, "http://127.0.0.1:1/v1"
    )
    monkeypatch.setenv(DEFAULT_PROVIDER_SETTINGS.api_key_env_var, "offline-test-key")
    monkeypatch.setenv("EXMEKH_LIB_LIVE_MODEL", "offline-test-model")
    monkeypatch.setattr(
        LocalProvider, "_get_available_models", lambda self: {"offline-test-model"}
    )
    with ExitStack() as cleanup:

        def build(provider_type=RecordingExecutionProvider):
            provider = provider_type(tmp_path / "cache")
            cleanup.callback(provider.client.close)
            cleanup.callback(provider.get_session().close)
            return provider

        yield build


def test_macro_live_test_bounds_model_turns_and_saves_its_trace(
    recording_provider, monkeypatch, tmp_path
):
    provider = recording_provider(
        lambda directory: DeepThinkingLiveProvider(
            "offline-test-model", tmp_path, max_turns=2
        )
    )
    monkeypatch.setattr(
        LocalProvider,
        "call",
        lambda self, request: ProviderResponse(
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "retry", FunctionCall("interact-web-page", {"actions": []})
                    )
                ],
            )
        ),
    )
    request = ProviderRequest([UserMessage("complete the macro")])
    provider.call(request)
    provider.call(request)
    with pytest.raises(AssertionError, match="FLOW: exceeded 2 macro turns"):
        provider.call(request)
    repeated = [{"name": "interact-web-page", "arguments": {"actions": []}}]
    assert json.loads((tmp_path / "macro-model-trace.json").read_text()) == [
        repeated,
        repeated,
    ]


def test_macro_fixture_limits_choices_without_substituting_real_arguments(
    recording_provider, monkeypatch, tmp_path
):
    provider = recording_provider(
        lambda directory: DeepThinkingLiveProvider(
            "offline-test-model", tmp_path, steps=[("work", {"value": "REQUESTED"})]
        )
    )
    seen = []
    response = ProviderResponse(
        AssistantMessage(
            "",
            tool_calls=[ToolCall("call", FunctionCall("work", {"value": "OBSERVED"}))],
        )
    )

    def call(self, request):
        seen.append(request)
        return response

    monkeypatch.setattr(LocalProvider, "call", call)
    request = ProviderRequest(
        [SystemMessage("test"), UserMessage("start")],
        tools=Tools(
            [
                Tool(lambda value, mode="safe": value, name="work"),
                Tool(lambda: None, name="other"),
            ]
        ),
        extra={
            "agentic": {"allowed_tools": ["work", "other"], "allowed_skills": ["skill"]}
        },
    )
    assert provider.call(request) is response
    (bounded,) = seen
    assert bounded.get_tools().list_tools() == ["work", "other"]
    assert bounded.get_tool_choice() == "required"
    assert '"mode": "safe"' in bounded.get_messages()[-1].get_message()
    assert bounded.get_extra()["agentic"] == {
        "allowed_tools": ["work"],
        "allowed_skills": [],
    }
    assert request.get_tools().list_tools() == ["work", "other"]
    assert len(request.get_messages()) == 2
    assert response.get_tool_calls()[0].get_arguments() == {"value": "OBSERVED"}


def test_macro_final_reply_keeps_the_real_return_checkpoint_last(
    recording_provider, monkeypatch, tmp_path
):
    provider = recording_provider(
        lambda directory: DeepThinkingLiveProvider(
            "offline-test-model", tmp_path, steps=[]
        )
    )
    seen = []
    monkeypatch.setattr(
        LocalProvider,
        "call",
        lambda self, request: (
            seen.append(request) or ProviderResponse(AssistantMessage("DONE"))
        ),
    )
    message = ToolMessage("complete", "skill-call", "deep-thinking")
    provider.call(
        ProviderRequest(
            [
                SystemMessage("test"),
                UserMessage("start"),
                AssistantMessage(
                    "",
                    tool_calls=[
                        ToolCall("skill-call", FunctionCall("deep-thinking", {}))
                    ],
                ),
                message,
            ],
            tools=Tools([]),
            extra={
                "agentic": {
                    "operation": "return",
                    "skill_call_id": "skill-call",
                    "allowed_tools": [],
                    "allowed_skills": [],
                }
            },
        )
    )
    (bounded,) = seen
    assert bounded.get_messages()[-1] is message
    assert bounded.get_tool_choice() == "none"
    assert bounded.get_extra()["agentic"]["skill_call_id"] == "skill-call"


@pytest.mark.parametrize(
    "provider_type", [RecordingExecutionProvider, RecordingLocalProvider]
)
def test_live_provider_increases_http_timeout_independently_of_tool_wait(
    recording_provider, monkeypatch, provider_type
):
    monkeypatch.delenv("EXMEKH_LIB_LIVE_REQUEST_TIMEOUT_SECONDS", raising=False)
    provider = recording_provider(provider_type)
    assert provider.client.timeout == DEFAULT_LIVE_REQUEST_TIMEOUT_SECONDS
    assert provider.client.max_retries == 0
    assert ExecutionOptions().wait_seconds == 60


@pytest.mark.parametrize(
    "provider_type", [RecordingExecutionProvider, RecordingLocalProvider]
)
@pytest.mark.parametrize("seconds", [120, 2.5])
def test_live_provider_accepts_configured_request_timeout(
    recording_provider, monkeypatch, seconds, provider_type
):
    monkeypatch.setenv("EXMEKH_LIB_LIVE_REQUEST_TIMEOUT_SECONDS", str(seconds))
    provider = recording_provider(provider_type)
    assert provider.client.timeout == seconds
    assert provider.client.max_retries == 0


@pytest.mark.parametrize(
    "provider_type", [RecordingExecutionProvider, RecordingLocalProvider]
)
@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf", "", "invalid"])
def test_live_provider_rejects_invalid_request_timeout(
    recording_provider, monkeypatch, value, provider_type
):
    monkeypatch.setenv("EXMEKH_LIB_LIVE_REQUEST_TIMEOUT_SECONDS", value)
    with pytest.raises(ValueError, match="EXMEKH_LIB_LIVE_REQUEST_TIMEOUT_SECONDS"):
        recording_provider(provider_type)


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


def test_default_minute_deadline_keeps_the_real_process_alive(tmp_path, monkeypatch):
    deadlines = []

    class ExpiredDefaultWait(Condition):
        def wait_for(self, predicate, timeout=None):
            if timeout == 60:
                deadlines.append(timeout)
                return False
            return super().wait_for(predicate, timeout)

    monkeypatch.setattr(execution_manager, "Condition", ExpiredDefaultWait)
    fixture = ProcessFixture(tmp_path)
    manager = ToolExecutionManager()
    tool = ProcessTool(fixture)
    operation = tool.create_execution_with_hooks(
        {"label": "minute-default"}, call_id="origin", work_root=tmp_path, history=()
    )
    try:
        result = manager.start(
            "origin",
            tool.name,
            {"label": "minute-default"},
            operation,
            ("root",),
            ExecutionOptions(),
            output_id="output",
        )
        assert deadlines == [60]
        assert result.status == "running"
        job = fixture.job("origin")
        assert job.process.poll() is None
        job.send("finish")
        assert manager.wait("origin", 5).status == "completed"
        assert job.receipt["sha256"] == fixture.sha256
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


@pytest.mark.parametrize("tool_type", [FailedProcessTool, NegativeProcessTool])
def test_real_nonzero_exit_distinguishes_failure_from_negative_result(
    tmp_path, tool_type
):
    fixture = ProcessFixture(tmp_path)
    manager = ToolExecutionManager()
    tool = tool_type(fixture)
    operation = tool.create_execution_with_hooks(
        {"label": "negative"}, call_id="origin", work_root=tmp_path, history=()
    )
    try:
        result = manager.start(
            "origin",
            tool.name,
            {"label": "negative"},
            operation,
            ("root",),
            ExecutionOptions(wait_seconds=5),
            output_id="output",
        )
        assert fixture.job("origin").process.poll() == 7
        if tool_type is FailedProcessTool:
            assert result.status == "failed"
            assert (
                result.error == "RuntimeError: Fixture process failed with exit code 7"
            )
        else:
            assert result.status == "completed"
            assert result.result["exit_code"] == 7
            assert result.result["sha256"] == fixture.sha256
    finally:
        fixture.close()


@pytest.mark.parametrize("tool_type", [ProcessTool, UncancellableProcessTool])
def test_real_process_cancellation_respects_operation_support(tmp_path, tool_type):
    fixture = ProcessFixture(tmp_path)
    manager = ToolExecutionManager()
    tool = tool_type(fixture)
    operation = tool.create_execution_with_hooks(
        {"label": "cancel"}, call_id="origin", work_root=tmp_path, history=()
    )
    try:
        manager.start(
            "origin",
            tool.name,
            {"label": "cancel"},
            operation,
            ("root",),
            ExecutionOptions(mode="background"),
            output_id="output",
        )
        job = fixture.job("origin")
        supported, cancelled, snapshot = manager.cancel("origin")
        if tool_type is ProcessTool:
            assert supported and cancelled
            assert snapshot.status == "cancelled"
            assert job.process.poll() is not None
            assert not job.receipt
        else:
            assert not supported and not cancelled
            assert snapshot.status == "running"
            assert job.process.poll() is None
            job.send("finish")
            assert manager.wait("origin", 5).status == "completed"
            assert job.receipt
    finally:
        fixture.close()


def test_controlled_macro_fixture_runs_every_real_reasoning_tool_offline(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(
        DEFAULT_PROVIDER_SETTINGS.base_url_env_var, "http://127.0.0.1:1/v1"
    )
    monkeypatch.setenv(DEFAULT_PROVIDER_SETTINGS.api_key_env_var, "offline-test-key")
    monkeypatch.setenv("EXMEKH_LIB_LIVE_DEEP_THINKING_MODEL", "offline-test-model")
    monkeypatch.setattr(
        LocalProvider, "_get_available_models", lambda self: {"offline-test-model"}
    )
    original = macro_live.DeepThinkingLiveProvider
    monkeypatch.setattr(
        macro_live,
        "DeepThinkingLiveProvider",
        lambda model, directory, **options: original(
            model,
            directory,
            tokenizer=CacheTestTokenizer("macro-fixture", directory / "cache"),
            **options,
        ),
    )
    steps = dict(macro_live.macro_steps())
    seen = []

    def call(self, request):
        validate_request_history(request)
        seen.append(request)
        if request.get_tool_choice() == "none":
            return ProviderResponse(AssistantMessage("DONE"))
        scope = request.get_extra()["agentic"]
        (name,) = scope["allowed_tools"] + scope["allowed_skills"]
        return ProviderResponse(
            AssistantMessage(
                "",
                tool_calls=[
                    ToolCall(f"call-{len(seen)}", FunctionCall(name, steps[name]))
                ],
            )
        )

    monkeypatch.setattr(LocalProvider, "call", call)
    monkeypatch.setattr(
        search,
        "search_web",
        lambda query, **arguments: {
            "result_count": 1,
            "results": [{"url": steps["read-web-page"]["url"]}],
        },
    )
    monkeypatch.setattr(
        crawler,
        "crawl_page",
        lambda **arguments: {
            "status_code": 200,
            "final_url": "https://docs.python.org/3/library/math.html"
            if arguments.get("actions")
            else arguments["url"],
            "markdown": "Actual fixture content",
            "actions_completed": True,
        },
    )
    macro_live.test_live_model_can_choose_and_combine_every_skill_tool(tmp_path)
    assert len(seen) == len(steps) + 1
