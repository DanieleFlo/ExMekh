from dataclasses import replace
from threading import Event
from unittest.mock import MagicMock
from xml.etree import ElementTree as ET

import pytest
from PIL import Image

from exmekh_lib.capabilities.tools.read_tool_output import summary
from exmekh_lib.capabilities.tools.read_tool_output.summary import ToolOutputSummarizer
from exmekh_lib.capabilities.tools.read_tool_output.tool_output_reader import (
    ToolOutputReader,
)
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.cache import SessionCache
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.cache.user_message_codec import serialize_user_message
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.tool import (
    ExecutionOptions,
    PreformattedToolResult,
    ToolExecutionOperation,
)
from exmekh_lib.core.models.user_message import ImagesUserMessage, UserMessage
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.interfaces.provider import Provider
from tests.core.tool_calls.test_output_reader import unavailable_cache


def split_read(result):
    header, body = result.content.split("</header>\n", 1)
    return ET.fromstring(header + "</header>"), body


@pytest.fixture
def provider():
    provider = MagicMock(spec=Provider)
    provider.get_tokenizer.return_value.count_tokens.side_effect = len
    provider.call.return_value = ProviderResponse(
        AssistantMessage("essential <&> facts")
    )
    return provider


@pytest.mark.parametrize("mode", ["none", "brief", "detailed"])
@pytest.mark.parametrize("tokens", [299, 300, 301])
def test_summary_threshold_ignores_metadata_and_default_never_calls_model(
    tmp_path, provider, mode, tokens
):
    original = ToolCallRecord("work", "output", {"metadata": "x" * 500}, "x" * tokens)
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        cache.store(original)
        reader = ToolOutputReader(
            lambda: cache, summarizer=ToolOutputSummarizer(provider)
        )
        header, body = split_read(reader.read("output", summary_mode=mode))
        summarized = mode != "none" and tokens >= 300
        assert body == ("essential <&> facts" if summarized else original.result)
        assert header.findtext("summary-mode") == (mode if summarized else None)
        assert header.findtext("id") == original.id
        assert header.findtext("name") == "work"
        assert cache.get("output") == original
    assert provider.call.call_count == int(summarized)
    assert provider.get_tokenizer.call_count == int(mode != "none")
    if summarized:
        request = provider.call.call_args.args[0]
        assert request.get_reasoning_effort_override() == "none"
        assert request.get_tools() is None
        assert request.get_extra() is None
        assert request.get_streaming() is False
        messages = request.get_messages()
        assert len(messages) == 2
        assert messages[0].get_message() == (
            RUNTIME.tool_output_brief_summary_prompt
            if mode == "brief"
            else RUNTIME.tool_output_detailed_summary_prompt
        )
        assert messages[1].get_message() == original.result


def test_summary_threshold_is_configurable(provider, monkeypatch):
    monkeypatch.setattr(
        summary, "RUNTIME", replace(RUNTIME, tool_output_summary_min_tokens=2)
    )
    output = PreformattedToolResult("<header>\n<id>output</id>\n</header>\nxx")
    assert (
        split_read(ToolOutputSummarizer(provider).summarize(output, "brief"))[1]
        == "essential <&> facts"
    )


@pytest.mark.parametrize("failure", ["tokenizer", "provider", "empty", "tool_call"])
def test_summary_failure_returns_complete_original_output(
    tmp_path, provider, failure, caplog
):
    original = ToolCallRecord("work", "output", {}, "x" * 300)
    if failure == "tokenizer":
        provider.get_tokenizer.side_effect = RuntimeError("tokenizer unavailable")
    elif failure == "provider":
        provider.call.side_effect = RuntimeError("model unavailable")
    else:
        provider.call.return_value = ProviderResponse(
            AssistantMessage("   " if failure == "empty" else None)
        )
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        cache.store(original)
        plain = ToolOutputReader(lambda: cache).read("output")
        actual = ToolOutputReader(
            lambda: cache, summarizer=ToolOutputSummarizer(provider)
        ).read("output", summary_mode="brief")
        assert actual == plain
        assert cache.get("output") == original
    assert "returning full output" in caplog.text


@pytest.mark.parametrize("mode", ["invalid", None, [], True])
def test_invalid_modes_fail_before_accessing_output(mode):
    with pytest.raises(ValueError, match="summary_mode"):
        ToolOutputReader(unavailable_cache).read("output", summary_mode=mode)


def test_summary_requires_provider_but_default_and_missing_outputs_do_not(
    tmp_path, provider
):
    reader = ToolOutputReader(unavailable_cache)
    with pytest.raises(ValueError, match="configured provider"):
        reader.read("output", summary_mode="brief")
    with SessionCache(tmp_path) as session:
        reader = ToolOutputReader(
            lambda: session.node("node"), summarizer=ToolOutputSummarizer(provider)
        )
        assert (
            "<found>false</found>"
            in reader.read("missing", summary_mode="brief").content
        )
    provider.call.assert_not_called()


def test_cursor_summary_keeps_original_cursor_attachments_and_separate_client_context(
    tmp_path, provider
):
    companion = UserMessage(
        ["attachment context", ImagesUserMessage([Image.new("RGB", (2, 2), "red")])]
    )
    record = ToolCallRecord(
        "work",
        "output",
        {},
        "first" + "x" * 300,
        user_message=serialize_user_message(companion),
        execution={"status": "completed", "elapsed_seconds": 4},
    )
    with SessionCache(tmp_path) as session:
        cache = session.node("node")
        cache.store(record)
        reader = ToolOutputReader(
            lambda: cache, summarizer=ToolOutputSummarizer(provider)
        )
        for mode in ("brief", "detailed"):
            result = reader.read("output", cursor=5, summary_mode=mode)
            header, body = split_read(result)
            assert int(header.findtext("next-cursor")) == 305
            assert header.findtext("elapsed") == "4s"
            assert header.findtext("status") == "completed"
            assert header.findtext("response-type") == "final"
            assert body == "essential <&> facts"
            image = next(
                part
                for part in result.attachments.iter_content_parts()
                if isinstance(part, Image.Image)
            )
            assert image.getpixel((0, 0)) == (255, 0, 0)
        assert cache.get("output") == record
    requests = [call.args[0] for call in provider.call.call_args_list]
    assert len(requests) == 2
    assert all(len(request.get_messages()) == 2 for request in requests)
    assert all(
        request.get_messages()[1].get_message() == "x" * 300 for request in requests
    )


@pytest.mark.parametrize("finished", [False, True])
def test_runtime_summary_preserves_status_progress_cursor_and_delivery(
    provider, finished
):
    release = Event()
    progress = "x" * 300
    manager = ToolExecutionManager()
    manager.start(
        "call",
        "work",
        {},
        ToolExecutionOperation(
            lambda: release.wait() and "final result",
            read_partial=lambda cursor: (progress[cursor:], len(progress)),
        ),
        ("owner",),
        ExecutionOptions(mode="background"),
        output_id="output",
    )
    reader = ToolOutputReader(
        unavailable_cache, manager, ToolOutputSummarizer(provider)
    )
    try:
        if finished:
            release.set()
            assert manager.wait("call", 1).is_finished
        result = reader.read("output", cursor=0, summary_mode="brief")
        header, body = split_read(result)
        assert header.findtext("status") == ("completed" if finished else "running")
        assert header.findtext("response-type") == ("final" if finished else "partial")
        assert header.findtext("next-cursor") == "300"
        assert header.find("elapsed") is not None
        assert body == "essential <&> facts"
        assert manager.next_delivery(("owner",)) is None
        request_body = provider.call.call_args.args[0].get_messages()[1].get_message()
        assert request_body == (progress + "\nfinal result" if finished else progress)
    finally:
        release.set()
        assert manager.wait("call", 1).is_finished


def test_summary_preserves_non_header_tool_authored_text(provider):
    original = PreformattedToolResult("# authored output")
    assert ToolOutputSummarizer(provider).summarize(original, "brief") is original
    provider.call.assert_not_called()


@pytest.mark.parametrize(
    "prefix", ["authored </header>\n", "<header><name></header>\n"]
)
def test_summaries_ignore_embedded_or_malformed_headers(provider, prefix):
    original = PreformattedToolResult(prefix + "x" * 300)
    assert ToolOutputSummarizer(provider).summarize(original, "brief") is original
    provider.call.assert_not_called()


def test_summary_response_with_unexpected_tool_calls_keeps_original_output(provider):
    original = PreformattedToolResult("<header><id>output</id></header>\n" + "x" * 300)
    provider.call.return_value = ProviderResponse(
        AssistantMessage(
            "some text",
            tool_calls=[ToolCall("unexpected", FunctionCall("unknown", {}))],
        )
    )
    assert ToolOutputSummarizer(provider).summarize(original, "brief") is original


@pytest.mark.parametrize("separator", ["", "\n", "\r\n"])
def test_summary_accepts_valid_headers_with_any_line_separator(provider, separator):
    original = PreformattedToolResult(
        "<header><id>output</id></header>" + separator + "x" * 300
    )
    result = ToolOutputSummarizer(provider).summarize(original, "brief")
    assert XmlHeader.body_from_content(result.content) == "essential <&> facts"
    provider.get_tokenizer.return_value.count_tokens.assert_called_once_with("x" * 300)


def test_direct_summarizer_rejects_invalid_mode_without_calling_provider(provider):
    with pytest.raises(ValueError, match="summary_mode"):
        ToolOutputSummarizer(provider).summarize(
            PreformattedToolResult("output"), "invalid"
        )
    provider.call.assert_not_called()
