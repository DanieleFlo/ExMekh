from threading import Event

import pytest

from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.external_tool_output import ExternalToolOutput
from exmekh_lib.core.models.tool import (
    PreformattedToolResult,
    Tool,
)
from exmekh_lib.interfaces.tool_hook import AfterTool
from tests.core.agents.node.test_node import SequenceProvider
from tests.core.agents.node.test_tool_executions import make_node as make_node
from tests.core.agents.node.test_tool_executions import read_calls, tool_message
from tests.tool_result_support import parse_tool_result


@pytest.mark.parametrize("mode", ["synchronous", "wait", "background"])
@pytest.mark.parametrize("cached", [True, False])
def test_large_outputs_are_replaced_in_history_cache_and_runtime_reads(
    make_node, tmp_path, mode, cached
):
    original = "PRIVATE" * 1000
    release = Event()

    def work():
        if mode == "background":
            assert release.wait(2)
        return original

    arguments = {} if mode == "synchronous" else {"execution_mode": mode}
    provider = SequenceProvider(
        [tool_message("mock-tool", arguments, "origin"), AssistantMessage("done")]
    )
    node = make_node(
        provider,
        Tool(
            work,
            name="mock-tool",
            cache_result=cached,
            background_enabled=mode != "synchronous",
        ),
    )
    try:
        result = node.run("work")
        if mode == "background":
            release.set()
            node.control_tool_execution("origin", wait_seconds=1)
        read = node.read_tool_output("output-1")
        recoverable = cached or mode != "synchronous"
        reference = (
            read.external_output
            if recoverable
            else ExternalToolOutput.from_dict(
                parse_tool_result(
                    next(
                        item["content"]
                        for item in result.history
                        if item["role"] == "tool"
                    )
                )
            )
        )
        assert reference is not None
        assert len(reference.files) == 2
        assert (
            "".join((tmp_path / item.path).read_text() for item in reference.files)
            == original
        )
        assert original not in read.content
        for summary_mode in ("none", "brief", "detailed") if recoverable else ():
            again = node.read_tool_output(
                "output-1", cursor=0, summary_mode=summary_mode
            )
            assert again.external_output == reference
            assert original not in again.content
        assert all(original not in item["content"] for item in result.history)
        records = list(provider.get_session().session_dir.rglob("*.json"))
        assert bool(records) == cached
        assert all(original not in path.read_text() for path in records)
    finally:
        release.set()


def test_background_automatic_delivery_and_saved_record_only_contain_files(
    make_node, tmp_path
):
    original = "raw text\n" * 1000
    release = Event()

    def work():
        assert release.wait(2)
        return original

    provider = SequenceProvider(
        [
            tool_message("mock-tool", {"execution_mode": "background"}, "origin"),
            tool_message(
                "control_tool_execution",
                {"tool_call_id": "origin", "wait_seconds": 1},
                "wait",
            ),
            AssistantMessage("done"),
            AssistantMessage("again"),
        ]
    )
    node = make_node(provider, Tool(work, name="mock-tool"))
    original_call = provider.call

    def call(request):
        if len(provider.requests) == 1:
            release.set()
        return original_call(request)

    provider.call = call
    try:
        result = node.run("work")
        reads = read_calls(provider.requests[-1].get_messages())
        assert len(reads) == 1
        read = node.read_tool_output("output-1")
        assert len(read.external_output.files) == 1
        assert (tmp_path / read.external_output.files[0].path).read_text() == original
        assert all(original not in item["content"] for item in result.history)
        assert node.run("next").get_message() == "again"
        assert len(read_calls(provider.requests[-1].get_messages())) == 1
        assert all(
            original not in path.read_text()
            for path in provider.get_session().session_dir.rglob("*.json")
        )
    finally:
        release.set()


def test_after_hook_filters_before_export_once(make_node, tmp_path):
    calls = []
    raw = "unfiltered secret" * 1000
    filtered = "public" * 1000

    class Filter(AfterTool):
        def handle(self, context, result):
            calls.append(result)
            return filtered

    provider = SequenceProvider(
        [tool_message("mock-tool", {}, "origin"), AssistantMessage("done")]
    )
    node = make_node(provider, Tool(lambda: raw, name="mock-tool", after=Filter()))
    node.run("work")
    read = node.read_tool_output("output-1")
    node.read_tool_output("output-1")
    assert calls == [raw]
    assert (
        "".join(
            (tmp_path / item.path).read_text() for item in read.external_output.files
        )
        == filtered
    )
    assert not any(raw in path.read_text() for path in tmp_path.rglob("*.txt"))


def test_preformatted_output_is_exported_without_losing_its_format(make_node, tmp_path):
    text = "# Markdown\n\n" + "text <&>  \n" * 1000
    provider = SequenceProvider(
        [tool_message("mock-tool", {}, "origin"), AssistantMessage("done")]
    )
    node = make_node(
        provider,
        Tool(
            lambda: PreformattedToolResult(text),
            name="mock-tool",
            background_enabled=False,
        ),
    )
    node.run("work")
    read = node.read_tool_output("output-1")
    assert (tmp_path / read.external_output.files[0].path).read_text() == text


def test_large_execution_error_is_exported_with_failed_status(make_node, tmp_path):
    diagnostic = "failure " * 1000

    def work():
        raise ValueError(diagnostic)

    provider = SequenceProvider(
        [tool_message("mock-tool", {}, "origin"), AssistantMessage("done")]
    )
    node = make_node(provider, Tool(work, name="mock-tool"))
    node.run("work")
    read = node.read_tool_output("output-1")
    assert "<response-type>failed</response-type>" in read.content
    assert diagnostic not in read.content
    assert diagnostic in (tmp_path / read.external_output.files[0].path).read_text()


def test_skill_items_keep_references_after_compaction(make_node, tmp_path):
    original = "PRIVATE" * 1000
    provider = SequenceProvider(
        [
            tool_message("mock-skill", {}, "open"),
            tool_message("mock-sub-tool", {}, "work"),
            tool_message(
                "exit_agent", {"text": "summary", "items": ["output-1"]}, "exit"
            ),
            AssistantMessage("done"),
        ]
    )
    node = make_node(provider, Tool(lambda: original, name="mock-sub-tool"))
    result = node.run("work in a skill")
    assert all(original not in item["content"] for item in result.history)
    assert original not in result.skill_executions[0].output
    assert "<files" in node.read_tool_output("output-1").content
    returned = node.read_tool_output("output-2").content
    assert "summary" in returned and "0001.txt" in returned
    assert original not in returned


def test_large_skill_return_is_exported_before_tree_and_cache_recording(
    make_node, tmp_path
):
    original = "result\n" * 1000
    provider = SequenceProvider(
        [
            tool_message("mock-skill", {}, "open"),
            tool_message("exit_agent", {"text": original, "items": []}, "exit"),
            AssistantMessage("done"),
        ]
    )
    node = make_node(provider, Tool(lambda: "unused", name="mock-sub-tool"))
    result = node.run("return a large result")
    returned = node.read_tool_output("output-1")
    assert returned.external_output is not None
    assert "<type>DeepSkill</type>" in returned.content
    assert len(returned.external_output.files) == 1
    assert (
        tmp_path / returned.external_output.files[0].path
    ).read_text() == original.strip()
    assert original not in result.skill_executions[0].output
    assert all(
        original not in path.read_text()
        for path in provider.get_session().session_dir.rglob("*.json")
    )
