import shutil
import time
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

import pytest

from qs.capabilities.tools.bash.tool import BashTool
from qs.capabilities.tools.read_tool_output.output_reader import ToolOutputReader
from qs.core.agents.capabilities.defaults import DefaultCapabilities
from qs.core.agents.node.node import Node
from qs.core.cache import SessionCache
from qs.core.models.assistant_message import AssistantMessage, FunctionCall, ToolCall
from qs.core.models.tool import ExecutionOptions, PreformattedToolResult, Tool
from qs.core.models.tool_message import ToolMessage
from qs.core.tool_calls.execution_manager import ToolExecutionManager
from qs.interfaces.tool_hook import (
    AfterTool,
    BeforeTool,
    ToolArguments,
    ToolContext,
    ToolResponse,
)
from tests.core.agents.node.test_node import (
    SequenceProvider,
    _write_nested_capabilities,
)


@pytest.fixture
def bash_executable() -> str:
    found = shutil.which("bash")
    if found is None:
        candidate = Path("C:/Program Files/Git/bin/bash.exe")
        if candidate.is_file():
            found = str(candidate)
    if found is None:
        pytest.skip("Bash is not installed")
    return found


def start_bash(
    manager: ToolExecutionManager,
    executable: str,
    work_root: Path,
    command: str,
    call_id: str,
    *,
    mode: str = "wait",
    wait_seconds: float = 1,
    tool: BashTool | None = None,
):
    tool = tool or BashTool(executable=executable)
    return manager.start(
        call_id,
        "bash",
        {"command": command},
        tool.create_execution_with_hooks(
            {"command": command},
            call_id=call_id,
            work_root=work_root,
            history=(),
        ),
        ("node",),
        ExecutionOptions(mode=mode, wait_seconds=wait_seconds),
        output_id=f"output-{call_id}",
    )


def test_bash_short_command_reports_cwd_output_and_nonzero_exit(
    bash_executable: str, tmp_path: Path
) -> None:
    manager = ToolExecutionManager()
    first = start_bash(manager, bash_executable, tmp_path, "pwd; printf hello", "one")
    assert first.status == "completed"
    assert first.result["exit_code"] == 0
    assert first.result["stdout"].endswith("hello")
    assert first.result["cwd"] == str(tmp_path)
    assert "test_bash_short_command" in first.result["stdout"]
    assert first.result["started_at"] <= first.result["finished_at"]

    failed_command = start_bash(
        manager, bash_executable, tmp_path, "printf problem >&2; exit 7", "two"
    )
    assert failed_command.status == "completed"
    assert failed_command.result["exit_code"] == 7
    assert failed_command.result["stderr"] == "problem"


def test_bash_background_progress_uses_cursor_and_finishes_after_initial_wait(
    bash_executable: str, tmp_path: Path
) -> None:
    manager = ToolExecutionManager()
    initial = start_bash(
        manager,
        bash_executable,
        tmp_path,
        "printf first; PATH=/usr/bin:$PATH sleep 1; printf second",
        "slow",
        wait_seconds=0.01,
    )
    assert initial.status == "running"
    reader = ToolOutputReader(lambda: pytest.fail("Runtime output used cache"), manager)
    deadline = time.monotonic() + 5
    while True:
        first_read = reader.read("output-slow", cursor=0).content
        if "[stdout] first" in first_read:
            break
        assert time.monotonic() < deadline
        time.sleep(0.02)
    header = ET.fromstring(first_read.split("</header>", 1)[0] + "</header>")
    cursor = int(header.findtext("next-cursor"))
    assert cursor > 0
    next_read = reader.read("output-slow", cursor=cursor).content
    if "<response-type>final</response-type>" not in next_read:
        assert "first" not in next_read.split("</header>", 1)[1]
    assert manager.wait("slow", 5).status == "completed"
    final = reader.read("output-slow", cursor=cursor).content
    assert "second" in final
    assert "<response-type>final</response-type>" in final
    assert '<exit_code type="integer">0</exit_code>' in final


def test_bash_background_calls_are_independent_and_cancellable(
    bash_executable: str, tmp_path: Path
) -> None:
    manager = ToolExecutionManager()
    for call_id in ("one", "two"):
        snapshot = start_bash(
            manager,
            bash_executable,
            tmp_path,
            "printf started; PATH=/usr/bin:$PATH sleep 20 & wait",
            call_id,
            mode="background",
        )
        assert snapshot.status == "running"
    try:
        deadline = time.monotonic() + 5
        while True:
            outputs = [
                manager.partial_output(f"output-{call_id}", 0)
                for call_id in ("one", "two")
            ]
            if all(output is not None and "started" in output[0] for output in outputs):
                break
            assert time.monotonic() < deadline
            time.sleep(0.02)
        for call_id in ("one", "two"):
            supported, cancelled, snapshot = manager.cancel(call_id)
            assert supported and cancelled
            assert snapshot.status == "cancelled"
            assert manager.wait(call_id, 1).status == "cancelled"
    finally:
        for call_id in ("one", "two"):
            if manager.wait(call_id, 0.01).status == "running":
                manager.cancel(call_id)


def test_bash_missing_executable_is_execution_failure(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    manager = ToolExecutionManager()
    snapshot = start_bash(manager, "", tmp_path, "echo never", "missing")
    assert snapshot.status == "failed"
    assert "Bash is unavailable" in snapshot.error
    assert "No other shell is used" in snapshot.error


def test_bash_process_start_failure_is_not_a_command_result(tmp_path: Path) -> None:
    manager = ToolExecutionManager()
    snapshot = start_bash(
        manager, str(tmp_path / "missing-bash"), tmp_path, "echo never", "startup"
    )
    assert snapshot.status == "failed"
    assert snapshot.error.startswith("FileNotFoundError:")


class ReplaceCommand(BeforeTool):
    def handle(self, context: ToolContext) -> ToolArguments:
        assert context.work_root.is_dir()
        return ToolArguments({"command": "printf filtered"})


class HideOutput(AfterTool):
    def handle(self, context: ToolContext, result):
        assert context.arguments["command"] == "printf filtered"
        return {"exit_code": result["exit_code"], "stdout": "redacted"}


class BlockCommand(BeforeTool):
    def handle(self, context: ToolContext) -> ToolResponse:
        return ToolResponse({"blocked": context.arguments["command"]})


def test_bash_hooks_modify_command_and_hide_raw_progress(
    bash_executable: str, tmp_path: Path
) -> None:
    tool = BashTool(
        executable=bash_executable, before=ReplaceCommand(), after=HideOutput()
    )
    manager = ToolExecutionManager()
    snapshot = start_bash(
        manager, bash_executable, tmp_path, "printf secret", "hooked", tool=tool
    )
    assert snapshot.status == "completed"
    assert snapshot.result == {"exit_code": 0, "stdout": "redacted"}
    assert manager.partial_output("output-hooked", 0) is None
    content = (
        ToolOutputReader(lambda: pytest.fail("No cache"), manager)
        .read("output-hooked")
        .content
    )
    assert "redacted" in content
    assert "filtered" not in content
    assert "secret" not in content.split("</header>", 1)[1]


def test_bash_before_hook_can_block_launch(tmp_path: Path) -> None:
    manager = ToolExecutionManager()
    tool = BashTool(executable="missing-bash", before=BlockCommand())
    snapshot = start_bash(
        manager, "missing-bash", tmp_path, "printf secret", "blocked", tool=tool
    )
    assert snapshot.status == "completed"
    assert snapshot.result == {"blocked": "printf secret"}
    assert manager.partial_output("output-blocked", 0) is None


def test_bash_result_stays_filtered_through_global_read_and_control_tools(
    bash_executable: str, tmp_path: Path
) -> None:
    node_holder: dict[str, Node] = {}

    def read_output(
        output_id: str, cursor: int | None = None
    ) -> PreformattedToolResult:
        return node_holder["node"].read_tool_output(output_id, cursor)

    def control_execution(
        tool_call_id: str,
        action: Literal["wait", "cancel"] = "wait",
        wait_seconds: float = 60,
    ) -> dict:
        return node_holder["node"].control_tool_execution(
            tool_call_id, action, wait_seconds
        )

    class FilterRead(AfterTool):
        def handle(self, context: ToolContext, result):
            assert isinstance(result, PreformattedToolResult)
            assert "redacted" in result.content
            return PreformattedToolResult("<filtered-read />")

    class FilterControl(AfterTool):
        def handle(self, context: ToolContext, result):
            return {"status": result["status"], "filtered": True}

    provider = SequenceProvider(
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "origin", FunctionCall("bash", {"command": "printf secret"})
                    )
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "read",
                        FunctionCall("read_tool_output", {"output_id": "output-1"}),
                    )
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "control",
                        FunctionCall(
                            "control_tool_execution", {"tool_call_id": "origin"}
                        ),
                    )
                ],
            ),
            AssistantMessage("done"),
        ]
    )
    counter = iter(range(1, 10))
    provider.generate_id = lambda: f"output-{next(counter)}"
    with SessionCache(tmp_path / "cache") as session:
        provider.get_session = lambda: session
        node = Node(
            [_write_nested_capabilities(tmp_path)],
            name_hds="main",
            default_capabilities=DefaultCapabilities(
                harness=False,
                include={"bash", "read_tool_output", "control_tool_execution"},
                overrides={
                    "bash": BashTool(
                        executable=bash_executable,
                        before=ReplaceCommand(),
                        after=HideOutput(),
                    ),
                    "read_tool_output": Tool(
                        read_output,
                        name="read_tool_output",
                        background_enabled=False,
                        after=FilterRead(),
                    ),
                    "control_tool_execution": Tool(
                        control_execution,
                        name="control_tool_execution",
                        background_enabled=False,
                        after=FilterControl(),
                    ),
                },
            ),
        )
        node_holder["node"] = node
        node.initialize(provider, work_root=tmp_path)
        assert node.run("go").get_message() == "done"
        messages = provider.requests[-1].get_messages()
        results = {
            message.get_name(): message.get_message()
            for message in messages
            if isinstance(message, ToolMessage)
        }
        assert "redacted" in results["bash"]
        assert "secret" not in results["bash"].split("</header>", 1)[1]
        assert results["read_tool_output"] == "<filtered-read />"
        assert (
            '<filtered type="boolean">true</filtered>'
            in results["control_tool_execution"]
        )
