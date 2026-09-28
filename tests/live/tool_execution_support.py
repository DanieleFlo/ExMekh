"""Real subprocess fixtures and observation of unmodified live model calls."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from threading import Condition, Event, Lock, Timer
from time import monotonic
from uuid import uuid4
from xml.etree import ElementTree as ET

from openai import OpenAI

from qs.config.cache import CACHE_FORMAT
from qs.core.agents.capabilities.defaults import DefaultCapabilities
from qs.core.agents.deep_agent.stream_event import (
    DeepAgentEvent,
    DeepAgentExecutionEvent,
)
from qs.core.agents.node import Node
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.assistant_message import AssistantMessage, ToolCall
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.reasoning_effort import ReasoningEffortLMH
from qs.core.models.tool import Tool, ToolExecutionOperation
from qs.core.models.tool_message import ToolMessage
from qs.core.models.xml_header import XmlHeader
from qs.providers.local import LocalProvider
from tests.event_handler import as_agent_handler
from tests.live.test_tool_cache_live import CacheTestTokenizer

# A stdin gate models externally completed work without relying on inference latency.
_WORKER = """\
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

mode, label = sys.argv[1:]
payload = Path('fixture-input.txt').read_bytes()
print(json.dumps({'phase': 'started', 'label': label, 'raw': payload.decode()}), flush=True)
if mode == 'gated':
    for command in sys.stdin:
        if command.strip() == 'progress':
            print(json.dumps({'phase': 'advanced', 'label': label}), flush=True)
        elif command.strip() == 'finish':
            break
receipt = {'receipt': str(uuid.uuid4()), 'label': label, 'pid': os.getpid(),
           'cwd': str(Path.cwd()), 'sha256': hashlib.sha256(payload).hexdigest(),
           'raw': payload.decode()}
print(json.dumps({'phase': 'final', **receipt}), flush=True)
sys.exit(7 if mode == 'negative' else 0)
"""


def split_output(content: str) -> tuple[dict[str, str], str]:
    closing = f"</{XmlHeader.ROOT_TAG}>"
    prefix, separator, body = content.partition(closing)
    assert separator, f"CLIENT: missing XML header in {content!r}"
    try:
        root = ET.fromstring(prefix + closing)
    except ET.ParseError as error:
        raise AssertionError("CLIENT: malformed XML header") from error
    return {child.tag: child.text or "" for child in root}, body.strip()


class LiveProcess:
    def __init__(self, directory: Path, call_id: str, label: str, mode: str) -> None:
        self.call_id = call_id
        self.label = label
        self.started = Event()
        self.advanced = Event()
        self.reader_done = Event()
        self.receipt: dict = {}
        self._output = ""
        self._output_lock = Lock()
        self._input_lock = Lock()
        self._finish_sent = False
        # Windows virtualenv launchers spawn another process; terminate the worker itself.
        executable = sys._base_executable if os.name == "nt" else sys.executable
        self.process = subprocess.Popen(
            [executable, "-u", str(directory / "fixture-worker.py"), mode, label],
            cwd=directory,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    def run(self) -> dict:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                value = json.loads(line)
                if value["phase"] == "final":
                    self.receipt = {
                        key: item for key, item in value.items() if key != "phase"
                    }
                else:
                    with self._output_lock:
                        self._output += line
                    if value["phase"] == "started":
                        self.started.set()
                    if value["phase"] == "advanced":
                        self.advanced.set()
            return {**self.receipt, "exit_code": self.process.wait(timeout=5)}
        finally:
            self.reader_done.set()

    def read_partial(self, cursor: int) -> tuple[str, int]:
        with self._output_lock:
            return self._output[cursor:], len(self._output)

    def send(self, command: str) -> None:
        assert self.started.wait(5), "PROCESS: child did not start"
        with self._input_lock:
            if self._finish_sent or self.process.poll() is not None:
                return
            assert self.process.stdin is not None
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            self._finish_sent = command == "finish"

    def cancel(self) -> bool:
        if self.process.poll() is not None:
            return False
        self.process.terminate()
        self.process.wait(timeout=5)
        assert self.reader_done.wait(5), "PROCESS: cancelled child reader did not exit"
        return True

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=5)
            assert self.reader_done.wait(5), (
                "PROCESS: child reader leaked during cleanup"
            )
        finally:
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe is not None:
                    pipe.close()


class ProcessFixture:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.raw = f"fixture-private-{uuid4()}"
        self.sha256 = hashlib.sha256(self.raw.encode()).hexdigest()
        (directory / "fixture-input.txt").write_text(self.raw, encoding="utf-8")
        (directory / "fixture-worker.py").write_text(_WORKER, encoding="utf-8")
        self.jobs: dict[str, LiveProcess] = {}
        self._created = Condition()
        self._closed = False

    def launch(self, call_id: str, label: str, mode: str) -> LiveProcess:
        with self._created:
            if self._closed:
                raise RuntimeError("PROCESS: fixture is already closed")
            assert call_id not in self.jobs, "CLIENT: process was launched twice"
            job = LiveProcess(self.directory, call_id, label, mode)
            self.jobs[call_id] = job
            self._created.notify_all()
        return job

    def job(self, call_id: str) -> LiveProcess:
        with self._created:
            assert self._created.wait_for(lambda: call_id in self.jobs, timeout=5), (
                f"PROCESS: no process for {call_id!r}"
            )
            job = self.jobs[call_id]
        assert job.started.wait(5), "PROCESS: child did not produce initial output"
        return job

    def close(self) -> None:
        with self._created:
            self._closed = True
            jobs = tuple(self.jobs.values())
        with ExitStack() as cleanup:
            for job in jobs:
                cleanup.callback(job.close)


class ProcessTool(Tool):
    def __init__(self, fixture: ProcessFixture, *, name="live_process_job", **kwargs):
        self.fixture = fixture
        super().__init__(
            self.parameters,
            name=name,
            description=(
                "Launch one real file-processing job. Its receipt is generated only "
                "when it finishes. Keep tool_call_id for control_tool_execution and "
                "the XML output ID for read_tool_output. Do not relaunch pending jobs."
            ),
            **kwargs,
        )

    @staticmethod
    def parameters(label: str) -> dict:
        raise AssertionError("PROCESS: use the execution operation")

    def create_execution(self, arguments, *, call_id, work_root, history):
        assert work_root == self.fixture.directory
        job = self.fixture.launch(call_id, arguments["label"], self.process_mode())
        return ToolExecutionOperation(job.run, job.cancel, job.read_partial)

    def process_mode(self) -> str:
        return "gated"


class QuickProcessTool(ProcessTool):
    def process_mode(self) -> str:
        return "quick"


class NegativeProcessTool(ProcessTool):
    def process_mode(self) -> str:
        return "negative"


class FailedProcessTool(NegativeProcessTool):
    def create_execution(self, arguments, **context):
        operation = super().create_execution(arguments, **context)

        def fail():
            result = operation.run()
            raise RuntimeError(
                f"Fixture process failed with exit code {result['exit_code']}"
            )

        return ToolExecutionOperation(fail, operation.cancel, operation.read_partial)


class UncancellableProcessTool(ProcessTool):
    def create_execution(self, arguments, **context):
        operation = super().create_execution(arguments, **context)
        operation.cancel = None
        return operation


class RecordingExecutionProvider(LocalProvider):
    def __init__(self, directory: Path, *, streaming: bool = True):
        self.responses: list[ProviderResponse] = []
        self.requests: list[ProviderRequest] = []
        self.streaming = streaming
        super().__init__(
            model=os.getenv("QS_LIVE_MODEL", "Qwen3.8-27B-UD-Q4_K_S"),
            reasoning_effort=ReasoningEffortLMH.LOW,
            api="responses",
            tokenizer=CacheTestTokenizer("live-tool-executions", directory),
        )

    def _build_client(self) -> OpenAI:
        configured_timeout = os.getenv("QS_LIVE_REQUEST_TIMEOUT_SECONDS")
        if configured_timeout is None:
            return super()._build_client().with_options(max_retries=0)
        message = (
            "QS_LIVE_REQUEST_TIMEOUT_SECONDS must be a finite number of seconds "
            "greater than zero"
        )
        try:
            timeout = float(configured_timeout)
        except ValueError as error:
            raise ValueError(message) from error
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(message)
        return super()._build_client().with_options(timeout=timeout, max_retries=0)

    def call(self, request: ProviderRequest) -> ProviderResponse:
        assert len(self.responses) < 20, (
            f"MODEL: exceeded live turn budget; calls={[call.get_name() for call in self.calls()]}"
        )
        self.verify_protocol(request)
        # Keep runtime events for coordination while selecting real provider transport.
        request.stream = self.streaming
        self.requests.append(request)
        response = super().call(request)
        self.responses.append(response)
        return response

    @staticmethod
    def verify_protocol(request: ProviderRequest) -> None:
        pending: set[str] = set()
        seen: set[str] = set()
        for message in request.get_messages():
            if isinstance(message, AssistantMessage):
                for call in message.get_tool_calls() or ():
                    assert call.id not in seen, "CLIENT: reused call ID in live history"
                    seen.add(call.id)
                    pending.add(call.id)
            elif isinstance(message, ToolMessage):
                call_id = message.get_tool_call_id()
                assert call_id in pending, "CLIENT: orphan or duplicate tool response"
                pending.remove(call_id)
        assert not pending, f"CLIENT: unanswered calls before inference: {pending}"

    def calls(self, name: str | None = None) -> list[ToolCall]:
        return [
            call
            for response in self.responses
            for call in response.get_tool_calls() or ()
            if name is None or call.get_name() == name
        ]

    def history_calls(self, name: str) -> list[ToolCall]:
        calls = {}
        for request in self.requests:
            for message in request.get_messages():
                if isinstance(message, AssistantMessage):
                    for call in message.get_tool_calls() or ():
                        if call.get_name() == name:
                            calls[call.id] = call
        return list(calls.values())

    def synthetic_reads(self) -> list[ToolCall]:
        model_ids = {call.id for call in self.calls()}
        return [
            call
            for call in self.history_calls("read_tool_output")
            if call.id not in model_ids
        ]

    def output(self, call_id: str) -> str:
        outputs = {
            message.get_message()
            for request in self.requests
            for message in request.get_messages()
            if isinstance(message, ToolMessage)
            and message.get_tool_call_id() == call_id
        }
        assert len(outputs) == 1, (
            f"CLIENT: missing or inconsistent output for {call_id}"
        )
        return outputs.pop()


def write_capabilities(directory: Path, tool_names: list[str]) -> Path:
    root = directory / "capabilities"
    hds = root / "hds"
    skills = root / "deep-skill"
    hds.mkdir(parents=True)
    skills.mkdir()
    for name in tool_names:
        tool_path = root / "tools" / name
        tool_path.mkdir(parents=True)
        (tool_path / "main.py").write_text(
            "from qs.core.agents.tools.decorator import deep_tool\n\n"
            f"@deep_tool(name={name!r}, description='Live fixture overridden during initialization')\n"
            "def run(work_root, label: str):\n"
            "    raise AssertionError('Live fixture override was not installed')\n",
            encoding="utf-8",
        )
    tools = "".join(f"  - {name}\n" for name in tool_names)
    tool_section = f"sub-tools:\n{tools}" if tool_names else "sub-tools: []\n"
    instructions = (
        "Follow the user's ordered test steps precisely. Use real returned IDs. "
        "Execution controls and output reads do not launch a new job. "
        "A running response is not completion. Never guess a receipt. "
        "Inside a skill return with exit_agent(text=..., items=[])."
    )
    (hds / "main.md").write_text(
        "---\ndescription: Live execution test\nexit-tool: exit_agent\n"
        f"sub-skills:\n  - work-scope\n{tool_section}---\n{instructions}",
        encoding="utf-8",
    )
    (skills / "work-scope.md").write_text(
        "---\ndescription: Process owner scope\nsub-skills:\n  - inner-scope\n"
        f"{tool_section}---\n{instructions}",
        encoding="utf-8",
    )
    (skills / "inner-scope.md").write_text(
        f"---\ndescription: Nested scope\n---\n{instructions}", encoding="utf-8"
    )
    return root


class LiveExecutionRig:
    def __init__(self, provider: RecordingExecutionProvider, fixture: ProcessFixture):
        self.provider = provider
        self.fixture = fixture
        self.events: list[DeepAgentExecutionEvent] = []
        self.outputs: dict[str, str] = {}
        self.durations: dict[str, float] = {}
        self.observer: Callable[[DeepAgentExecutionEvent], None] | None = None
        self.node: Node | None = None
        self._starts: dict[str, float] = {}
        self._timers: list[Timer] = []
        self._timer_errors: list[Exception] = []

    def initialize(self, tools: list[Tool], *, children=()) -> Node:
        path = write_capabilities(self.fixture.directory, [tool.name for tool in tools])
        defaults = DefaultCapabilities(
            harness=False,
            include={
                "exit_agent",
                "get_execution_history",
                "search_tool_outputs",
                "read_tool_output",
                "control_tool_execution",
                "call_sub_agent",
            },
            overrides={tool.name: tool for tool in tools},
        )
        self.node = Node(
            [path],
            name_hds="main",
            children=children,
            agent_context_id=f"live-executions-{uuid4().hex}",
            default_capabilities=defaults,
        )
        self.node.initialize(
            self.provider,
            work_root=self.fixture.directory,
            on_stream=as_agent_handler(self.handle),
        )
        return self.node

    def handle(self, event: DeepAgentEvent) -> None:
        if not isinstance(event, DeepAgentExecutionEvent):
            return
        self.events.append(event)
        if event.is_tool_running:
            self._starts[event.tool_call_id] = monotonic()
            calls = [
                call for call in self.provider.calls() if call.id == event.tool_call_id
            ]
            if event.tool_name == "control_tool_execution" and calls:
                arguments = calls[0].get_arguments()
                if (
                    arguments.get("action", "wait") == "wait"
                    and arguments["wait_seconds"] >= 1
                ):
                    job = self.fixture.job(arguments["tool_call_id"])
                    self.schedule_finish(job)
        else:
            self.durations[event.tool_call_id] = (
                monotonic() - self._starts[event.tool_call_id]
            )
            if event.history and event.history[-1]["role"] == "tool":
                self.outputs[event.tool_call_id] = event.history[-1]["content"]
        if self.observer is not None:
            self.observer(event)

    def schedule_finish(self, job: LiveProcess) -> None:
        def finish():
            try:
                job.send("finish")
            except Exception as error:
                self._timer_errors.append(error)

        timer = Timer(0.25, finish)
        timer.daemon = True
        self._timers.append(timer)
        timer.start()

    def finish_all(self) -> None:
        assert self.node is not None
        for job in self.fixture.jobs.values():
            job.send("finish")
            status = self.node.control_tool_execution(job.call_id, wait_seconds=5)
            assert status["status"] != "running", "PROCESS: child did not complete"

    def one_call(self, name: str) -> ToolCall:
        calls = self.provider.calls(name)
        assert len(calls) == 1, f"MODEL: expected one {name}, got {len(calls)}"
        return calls[0]

    def saved_outputs(self) -> list[ToolCallRecord]:
        directory = (
            self.provider.get_session().session_dir / CACHE_FORMAT.tool_calls_folder
        )
        return [
            ToolCallRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in directory.rglob("*.json")
        ]

    def saved_output(self, output_id: str) -> ToolCallRecord | None:
        return next(
            (record for record in self.saved_outputs() if record.id == output_id), None
        )

    def result(self, call: ToolCall) -> tuple[dict[str, str], dict]:
        header, body = split_output(
            self.outputs.get(call.id) or self.provider.output(call.id)
        )
        return header, json.loads(body)

    def assert_completed_job(self, call: ToolCall) -> LiveProcess:
        job = self.fixture.job(call.id)
        assert job.receipt, "PROCESS: no real receipt generated"
        delivered = [
            message.get_message()
            for request in self.provider.requests
            for message in request.get_messages()
            if isinstance(message, ToolMessage)
        ]
        assert any(job.receipt["receipt"] in content for content in delivered), (
            "CLIENT: real process result never reached the model"
        )
        assert job.receipt["sha256"] == self.fixture.sha256
        assert Path(job.receipt["cwd"]) == self.fixture.directory
        assert job.receipt["pid"] == job.process.pid
        assert job.process.poll() == 0, "PROCESS: child still alive or failed"
        return job

    def save_trace(self) -> None:
        (self.fixture.directory / "live-trace.json").write_text(
            json.dumps(
                {
                    "model_calls": [
                        {
                            "id": call.id,
                            "name": call.get_name(),
                            "args": call.get_arguments(),
                        }
                        for call in self.provider.calls()
                    ],
                    "events": [
                        {
                            "type": event.type.value,
                            "skill_path": event.skill_path,
                            "name": event.tool_name,
                            "id": event.tool_call_id,
                            "duration": self.durations.get(event.tool_call_id),
                            "output": self.outputs.get(event.tool_call_id),
                        }
                        for event in self.events
                        if not event.is_tool_running
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def close(self) -> None:
        with ExitStack() as cleanup:
            cleanup.callback(self.fixture.close)
            for timer in self._timers:
                cleanup.callback(timer.join, timeout=5)
                timer.cancel()
            self.save_trace()
        assert not self._timer_errors, (
            f"PROCESS: coordinator failed: {self._timer_errors}"
        )
