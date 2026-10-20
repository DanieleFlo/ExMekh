"""Real DeepAgent execution plus adapters for existing decision checkpoints."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from evaluations.benchmark.accounting import MeteredProvider, RunBudgetExceeded
from evaluations.benchmark.compaction import configure_compaction
from evaluations.benchmark.models import Check, Observation, Task, Variant
from evaluations.benchmark.sandbox import workspace_path
from evaluations.tool_argument_grading import RawToolCall
from evaluations.workflow_cases import build_cases, grade_workflow
from evaluations.workflow_prompts import PromptCaptureProvider
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.deep_agent.stream_event import (
    DeepAgentCompactionProgressEvent,
    DeepAgentEvent,
    DeepAgentExecutionEvent,
    DeepAgentHistoryCompactionEvent,
)
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.deep_skill.parser import SkillMarkdownParser
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.request_budget import ContextBudgetExceededError
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.event_handler import DeepAgentEventHandler
from exmekh_lib.interfaces.message import Message
from exmekh_lib.interfaces.provider import ProviderResponseIncompleteError

type Driver = Callable[[Task, Variant, Path, MeteredProvider], Observation]


class DriverRegistry:
    def __init__(self) -> None:
        self._drivers: dict[str, Driver] = {}

    def register(self, name: str, driver: Driver) -> None:
        if name in self._drivers:
            raise ValueError(f"Driver already registered: {name}")
        self._drivers[name] = driver

    def validate(self, task: Task) -> None:
        if task.driver not in self._drivers:
            raise ValueError(f"Unknown driver: {task.driver}")

    def run(
        self, task: Task, variant: Variant, directory: Path, provider: MeteredProvider
    ) -> Observation:
        self.validate(task)
        return self._drivers[task.driver](task, variant, directory, provider)


class Observer(DeepAgentEventHandler):
    def __init__(self, provider: MeteredProvider) -> None:
        self.provider = provider
        self.events: list[dict[str, Any]] = []

    def handle(self, event: DeepAgentEvent) -> None:
        if isinstance(event, DeepAgentCompactionProgressEvent):
            if event.progress.finished:
                self.provider.end_compaction()
            else:
                self.provider.begin_compaction()
        if isinstance(event, DeepAgentHistoryCompactionEvent):
            self.events.append({"kind": "history_compaction", **asdict(event)})
        elif isinstance(event, DeepAgentExecutionEvent):
            self.events.append(
                {
                    "kind": event.type.value,
                    "name": event.tool_name,
                    "call_id": event.tool_call_id,
                    "scope": event.skill_path,
                    "depth": event.depth,
                    "failed": event.failed,
                }
            )


def prepare_actor(directory: Path, *, children: bool = False) -> Node:
    capabilities = directory.resolve() / "actor/capabilities"
    (capabilities / "hds").mkdir(parents=True)
    hds = """---
description: Complete bounded research and workspace tasks.
exit-tool: exit_agent
sub-skills:
  - deep-thinking
  - file-processing
---
Complete the user's task using the available scopes. Check actual results before
claiming completion. At the node's final return, provide the requested deliverable.
"""
    (capabilities / "hds/main.md").write_text(hds, encoding="utf-8")
    (capabilities / "hds/worker.md").write_text(
        hds + "\nReturn your completed work using exit_agent(text, items).",
        encoding="utf-8",
    )
    tool = capabilities / "tools/run-python"
    tool.mkdir(parents=True)
    (tool / "main.py").write_text(
        "from pathlib import Path\nfrom typing import Annotated, Any\n"
        "from exmekh_lib.core.agents.tools.decorator import deep_tool\n"
        "from evaluations.benchmark.sandbox import run_python\n"
        "@deep_tool(name='run-python', description='Run a workroot Python script with standard library only, no network and bounded resources; returns exit_code/stdout/stderr.', background_enabled=False)\n"
        "def run(work_root: Path, path: Annotated[str, 'Relative script path.']) -> dict[str, Any]:\n"
        "    return run_python(work_root, path)\n",
        encoding="utf-8",
    )
    skill = DeepSkill.from_markdown(
        CAPABILITY_FORMAT.bundled_root / "deep-skill/file-processing.md",
        SkillMarkdownParser(),
    )
    skill = DeepSkill(
        name=skill.name,
        markdown_path=skill.markdown_path,
        metadata=replace(
            skill.metadata, sub_tools=(*skill.metadata.sub_tools, "run-python")
        ),
        body=skill.body
        + "\nUse run-python to check generated scripts; errors are evidence for correction, not success.",
        references=skill.references,
    )
    selected = (
        "exit_agent",
        "deep-thinking",
        "file-processing",
        "calculate",
        "date-time",
        "inspect-file",
        "read-file",
        "new-file",
        "edit-file",
        "delete-file",
        "get_execution_history",
        "search_tool_outputs",
        "read_tool_output",
        "call_sub_agent",
    )
    defaults = DefaultCapabilities(
        harness=False, include=selected, overrides={"file-processing": skill}
    )
    child = (
        Node(
            [capabilities],
            name_hds="worker",
            default_capabilities=defaults,
            agent_context_id=uuid4().hex,
        )
        if children
        else None
    )
    return Node(
        [capabilities],
        name_hds="main",
        default_capabilities=defaults,
        children=[child] if child else (),
        agent_context_id=uuid4().hex,
    )


def run_native(
    task: Task, variant: Variant, directory: Path, provider: MeteredProvider
) -> Observation:
    work_root = directory / "work"
    work_root.mkdir()
    for path, text in task.files.items():
        destination = workspace_path(work_root, path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
    node = prepare_actor(directory, children=bool(task.parameters.get("children")))
    observer = Observer(provider)
    agent = DeepAgent(
        provider,
        node,
        work_root,
        on_stream=observer,
        parameter_descriptions_in_schema=variant.parameter_descriptions_in_schema,
    )
    configure_compaction(node, provider)
    status, answer, error = "completed", "", None
    try:
        answer = str(agent.run(task.prompt).get_message() or "")
    except (RunBudgetExceeded, ContextBudgetExceededError) as failure:
        status, error = "budget_exceeded", str(failure)
    except RuntimeError as failure:
        cause: BaseException | None = failure
        while cause is not None and not isinstance(
            cause,
            (
                RunBudgetExceeded,
                ContextBudgetExceededError,
                ProviderResponseIncompleteError,
            ),
        ):
            cause = cause.__cause__
        if isinstance(cause, ProviderResponseIncompleteError):
            status, error = "incomplete_output", str(cause)
        elif cause is not None:
            status, error = "budget_exceeded", str(cause)
        else:
            raise
    artifacts = {}
    for path in task.artifacts:
        destination = workspace_path(work_root, path)
        if destination.is_file():
            if destination.stat().st_size > 200_000:
                status, error = (
                    "artifact_limit",
                    f"Artifact exceeds 200000 bytes: {path}",
                )
            else:
                artifacts[path] = destination.read_text(encoding="utf-8")
    return Observation(
        status,
        answer,
        artifacts,
        provider.get_tool_calls(),
        tuple(observer.events),
        error,
    )


def run_checkpoint(
    task: Task, variant: Variant, directory: Path, provider: MeteredProvider
) -> Observation:
    case = next(
        (item for item in build_cases() if item.name == task.parameters["case"]), None
    )
    if case is None:
        raise ValueError("Unknown workflow checkpoint")
    work_root = directory / "work"
    work_root.mkdir()
    capture = PromptCaptureProvider()
    node = Node(
        name_hds="main",
        default_capabilities=True,
        agent_context_id=uuid4().hex,
        children=[
            Node(
                name_hds="main", default_capabilities=True, agent_context_id=uuid4().hex
            )
        ],
    )
    agent = DeepAgent(
        capture,
        node,
        work_root,
        parameter_descriptions_in_schema=variant.parameter_descriptions_in_schema,
    )
    agent.run("Capture without inference.")
    assert capture.request is not None
    data = agent.capabilities_by_node[node]
    captured_tools = capture.request.get_tools()
    assert captured_tools is not None
    tools = Tools(
        [
            *captured_tools.get_tools(),
            Tool(
                lambda: None,
                name="review-stage",
                description="Open the application's review scope, which exposes file-processing.",
                background_enabled=False,
            ),
        ]
    )
    exit_tool = data.get_exit_tool("exit_agent")
    skills = {
        skill.name: skill.render_instructions(exit_tool.render_prompt_section())
        for skill in data.deep_skills
    }
    messages: list[Message[object]] = [
        *capture.request.get_messages()[:1],
        UserMessage(task.prompt),
    ]
    for index, step in enumerate(case.steps):
        messages.append(
            AssistantMessage(
                "",
                tool_calls=[
                    ToolCall(f"step-{index}", FunctionCall(step.name, step.arguments))
                ],
            )
        )
        messages.append(
            ToolMessage(
                step.output if step.output is not None else skills[step.name],
                f"step-{index}",
                step.name,
            )
        )
    response = provider.call(ProviderRequest(messages, tools=tools, tool_choice="auto"))
    return Observation(
        "completed", str(response.get_message() or ""), calls=provider.get_tool_calls()
    )


def checkpoint_score(task, observation, workspace, parameters):
    case = next(item for item in build_cases() if item.name == task.parameters["case"])
    calls = tuple(
        RawToolCall(call["name"], json.dumps(call["arguments"]))
        for call in observation.calls
    )
    grade = grade_workflow(case, calls, observation.answer)
    return Check(grade.score, grade.checks, grade.issue)
