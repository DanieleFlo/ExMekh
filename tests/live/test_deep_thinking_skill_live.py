"""Opt-in end-to-end deep-thinking check with the real model and public web."""

import inspect
import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
from dotenv import load_dotenv

from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.capabilities.defaults import (
    BUILTIN_CAPABILITIES_PATH,
    DefaultCapabilities,
)
from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.tool_hooks import ToolHooks
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.tool_hook import AfterTool
from tests.live.model_support import LiveModelProvider, fresh_live_context_id

pytestmark = pytest.mark.live


class DeepThinkingLiveProvider(LiveModelProvider):
    """Bound the macro workflow and preserve actual calls for diagnosis."""

    def __init__(
        self,
        model: str,
        directory: Path,
        *,
        max_turns: int = 24,
        steps=None,
        tokenizer=None,
    ):
        self._trace_path = directory / "macro-model-trace.json"
        self._turns: list[list[dict[str, object]]] = []
        self._max_turns = max_turns
        self._steps = deepcopy(steps)
        super().__init__(
            model=model,
            reasoning_effort=ReasoningEffortLMH.LOW,
            api="responses",
            tokenizer=tokenizer,
        )

    def call(self, request: ProviderRequest) -> ProviderResponse:
        if len(self._turns) >= self._max_turns:
            raise AssertionError(
                f"FLOW: exceeded {self._max_turns} macro turns; last calls={self._turns[-5:]}"
            )
        if self._steps is not None:
            request = self._step_request(request)
        result = super().call(request)
        if self._steps is not None and len(self._turns) < len(self._steps):
            name = self._steps[len(self._turns)][0]
            calls = result.get_tool_calls() or []
            assert len(calls) == 1 and calls[0].get_name() == name, (
                f"COVERAGE: expected one real {name} call, got {calls!r}"
            )
        self._turns.append(
            [
                {"name": call.get_name(), "arguments": deepcopy(call.get_arguments())}
                for call in result.get_tool_calls() or []
            ]
        )
        self._trace_path.write_text(
            json.dumps(self._turns, ensure_ascii=True, indent=2)
        )
        return result

    def _step_request(self, request):
        messages = list(request.get_messages())
        extra = deepcopy(request.get_extra() or {})
        scope = extra["agentic"]
        tools = request.get_tools()
        names = set()
        choice = "none"
        if len(self._turns) < len(self._steps):
            name, arguments = self._steps[len(self._turns)]
            selected = tools.get_tool(name) if tools is not None else None
            assert selected is not None, (
                f"COVERAGE: {name} is absent from the active scope"
            )
            assert name in scope["allowed_tools"] + scope["allowed_skills"], (
                f"COVERAGE: {name} is not allowed in the active scope"
            )
            defaults = {
                name: parameter.default
                for name, parameter in selected.get_signature().parameters.items()
                if parameter.default is not inspect.Parameter.empty
            }
            arguments = {**defaults, **arguments}
            names = {name}
            messages.append(
                UserMessage(
                    f"Next, call {name} once with this argument object: "
                    + json.dumps(arguments)
                    + ". Preserve the supplied optional defaults as well as required fields."
                )
            )
            choice = "required"
        for key in ("allowed_tools", "allowed_skills"):
            scope[key] = [name for name in scope[key] if name in names]
        extra["max_output_tokens"] = 768
        bounded = ProviderRequest(
            messages,
            tools=tools,
            extra=extra,
            tool_choice=choice,
            stream=request.get_streaming(),
            on_event=request.on_event,
            reasoning_effort_override=request.get_reasoning_effort_override(),
        )
        bounded.interactive_handler = request.interactive_handler
        return bounded


class ObserveMacroResult(AfterTool):
    def __init__(self, name, results):
        self.name = name
        self.results = results

    def handle(self, context, result):
        self.results.setdefault(self.name, []).append(
            (deepcopy(context.arguments), deepcopy(result))
        )
        return result


def macro_steps():
    def comparison(operator, value):
        return {"op": operator, "left": {"var": "m"}, "right": {"const": value}}

    statistics = "https://docs.python.org/3/library/statistics.html"
    maths = "https://docs.python.org/3/library/math.html"
    variables = {"m": "Real"}
    premise = [comparison("=", 5)]
    return [
        ("deep-thinking", {}),
        (
            "search-web",
            {"query": "Python statistics mean documentation", "max_results": 3},
        ),
        ("read-web-page", {"url": statistics, "output_mode": "markdown"}),
        (
            "interact-web-page",
            {
                "url": statistics,
                "actions": [{"kind": "navigate", "url": maths}],
                "output_mode": "markdown",
            },
        ),
        ("calculate", {"operation": "evaluate", "expression": "(3+5+7)/3"}),
        ("logic-consistency", {"variables": variables, "premises": premise}),
        (
            "logic-check",
            {
                "variables": variables,
                "premises": premise,
                "conclusion": comparison(">", 4),
            },
        ),
        (
            "logic-counterexample",
            {
                "variables": variables,
                "premises": [comparison(">", 0)],
                "conclusion": comparison(">", 10),
            },
        ),
        (
            "exit_agent",
            {"text": "Mean 5; sources: " + statistics + " and " + maths, "items": []},
        ),
    ]


def test_live_model_can_choose_and_combine_every_skill_tool(tmp_path: Path) -> None:
    load_dotenv()
    model = os.getenv("EXMEKH_LIB_LIVE_DEEP_THINKING_MODEL")
    if not model:
        pytest.skip(
            "set EXMEKH_LIB_LIVE_DEEP_THINKING_MODEL for the model-driven macro test"
        )
    provider = DeepThinkingLiveProvider(model, tmp_path, steps=macro_steps())
    results = {}
    loaded = Capabilities([BUILTIN_CAPABILITIES_PATH]).load(work_root=tmp_path)
    work_names = {name for name, _ in macro_steps()} - {"deep-thinking", "exit_agent"}
    overrides = {
        tool.get_name(): tool.without_background_execution().with_hooks(
            ToolHooks(after=ObserveMacroResult(tool.get_name(), results))
        )
        for tool in loaded.tools
        if tool.get_name() in work_names
    }
    node = Node(
        [BUILTIN_CAPABILITIES_PATH],
        name_hds="main",
        agent_context_id=fresh_live_context_id("live-deep-thinking"),
        default_capabilities=DefaultCapabilities(include={"main"}, overrides=overrides),
    )
    try:
        response = DeepAgent(provider=provider, node=node, work_root=tmp_path).run(
            "Use deep-thinking to complete this audit. Search for the official Python "
            "statistics.mean documentation, read that page, navigate on the same "
            "docs.python.org site to the math module page with interact-web-page, "
            "calculate the mean of 3, 5, 7 with calculate. Use logic-consistency "
            "to check that m = 5 is consistent, logic-check to prove m > 4 from "
            "m = 5, and logic-counterexample to refute m > 10 from m > 0. Use all "
            "seven tools and then report only the mean and source URLs."
        )
    finally:
        provider.get_session().close()
        provider.client.close()
    names = {
        call["function"]["name"]
        for message in response.history
        for call in json.loads(message.get("tool_calls", "null")) or []
    }
    names.update(
        call.get_name()
        for execution in response.skill_executions
        for message in execution.history
        if isinstance(message, AssistantMessage)
        for call in message.get_tool_calls() or []
    )
    assert {
        "deep-thinking",
        "search-web",
        "read-web-page",
        "interact-web-page",
        "calculate",
        "logic-check",
        "logic-consistency",
        "logic-counterexample",
    } <= names
    assert set(results) == work_names, "COVERAGE: a real work tool was not executed"
    for name, outcomes in results.items():
        assert len(outcomes) == 1, (name, outcomes)
        assert outcomes[0][1].get("ok", True) is not False, (name, outcomes)
    values = {name: outcomes[0][1] for name, outcomes in results.items()}
    assert float(values["calculate"]["result"]) == 5
    assert values["logic-consistency"]["status"] == "consistent"
    assert values["logic-check"]["status"] == "valid"
    assert values["logic-counterexample"]["status"] == "found"
    assert values["read-web-page"]["status_code"] == 200
    assert values["interact-web-page"]["actions_completed"] is True
    assert (
        values["interact-web-page"]["final_url"]
        == "https://docs.python.org/3/library/math.html"
    )
