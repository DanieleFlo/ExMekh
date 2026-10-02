from pathlib import Path
from typing import Any

import pytest

from exmekh_lib.capabilities.tools.bash.bash_tool import BashTool
from exmekh_lib.capabilities.tools.call_sub_agent.sub_agent_tool import SubAgentTool
from exmekh_lib.core.agents.node import (
    DefaultCapabilities,
    Node,
    ToolHooks,
    ToolOverride,
)
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.interfaces.tool_hook import (
    AfterTool,
    BeforeTool,
    ToolArguments,
    ToolContext,
    ToolResponse,
)
from tests.core.agents.capabilities.test_default_capabilities import (
    SequenceProvider,
    replacement_tool,
    write_hds,
    write_tool,
)


class ChangeText(BeforeTool):
    def handle(self, context: ToolContext) -> ToolArguments:
        return ToolArguments({**context.arguments, "text": "changed"})


class AnswerEarly(BeforeTool):
    def handle(self, context: ToolContext) -> ToolResponse:
        return ToolResponse("shortcut")


class Suffix(AfterTool):
    def __init__(self, suffix: str = " hooked") -> None:
        self.suffix = suffix

    def handle(self, context: ToolContext, result: Any) -> str:
        return f"{result}{self.suffix}"


def tool_results(provider: SequenceProvider) -> list[str]:
    return [
        message.get_message()
        for message in provider.requests[-1].get_messages()
        if isinstance(message, ToolMessage)
    ]


@pytest.mark.parametrize(
    ("options", "error_type", "message"),
    [
        ({"tool_hooks": []}, TypeError, "tool_hooks must be a mapping"),
        ({"tool_hooks": {1: ToolHooks()}}, TypeError, "must contain only strings"),
        (
            {"tool_hooks": {"missing": ToolHooks()}},
            ValueError,
            "Unknown default capabilities: missing",
        ),
        (
            {"tool_hooks": {"main": ToolHooks()}},
            TypeError,
            "require a tool or exit tool",
        ),
        (
            {"tool_hooks": {"deep-thinking": ToolHooks()}},
            TypeError,
            "require a tool or exit tool",
        ),
        (
            {"tool_hooks": {"bash": object()}},
            TypeError,
            "must be a ToolHooks instance",
        ),
        (
            {"tool_hooks": {"bash": ToolHooks()}, "exclude": {"bash"}},
            ValueError,
            "Tool hooks require selected capabilities: bash",
        ),
        (
            {"tool_hooks": {"bash": ToolHooks()}, "harness": False},
            ValueError,
            "Tool hooks require selected capabilities: bash",
        ),
        (
            {"tool_hooks": {"mock-tool": ToolHooks()}},
            ValueError,
            "Tool hooks require selected capabilities: mock-tool",
        ),
    ],
)
def test_default_hook_configuration_rejects_invalid_targets(
    options, error_type, message
) -> None:
    with pytest.raises(error_type, match=message):
        DefaultCapabilities(**options)


def test_default_hook_mapping_is_copied_and_read_only() -> None:
    hooks = {"bash": ToolHooks(before=AnswerEarly())}
    defaults = DefaultCapabilities(harness=False, include={"bash"}, tool_hooks=hooks)

    hooks.clear()

    assert defaults.selected_names == {"bash"}
    assert set(defaults.tool_hooks) == {"bash"}
    assert defaults.tool_hooks_for("bash") is not None
    assert defaults.tool_hooks_for("search-web") is None
    assert defaults.customizes_tool("bash")
    assert not defaults.customizes_tool("search-web")
    assert DefaultCapabilities().tool_hooks == {}
    with pytest.raises(TypeError):
        defaults.tool_hooks["bash"] = ToolHooks()


def test_bundled_hooks_run_and_only_the_final_result_is_cached(tmp_path: Path) -> None:
    provider = SequenceProvider(
        tmp_path / "cache",
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall("work", FunctionCall("mock-tool", {"text": "original"}))
                ],
            ),
            AssistantMessage("done"),
        ],
    )
    provider.generate_id = lambda: "saved-output"
    node = Node(
        name_hds="main",
        default_capabilities=DefaultCapabilities(
            mocks=True,
            tool_hooks={"mock-tool": ToolHooks(before=ChangeText(), after=Suffix())},
        ),
    )
    node.initialize(provider, work_root=tmp_path)

    assert node.run("run tool").get_message() == "done"
    content = node.read_tool_output("saved-output").content
    assert (
        content.split("</header>", 1)[-1].strip() == "Mock tool result: changed hooked"
    )


@pytest.mark.parametrize(
    ("hooks", "expected"),
    [
        (ToolHooks(after=Suffix()), "replacement-tool: original hooked"),
        (ToolHooks(after=None), "replacement-tool: original"),
        (ToolHooks(before=ChangeText()), "replacement-tool: changed source"),
    ],
)
def test_hooks_apply_after_named_overrides_without_changing_the_source(
    tmp_path: Path, hooks: ToolHooks, expected: str
) -> None:
    path = write_hds(tmp_path, "application", tools=("mock-tool",))
    write_tool(path, "replacement-tool")
    entrypoint = path / "tools" / "replacement-tool-directory" / "main.py"
    entrypoint.write_text(
        "from exmekh_lib.core.agents.tools.decorator import deep_tool\n"
        "from exmekh_lib.interfaces.tool_hook import AfterTool\n"
        "class OriginalAfter(AfterTool):\n"
        "    def handle(self, context, result):\n"
        "        return result + ' source'\n"
        "@deep_tool(name='replacement-tool', description='Source', after=OriginalAfter())\n"
        "def run(work_root, text: str) -> str:\n"
        "    return 'replacement-tool: ' + text\n"
    )
    defaults = DefaultCapabilities(
        overrides=ToolOverride("mock-tool", "replacement-tool"),
        tool_hooks={"mock-tool": hooks},
    )
    node = Node([path], name_hds="application", default_capabilities=defaults)
    loaded = node.load_capabilities(work_root=tmp_path)
    tools = {tool.get_name(): tool for tool in loaded.tools}

    assert (
        tools["mock-tool"].execute_with_hooks(
            {"text": "original"}, work_root=tmp_path, history=()
        )
        == expected
    )
    assert (
        tools["replacement-tool"].execute_with_hooks(
            {"text": "original"}, work_root=tmp_path, history=()
        )
        == "replacement-tool: original source"
    )


def test_hooks_do_not_mutate_object_overrides_shared_by_nodes(tmp_path: Path) -> None:
    original = Tool(replacement_tool, name="mock-tool", after=Suffix(" original"))
    nodes = [
        Node(
            name_hds="main",
            default_capabilities=DefaultCapabilities(
                mocks=True,
                overrides={"mock-tool": original},
                tool_hooks=hooks,
            ),
        )
        for hooks in ({"mock-tool": ToolHooks(after=Suffix())}, {})
    ]
    results = []
    for node in nodes:
        loaded = node.load_capabilities(work_root=tmp_path)
        tool = next(tool for tool in loaded.tools if tool.get_name() == "mock-tool")
        results.append(
            tool.execute_with_hooks({"text": "input"}, work_root=tmp_path, history=())
        )

    assert results == ["Replacement: input hooked", "Replacement: input original"]
    assert (
        original.execute_with_hooks({"text": "input"}, work_root=tmp_path, history=())
        == "Replacement: input original"
    )


@pytest.mark.parametrize(
    "name",
    [
        "bash",
        "get_execution_history",
        "search_tool_outputs",
        "read_tool_output",
        "control_tool_execution",
    ],
)
def test_runtime_tools_execute_configured_hooks(tmp_path: Path, name: str) -> None:
    path = write_hds(tmp_path, "application")
    provider = SequenceProvider(
        tmp_path / "cache",
        [
            AssistantMessage(
                None, tool_calls=[ToolCall("work", FunctionCall(name, {}))]
            ),
            AssistantMessage("done"),
        ],
    )
    node = Node(
        [path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(
            tool_hooks={name: ToolHooks(before=AnswerEarly(), after=Suffix())}
        ),
    )
    node.initialize(provider, work_root=tmp_path)
    assert node.run("run tool").get_message() == "done"
    tool = provider.requests[0].get_tools().get_tool(name)

    assert tool is not None
    assert tool.background_enabled is (name == "bash")
    if name == "bash":
        assert isinstance(tool, BashTool)
    results = tool_results(provider)
    assert len(results) == 1
    assert "shortcut hooked" in results[0]


def test_runtime_hooks_apply_after_named_overrides(tmp_path: Path) -> None:
    path = write_hds(tmp_path, "application")
    write_tool(path, "replacement-tool")
    provider = SequenceProvider(tmp_path / "cache", [AssistantMessage("done")])
    node = Node(
        [path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(
            overrides=ToolOverride("get_execution_history", "replacement-tool"),
            tool_hooks={"get_execution_history": ToolHooks(after=Suffix())},
        ),
    )
    node.initialize(provider, work_root=tmp_path)
    node.run("show tools")
    tool = provider.requests[0].get_tools().get_tool("get_execution_history")

    assert tool is not None
    assert tool.background_enabled is False
    assert (
        tool.execute_with_hooks({"text": "input"}, work_root=tmp_path, history=())
        == "replacement-tool: input hooked"
    )


def test_runtime_uses_overrides_with_false_truth_values(tmp_path: Path) -> None:
    class FalseyTool(Tool):
        def __bool__(self) -> bool:
            return False

    path = write_hds(tmp_path, "application")
    provider = SequenceProvider(
        tmp_path / "cache",
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "history",
                        FunctionCall("get_execution_history", {"text": "input"}),
                    )
                ],
            ),
            AssistantMessage("done"),
        ],
    )
    node = Node(
        [path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(
            overrides={
                "get_execution_history": FalseyTool(
                    replacement_tool, name="get_execution_history"
                )
            },
            tool_hooks={"get_execution_history": ToolHooks(after=Suffix())},
        ),
    )
    node.initialize(provider, work_root=tmp_path)

    assert node.run("read history").get_message() == "done"
    results = tool_results(provider)
    assert len(results) == 1
    assert "Replacement: input hooked" in results[0]


def test_exit_hooks_preserve_the_exit_contract(tmp_path: Path) -> None:
    class ChangeExit(AfterTool):
        def handle(self, context: ToolContext, result: Any) -> dict[str, Any]:
            return {**result, "text": "hooked exit"}

    loaded = Node(
        name_hds="main",
        default_capabilities=DefaultCapabilities(
            mocks=True,
            tool_hooks={"exit_agent": ToolHooks(after=ChangeExit())},
        ),
    ).load_capabilities(work_root=tmp_path)
    exit_tool = loaded.get_exit_tool("exit_agent")

    assert exit_tool.execute_with_hooks(
        {"text": "original", "items": []}, work_root=tmp_path, history=()
    ) == {"text": "hooked exit", "items": []}
    assert exit_tool.render_prompt_section()


def test_sub_agent_hooks_preserve_delegation_and_description_updates(
    tmp_path: Path,
) -> None:
    path = write_hds(tmp_path, "parent")
    write_hds(tmp_path, "child")
    child = Node([path], name_hds="child", default_capabilities=DefaultCapabilities())
    child.initialize(
        SequenceProvider(tmp_path / "child-cache", [AssistantMessage("child result")]),
        work_root=tmp_path,
    )
    parent = Node(
        [path],
        name_hds="parent",
        children=[child],
        default_capabilities=DefaultCapabilities(
            tool_hooks={"call_sub_agent": ToolHooks(after=Suffix())}
        ),
    )
    provider = SequenceProvider(
        tmp_path / "parent-cache",
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        "delegate",
                        FunctionCall(
                            "call_sub_agent", {"sub_agent": "child", "text": "hello"}
                        ),
                    )
                ],
            ),
            AssistantMessage("done"),
        ],
    )
    parent.initialize(provider, work_root=tmp_path)
    parent.update_sub_agent_tool_description()
    assert parent.run("delegate").get_message() == "done"
    tool = provider.requests[0].get_tools().get_tool("call_sub_agent")

    assert isinstance(tool, SubAgentTool)
    assert "<description>child</description>" in tool.get_description()
    assert tuple(tool.get_signature().parameters) == ("sub_agent", "text")
    results = tool_results(provider)
    assert len(results) == 1
    assert "child result hooked" in results[0]


def test_sub_agent_override_hooks_preserve_synchronous_execution(
    tmp_path: Path,
) -> None:
    path = write_hds(tmp_path, "parent")
    write_hds(tmp_path, "child")
    original = Tool(replacement_tool, name="call_sub_agent")
    provider = SequenceProvider(tmp_path / "cache", [AssistantMessage("done")])
    parent = Node(
        [path],
        name_hds="parent",
        children=[Node([path], name_hds="child")],
        default_capabilities=DefaultCapabilities(
            overrides={"call_sub_agent": original},
            tool_hooks={"call_sub_agent": ToolHooks(after=Suffix())},
        ),
    )
    parent.initialize(provider, work_root=tmp_path)
    parent.run("show tools")
    tool = provider.requests[0].get_tools().get_tool("call_sub_agent")

    assert tool is not None
    assert tool.background_enabled is False
    assert original.after is None
    assert original.background_enabled is True
    assert (
        tool.execute_with_hooks({"text": "input"}, work_root=tmp_path, history=())
        == "Replacement: input hooked"
    )


def test_sub_agent_hooks_require_children(tmp_path: Path) -> None:
    path = write_hds(tmp_path, "application")
    node = Node(
        [path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(
            tool_hooks={"call_sub_agent": ToolHooks()}
        ),
    )

    with pytest.raises(ValueError, match="call_sub_agent requires at least one child"):
        node.initialize(SequenceProvider(tmp_path / "cache", []), work_root=tmp_path)


def test_excluded_sub_agent_tools_do_not_validate_unused_children(
    tmp_path: Path,
) -> None:
    path = write_hds(tmp_path, "parent")
    write_hds(tmp_path, "child")
    child = Node([path], name_hds="child")
    provider = SequenceProvider(tmp_path / "cache", [AssistantMessage("done")])
    parent = Node(
        [path],
        name_hds="parent",
        children=[child, child],
        default_capabilities=DefaultCapabilities(exclude={"call_sub_agent"}),
    )

    parent.initialize(provider, work_root=tmp_path)

    assert parent.run("run").get_message() == "done"
    assert "call_sub_agent" not in provider.requests[0].get_tools().list_tools()
