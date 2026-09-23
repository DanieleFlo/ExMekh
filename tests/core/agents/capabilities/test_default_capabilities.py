from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

import qs.core.agents.capabilities.capabilities as capability_module
from qs.core.agents.capabilities.defaults import (
    BUILTIN_CAPABILITIES_PATH,
    HARNESS_NAMES,
    MOCK_NAMES,
    DefaultCapabilities,
)
from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.deep_skill.metadata import DeepSkillMetadata
from qs.core.agents.exit_tool.data import ExitToolData
from qs.core.agents.exit_tool.exit_tool import ExitTool
from qs.core.agents.hds.hds import HDS
from qs.core.agents.hds.metadata import HDSMetadata
from qs.core.agents.node import DeepSkillOverride, Node, ToolOverride
from qs.core.cache.cache_manager import CacheManager
from qs.core.cache.session_cache import SessionCache
from qs.core.models.assistant_message import AssistantMessage, FunctionCall, ToolCall
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.tool import Tool
from qs.interfaces.provider import Provider
from qs.interfaces.tokenizer import Tokenizer


class StubTokenizer(Tokenizer):
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def initialize(self) -> None:
        pass

    def generate_id(self) -> str:
        return "default-capability-id"

    def get_session(self) -> SessionCache:
        return CacheManager.get_session(
            "default-capability-tests", "default", self.directory
        )


class SequenceProvider(Provider):
    def __init__(self, directory: Path, responses: list[AssistantMessage]) -> None:
        super().__init__(StubTokenizer(directory))
        self.responses = responses
        self.requests: list[ProviderRequest] = []

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(self.responses.pop(0))

    def call_structured[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        del request
        raise NotImplementedError


def write_hds(
    root: Path,
    name: str,
    *,
    body: str = "Custom instructions",
    tools: tuple[str, ...] = (),
    skills: tuple[str, ...] = (),
) -> Path:
    path = root / "capabilities"
    hds_dir = path / "hds"
    hds_dir.mkdir(parents=True, exist_ok=True)
    tool_references = (
        "sub-tools:\n" + "".join(f"  - {tool}\n" for tool in tools) if tools else ""
    )
    skill_references = (
        "sub-skills:\n" + "".join(f"  - {skill}\n" for skill in skills)
        if skills
        else ""
    )
    (hds_dir / f"{name}.md").write_text(
        f"---\ndescription: {name}\nexit-tool: exit_agent\n"
        f"{skill_references}{tool_references}---\n{body}"
    )
    return path


def write_tool(path: Path, name: str) -> None:
    tool_path = path / "tools" / f"{name}-directory"
    tool_path.mkdir(parents=True)
    (tool_path / "main.py").write_text(
        "from qs.core.agents.tools.decorator import deep_tool\n"
        f"@deep_tool(name='{name}', description='{name} application tool')\n"
        "def run(work_root, text: str) -> str:\n"
        f"    return '{name}: ' + text\n"
    )


def write_skill(path: Path, name: str) -> None:
    skill_path = path / "deep-skill" / f"{name}.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(f"---\ndescription: {name} description\n---\n{name} body")


def make_hds(
    name: str, *, skills: tuple[str, ...] = (), tools: tuple[str, ...] = ()
) -> HDS:
    return HDS(
        name=name,
        markdown_path=Path(f"{name}.md"),
        metadata=HDSMetadata(
            description="Replacement HDS",
            sub_skills=skills,
            sub_tools=tools,
            exit_tool="exit_agent",
        ),
        body="Replacement instructions",
        references=(),
    )


def make_skill() -> DeepSkill:
    return DeepSkill(
        name="mock-skill",
        markdown_path=Path("replacement-skill.md"),
        metadata=DeepSkillMetadata(
            description="Replacement skill",
            inherits_parent_tools=False,
            sub_skills=(),
            sub_tools=(),
        ),
        body="Replacement skill instructions",
        references=(),
    )


def replacement_tool(text: str) -> str:
    return f"Replacement: {text}"


def replacement_exit(text: str, items: list[str]) -> ExitToolData:
    return ExitToolData(text=f"Replacement exit: {text}", items=items)


def make_exit() -> ExitTool:
    return ExitTool(
        replacement_exit,
        name="exit_agent",
        description="Replacement exit",
        instructions_path=Path("replacement-exit.md"),
        instructions="Use the replacement exit.",
        folder_path=Path("replacement-exit"),
    )


def test_group_presets_and_individual_selection() -> None:
    assert DefaultCapabilities().selected_names == HARNESS_NAMES
    assert DefaultCapabilities(mocks=True, harness=False).selected_names == MOCK_NAMES
    assert DefaultCapabilities(mocks=True).selected_names == MOCK_NAMES | HARNESS_NAMES
    selected = DefaultCapabilities(
        mocks=True,
        harness=True,
        include={"mock-tool"},
        exclude={"mock-skill", "read_saved_tool_output"},
    )
    assert selected.includes("mock-tool")
    assert selected.mocks and selected.harness
    assert selected.include == {"mock-tool"}
    assert selected.exclude == {"mock-skill", "read_saved_tool_output"}
    assert not selected.includes("mock-skill")
    assert not selected.includes("read_saved_tool_output")
    assert DefaultCapabilities(
        harness=False, include={"read_saved_tool_output"}
    ).selected_names == {"read_saved_tool_output"}


def test_selection_rejects_unknown_conflicting_and_invalid_overrides() -> None:
    with pytest.raises(TypeError, match="mocks and harness must be bool"):
        DefaultCapabilities(mocks=cast(bool, "yes"))
    with pytest.raises(TypeError, match="mocks and harness must be bool"):
        DefaultCapabilities(harness=cast(bool, 1))
    with pytest.raises(TypeError, match="include must be an iterable"):
        DefaultCapabilities(include=cast(set[str], "mock-tool"))
    with pytest.raises(TypeError, match="include must be an iterable"):
        DefaultCapabilities(include=cast(set[str], 1))
    with pytest.raises(TypeError, match="exclude must contain only strings"):
        DefaultCapabilities(exclude=cast(set[str], {1}))
    with pytest.raises(TypeError, match="overrides must be a mapping or iterable"):
        DefaultCapabilities(overrides=cast(dict[str, Tool], 1))
    with pytest.raises(ValueError, match="Unknown default capabilities: unknown"):
        DefaultCapabilities(include={"unknown"})
    with pytest.raises(ValueError, match="included or overridden and excluded"):
        DefaultCapabilities(include={"mock-tool"}, exclude={"mock-tool"})
    with pytest.raises(ValueError, match="included or overridden and excluded"):
        DefaultCapabilities(
            exclude={"mock-tool"},
            overrides={"mock-tool": Tool(replacement_tool, name="mock-tool")},
        )
    with pytest.raises(
        TypeError, match="Override for 'mock-skill' must be a DeepSkill"
    ):
        DefaultCapabilities(overrides={"mock-skill": Tool(replacement_tool)})
    with pytest.raises(ValueError, match="must have name 'mock-tool'"):
        DefaultCapabilities(overrides={"mock-tool": Tool(replacement_tool)})
    with pytest.raises(TypeError, match="Override for 'mock-tool' must be a Tool"):
        DefaultCapabilities(overrides={"mock-tool": make_hds("mock-tool")})
    with pytest.raises(TypeError, match="Override for 'mock-tool' must be a Tool"):
        DefaultCapabilities(overrides={"mock-tool": "replacement-tool"})
    with pytest.raises(TypeError, match="overrides must contain only"):
        DefaultCapabilities(overrides=cast(list[ToolOverride], [object()]))
    with pytest.raises(ValueError, match="Duplicate override for 'mock-tool'"):
        DefaultCapabilities(
            overrides=[
                ToolOverride("mock-tool", "first"),
                ToolOverride("mock-tool", "second"),
            ]
        )
    with pytest.raises(ValueError, match="must target 'mock-tool'"):
        DefaultCapabilities(
            overrides={"mock-tool": ToolOverride("different", "replacement-tool")}
        )
    with pytest.raises(TypeError, match="must match DeepSkill"):
        DefaultCapabilities(overrides=ToolOverride("mock-skill", "replacement-tool"))
    with pytest.raises(TypeError, match="must match Tool"):
        DefaultCapabilities(overrides=DeepSkillOverride("mock-tool", "other-skill"))
    with pytest.raises(TypeError, match="target_name must be a string"):
        ToolOverride(cast(str, 1), "replacement-tool")
    with pytest.raises(TypeError, match="replacement_name must be a string"):
        ToolOverride("mock-tool", cast(str, 1))
    with pytest.raises(ValueError, match="target_name cannot be empty"):
        ToolOverride("", "replacement-tool")
    with pytest.raises(ValueError, match="replacement_name cannot be empty"):
        DeepSkillOverride("mock-skill", "")
    with pytest.raises(TypeError, match="not a resolved Tool"):
        DefaultCapabilities(
            overrides=ToolOverride("mock-tool", "replacement-tool")
        ).tool_override("mock-tool")
    with pytest.raises(TypeError, match="default_capabilities must be"):
        Node(name_hds="main", default_capabilities=cast(DefaultCapabilities, object()))
    with pytest.raises(TypeError, match="capabilities_paths must be a sequence"):
        Node(
            capabilities_paths=cast(list[Path], "bad"),
            name_hds="main",
            default_capabilities=DefaultCapabilities(),
        )
    with pytest.raises(TypeError, match="capabilities_paths must be a sequence"):
        Node(
            capabilities_paths=cast(list[Path], 3),
            name_hds="main",
            default_capabilities=DefaultCapabilities(),
        )
    with pytest.raises(TypeError, match="capabilities_paths must contain only Path"):
        Node(
            capabilities_paths=[cast(Path, "bad")],
            name_hds="main",
            default_capabilities=DefaultCapabilities(),
        )


def test_harness_defaults_support_an_application_hds_without_mocks(
    tmp_path: Path,
) -> None:
    path = write_hds(tmp_path, "application")
    provider = SequenceProvider(tmp_path / "cache", [AssistantMessage("done")])
    node = Node(
        [path], name_hds="application", default_capabilities=DefaultCapabilities()
    )
    loaded = node.initialize(provider, work_root=tmp_path)

    assert {resource.name for resource in loaded.hds} == {"application"}
    assert {resource.name for resource in loaded.exit_tools} == {"exit_agent"}
    assert loaded.deep_skills == loaded.tools == ()
    assert node.run("hello").get_message() == "done"
    request = provider.requests[0]
    assert set(request.get_tools().list_tools()) == HARNESS_NAMES - {"call_sub_agent"}
    assert request.get_extra()["agentic"]["allowed_tools"] == [
        "get_execution_history",
        "search_saved_tool_outputs",
        "read_saved_tool_output",
    ]


def test_mock_preset_works_without_application_paths_and_deduplicates_builtin_path(
    tmp_path: Path,
) -> None:
    defaults = DefaultCapabilities(mocks=True)
    first = Node(name_hds="main", default_capabilities=defaults)
    second = Node(
        [BUILTIN_CAPABILITIES_PATH], name_hds="main", default_capabilities=defaults
    )

    for node in (first, second):
        loaded = node.load_capabilities(work_root=tmp_path)
        assert {resource.name for resource in loaded.hds} == {"main", "mock-sub-agent"}
        assert {resource.name for resource in loaded.deep_skills} == {
            "mock-skill",
            "nested-skill",
        }
        assert len(loaded.tools) == 4


def test_excluded_references_fail_before_catalog_attachment(tmp_path: Path) -> None:
    node = Node(
        name_hds="main",
        default_capabilities=DefaultCapabilities(include={"main"}),
    )
    with pytest.raises(ValueError, match="HDS 'main' has invalid references"):
        node.load_capabilities(work_root=tmp_path)


def test_overrides_enable_and_replace_mock_resources_per_node(tmp_path: Path) -> None:
    skill = make_skill()
    defaults = DefaultCapabilities(
        overrides={
            "main": make_hds("main", skills=("mock-skill",), tools=("mock-tool",)),
            "mock-skill": skill,
            "mock-tool": Tool(replacement_tool, name="mock-tool"),
            "exit_agent": make_exit(),
        }
    )
    provider = SequenceProvider(
        tmp_path / "cache",
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        id="replacement-call",
                        function=FunctionCall("mock-tool", {"text": "hello"}),
                    )
                ],
            ),
            AssistantMessage("done"),
        ],
    )
    node = Node(name_hds="main", default_capabilities=defaults)
    loaded = node.initialize(provider, work_root=tmp_path)
    other = Node(name_hds="main", default_capabilities=defaults)
    other_loaded = other.load_capabilities(work_root=tmp_path)

    assert loaded.get_hds("main").get_hds() == "Replacement instructions"
    assert (
        loaded.get_exit_tool("exit_agent")
        .render_prompt_section()
        .endswith("Use the replacement exit.</exit-tool>")
    )
    assert (
        loaded.deep_skills[0]
        .render_instructions("exit")
        .startswith("Replacement skill instructions")
    )
    assert loaded.deep_skills[0] is not other_loaded.deep_skills[0]
    assert skill.catalog is None
    assert provider.requests == []
    response = node.run("hello")
    assert response.get_message() == "done"
    assert provider.requests[0].get_messages()[0].get_message() == (
        "Replacement instructions"
    )
    assert "Replacement: hello" in str(response.history)
    assert {tool.name for tool in provider.requests[0].get_tools().get_tools()} >= {
        "mock-tool",
        "mock-skill",
    }
    assert provider.requests[0].get_extra()["agentic"]["allowed_skills"] == [
        "mock-skill"
    ]


def test_replacement_exit_runs_when_a_skill_closes(tmp_path: Path) -> None:
    defaults = DefaultCapabilities(
        overrides={
            "main": make_hds("main", skills=("mock-skill",)),
            "mock-skill": make_skill(),
            "exit_agent": make_exit(),
        }
    )
    provider = SequenceProvider(
        tmp_path / "cache",
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(id="open", function=FunctionCall("mock-skill", {}))
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        id="close",
                        function=FunctionCall(
                            "exit_agent", {"text": "done", "items": ["done"]}
                        ),
                    )
                ],
            ),
            AssistantMessage("complete"),
        ],
    )
    node = Node(name_hds="main", default_capabilities=defaults)
    node.initialize(provider, work_root=tmp_path)
    response = node.run("run the skill")

    assert response.get_message() == "complete"
    assert "Replacement exit: done" in str(response.history)
    assert "Replacement skill instructions" in str(
        [message.get_message() for message in provider.requests[1].get_messages()]
    )
    assert "Use the replacement exit." in str(
        [message.get_message() for message in provider.requests[1].get_messages()]
    )


def test_harness_runtime_tools_can_be_selected_and_overridden(tmp_path: Path) -> None:
    path = write_hds(tmp_path, "application")

    def custom_history() -> str:
        return "custom history"

    defaults = DefaultCapabilities(
        harness=False,
        include={"exit_agent", "get_execution_history"},
        overrides={
            "get_execution_history": Tool(
                custom_history, name="get_execution_history", cache_result=False
            )
        },
    )
    provider = SequenceProvider(
        tmp_path / "cache",
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        id="custom-history-call",
                        function=FunctionCall("get_execution_history", {}),
                    )
                ],
            ),
            AssistantMessage("done"),
        ],
    )
    node = Node([path], name_hds="application", default_capabilities=defaults)
    node.initialize(provider, work_root=tmp_path)
    response = node.run("show history")

    first = provider.requests[0]
    assert set(first.get_tools().list_tools()) == {
        "exit_agent",
        "get_execution_history",
    }
    assert first.get_extra()["agentic"]["allowed_tools"] == ["get_execution_history"]
    assert "custom history" in str(response.history)


def test_sub_agent_selection_and_override_are_node_local(tmp_path: Path) -> None:
    path = write_hds(tmp_path, "parent")
    write_hds(tmp_path, "child")

    def custom_delegate(text: str) -> str:
        return f"Custom delegate: {text}"

    child = Node([path], name_hds="child", default_capabilities=DefaultCapabilities())
    disabled = Node(
        [path],
        name_hds="parent",
        children=[child],
        default_capabilities=DefaultCapabilities(exclude={"call_sub_agent"}),
    )
    disabled.initialize(SequenceProvider(tmp_path / "cache1", []), work_root=tmp_path)
    assert "call_sub_agent" not in disabled._runtime._client.tools.list_tools()

    standard = Node(
        [path],
        name_hds="parent",
        children=[child],
        default_capabilities=DefaultCapabilities(),
    )
    standard.initialize(
        SequenceProvider(tmp_path / "cache-default", []), work_root=tmp_path
    )
    assert "call_sub_agent" in standard._runtime._client.tools.list_tools()

    enabled = Node(
        [path],
        name_hds="parent",
        children=[child],
        default_capabilities=DefaultCapabilities(
            overrides={"call_sub_agent": Tool(custom_delegate, name="call_sub_agent")}
        ),
    )
    enabled.initialize(SequenceProvider(tmp_path / "cache2", []), work_root=tmp_path)
    tool = enabled._runtime._client.tools.get_tool("call_sub_agent")
    assert tool.execute(text="hello") == "Custom delegate: hello"
    assert list(tool.get_signature().parameters) == ["text"]

    leaf = Node(
        [path],
        name_hds="child",
        default_capabilities=DefaultCapabilities(
            overrides={"call_sub_agent": Tool(custom_delegate, name="call_sub_agent")}
        ),
    )
    with pytest.raises(ValueError, match="call_sub_agent requires at least one child"):
        leaf.initialize(SequenceProvider(tmp_path / "cache3", []), work_root=tmp_path)


def test_excluding_memory_readers_preserves_internal_cache(tmp_path: Path) -> None:
    defaults = DefaultCapabilities(
        exclude={
            "get_execution_history",
            "search_saved_tool_outputs",
            "read_saved_tool_output",
        },
        overrides={
            "main": make_hds("main", tools=("mock-tool",)),
            "mock-tool": Tool(replacement_tool, name="mock-tool"),
        },
    )
    provider = SequenceProvider(
        tmp_path / "cache",
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        id="work",
                        function=FunctionCall("mock-tool", {"text": "cached"}),
                    )
                ],
            ),
            AssistantMessage("done"),
        ],
    )
    node = Node(name_hds="main", default_capabilities=defaults)
    node.initialize(provider, work_root=tmp_path)
    node.run("do work")

    assert set(provider.requests[0].get_tools().list_tools()) == {
        "mock-tool",
        "exit_agent",
    }
    assert [record.result for record in node._runtime._tool_cache.get_all()] == [
        "Replacement: cached"
    ]
    assert [entry.name for entry in node._runtime._execution_history.entries] == [
        "mock-tool"
    ]


def test_application_name_collision_requires_explicit_override(tmp_path: Path) -> None:
    path = write_hds(tmp_path, "main")
    node = Node(
        [path], name_hds="main", default_capabilities=DefaultCapabilities(mocks=True)
    )
    with pytest.raises(ValueError, match="Duplicate HDS name: 'main'"):
        node.load_capabilities(work_root=tmp_path)


def test_tool_override_replaces_same_named_application_and_bundled_tools(
    tmp_path: Path,
) -> None:
    path = write_hds(tmp_path, "application")
    tool_path = path / "tools" / "mock-tool"
    tool_path.mkdir(parents=True)
    (tool_path / "main.py").write_text(
        "from qs.core.agents.tools.decorator import deep_tool\n"
        "@deep_tool(name='mock-tool', description='Application tool')\n"
        "def run(work_root, text: str) -> str:\n"
        "    return 'application: ' + text\n"
    )
    without_override = Node(
        [path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(include={"mock-tool"}),
    )
    with pytest.raises(ValueError, match="Duplicate tool name: 'mock-tool'"):
        without_override.load_capabilities(work_root=tmp_path)

    with_override = Node(
        [path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(
            overrides={"mock-tool": Tool(replacement_tool, name="mock-tool")}
        ),
    )
    loaded = with_override.load_capabilities(work_root=tmp_path)
    assert [tool.execute(text="ok") for tool in loaded.tools] == ["Replacement: ok"]


def test_application_tool_override_uses_names_from_every_capabilities_path(
    tmp_path: Path,
) -> None:
    first_path = write_hds(
        tmp_path / "first",
        "application",
        tools=("first-tool", "second-tool"),
    )
    write_tool(first_path, "first-tool")
    second_path = tmp_path / "second" / "capabilities"
    second_path.mkdir(parents=True)
    write_tool(second_path, "second-tool")
    provider = SequenceProvider(
        tmp_path / "cache",
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        id="replacement-call",
                        function=FunctionCall("first-tool", {"text": "hello"}),
                    )
                ],
            ),
            AssistantMessage("done"),
        ],
    )
    defaults = DefaultCapabilities(
        overrides=[ToolOverride("first-tool", "second-tool")]
    )
    node = Node(
        [first_path, second_path],
        name_hds="application",
        default_capabilities=defaults,
    )
    loaded = node.initialize(provider, work_root=tmp_path)

    tools = {tool.name: tool for tool in loaded.tools}
    assert set(tools) == {"first-tool", "second-tool"}
    assert tools["first-tool"].execute(text="ok") == "second-tool: ok"
    assert tools["second-tool"].execute(text="ok") == "second-tool: ok"
    response = node.run("hello")
    assert response.get_message() == "done"
    request = provider.requests[0]
    assert {"first-tool", "second-tool"} <= set(request.get_tools().list_tools())
    exposed_tool = request.get_tools().get_tool("first-tool")
    assert exposed_tool is not None
    assert exposed_tool.get_description() == "second-tool application tool"
    assert {"first-tool", "second-tool"} <= set(
        request.get_extra()["agentic"]["allowed_tools"]
    )
    assert "second-tool: hello" in str(response.history)


def test_named_tool_override_replaces_a_bundled_tool(tmp_path: Path) -> None:
    path = write_hds(tmp_path, "application", tools=("mock-tool",))
    write_tool(path, "replacement-tool")
    node = Node(
        [path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(
            overrides={
                "mock-tool": ToolOverride("mock-tool", "replacement-tool"),
                "exit_agent": make_exit(),
            }
        ),
    )

    loaded = node.load_capabilities(work_root=tmp_path)

    assert {tool.name for tool in loaded.tools} == {"mock-tool", "replacement-tool"}
    assert (
        next(tool for tool in loaded.tools if tool.name == "mock-tool").execute(
            text="ok"
        )
        == "replacement-tool: ok"
    )


def test_named_deep_skill_override_replaces_a_bundled_skill(tmp_path: Path) -> None:
    path = write_hds(tmp_path, "application", skills=("mock-skill",))
    write_skill(path, "replacement-skill")
    node = Node(
        [path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(
            overrides=DeepSkillOverride("mock-skill", "replacement-skill")
        ),
    )

    loaded = node.load_capabilities(work_root=tmp_path)

    assert next(
        skill for skill in loaded.deep_skills if skill.name == "mock-skill"
    ).body == ("replacement-skill body")


def test_named_tool_override_replaces_a_harness_runtime_tool(tmp_path: Path) -> None:
    path = write_hds(tmp_path, "application")
    write_tool(path, "replacement-tool")
    provider = SequenceProvider(
        tmp_path / "cache",
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        id="history",
                        function=FunctionCall(
                            "get_execution_history", {"text": "hello"}
                        ),
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
            overrides=ToolOverride("get_execution_history", "replacement-tool")
        ),
    )

    node.initialize(provider, work_root=tmp_path)
    response = node.run("show history")

    assert "get_execution_history" in provider.requests[0].get_tools().list_tools()
    assert "replacement-tool: hello" in str(response.history)


def test_named_deep_skill_override_uses_loaded_skill_and_executes(
    tmp_path: Path,
) -> None:
    first_path = write_hds(tmp_path / "first", "application", skills=("target-skill",))
    write_skill(first_path, "target-skill")
    second_path = tmp_path / "second" / "capabilities"
    second_path.mkdir(parents=True)
    write_skill(second_path, "replacement-skill")
    provider = SequenceProvider(
        tmp_path / "cache",
        [
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(id="open", function=FunctionCall("target-skill", {}))
                ],
            ),
            AssistantMessage(
                None,
                tool_calls=[
                    ToolCall(
                        id="close",
                        function=FunctionCall(
                            "exit_agent", {"text": "done", "items": ["done"]}
                        ),
                    )
                ],
            ),
            AssistantMessage("complete"),
        ],
    )
    defaults = DefaultCapabilities(
        overrides=[DeepSkillOverride("target-skill", "replacement-skill")]
    )
    node = Node(
        [first_path, second_path],
        name_hds="application",
        default_capabilities=defaults,
    )
    loaded = node.initialize(provider, work_root=tmp_path)
    other = Node(
        [first_path, second_path],
        name_hds="application",
        default_capabilities=defaults,
    )
    other_loaded = other.load_capabilities(work_root=tmp_path)

    skills = {skill.name: skill for skill in loaded.deep_skills}
    assert set(skills) == {"target-skill", "replacement-skill"}
    assert skills["target-skill"].body == "replacement-skill body"
    assert skills["target-skill"] is not skills["replacement-skill"]
    assert skills["target-skill"] is not next(
        skill for skill in other_loaded.deep_skills if skill.name == "target-skill"
    )
    assert node.run("open skill").get_message() == "complete"
    assert provider.requests[0].get_extra()["agentic"]["allowed_skills"] == [
        "target-skill"
    ]
    assert "replacement-skill body" in str(
        [message.get_message() for message in provider.requests[1].get_messages()]
    )


def test_named_overrides_reject_missing_sources(tmp_path: Path) -> None:
    path = write_hds(tmp_path, "application")
    write_tool(path, "target-tool")
    write_skill(path, "target-skill")

    for override, expected_error in (
        (
            ToolOverride("target-tool", "missing-tool"),
            "Override tool not found in capabilities_paths: missing-tool",
        ),
        (
            DeepSkillOverride("target-skill", "missing-skill"),
            "Override deep skill not found in capabilities_paths: missing-skill",
        ),
    ):
        node = Node(
            [path],
            name_hds="application",
            default_capabilities=DefaultCapabilities(overrides=[override]),
        )
        with pytest.raises(ValueError, match=expected_error):
            node.load_capabilities(work_root=tmp_path)


def test_named_tool_override_rejects_ambiguous_source(tmp_path: Path) -> None:
    first_path = write_hds(tmp_path / "first", "application")
    write_tool(first_path, "replacement-tool")
    second_path = tmp_path / "second" / "capabilities"
    second_path.mkdir(parents=True)
    write_tool(second_path, "replacement-tool")
    node = Node(
        [first_path, second_path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(
            overrides=ToolOverride("mock-tool", "replacement-tool")
        ),
    )

    with pytest.raises(
        ValueError, match="Override tool name is ambiguous: replacement-tool"
    ):
        node.load_capabilities(work_root=tmp_path)


def test_named_deep_skill_override_requires_a_loaded_target(tmp_path: Path) -> None:
    path = write_hds(tmp_path, "application")
    write_skill(path, "replacement-skill")
    node = Node(
        [path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(
            overrides=[DeepSkillOverride("missing-skill", "replacement-skill")]
        ),
    )

    with pytest.raises(
        ValueError,
        match="Override deep skills not found in capabilities_paths: missing-skill",
    ):
        node.load_capabilities(work_root=tmp_path)


@pytest.mark.parametrize("name", ["missing-tool", "application"])
def test_application_tool_override_requires_a_loaded_tool(
    tmp_path: Path, name: str
) -> None:
    path = write_hds(tmp_path, "application")
    node = Node(
        [path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(
            overrides={name: Tool(replacement_tool, name=name)}
        ),
    )

    with pytest.raises(
        ValueError, match=f"Override tools not found in capabilities_paths: {name}"
    ):
        node.load_capabilities(work_root=tmp_path)


def test_missing_bundled_resource_is_reported(tmp_path: Path, monkeypatch) -> None:
    empty_bundle = tmp_path / "empty" / "capabilities"
    empty_bundle.mkdir(parents=True)
    monkeypatch.setattr(capability_module, "BUILTIN_CAPABILITIES_PATH", empty_bundle)

    node = Node(name_hds="main", default_capabilities=DefaultCapabilities())
    with pytest.raises(
        ValueError, match="Selected default capabilities are missing: exit_agent"
    ):
        node.load_capabilities(work_root=tmp_path)


def test_legacy_node_still_uses_explicit_builtin_path(tmp_path: Path) -> None:
    loaded = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main").load_capabilities(
        work_root=tmp_path
    )
    assert {resource.name for resource in loaded.hds} == {"main", "mock-sub-agent"}
