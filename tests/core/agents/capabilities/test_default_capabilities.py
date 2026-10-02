from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.capabilities.defaults import (
    BUILTIN_CAPABILITIES_PATH,
    HARNESS_NAMES,
    MOCK_NAMES,
    RUNTIME_NAMES,
    DefaultCapabilities,
)
from exmekh_lib.core.agents.capabilities.layout import CapabilitiesLayout
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.deep_skill.metadata import DeepSkillMetadata
from exmekh_lib.core.agents.exit_tool.data import ExitToolData
from exmekh_lib.core.agents.exit_tool.exit_tool import ExitTool
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.agents.hds.metadata import HDSMetadata
from exmekh_lib.core.agents.node import DeepSkillOverride, Node, ToolOverride
from exmekh_lib.core.cache.cache_manager import CacheManager
from exmekh_lib.core.cache.session_cache import SessionCache
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.interfaces.provider import Provider
from exmekh_lib.interfaces.tokenizer import Tokenizer

DEEP_THINKING_TOOLS = {
    "calculate",
    "date-time",
    "interact-web-page",
    "read-web-page",
    "search-web",
    "logic-check",
    "logic-consistency",
    "logic-counterexample",
}


def test_runtime_tool_sources_use_the_bundled_tools_directory(tmp_path: Path) -> None:
    bundled_tools = BUILTIN_CAPABILITIES_PATH / "tools"
    assert all((bundled_tools / name / "main.py").is_file() for name in RUNTIME_NAMES)

    application_root = tmp_path / "capabilities"
    application_bash = application_root / "tools" / "bash"
    application_bash.mkdir(parents=True)
    layout = CapabilitiesLayout(
        [application_root], default_path=BUILTIN_CAPABILITIES_PATH
    )
    layout.prepare()
    paths = layout.tool_paths()
    assert application_bash in paths
    assert all(bundled_tools / name not in paths for name in RUNTIME_NAMES)


class StubTokenizer(Tokenizer):
    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

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
        "from exmekh_lib.core.agents.tools.decorator import deep_tool\n"
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
    assert not RUNTIME_NAMES.intersection(DEEP_THINKING_TOOLS | {"deep-thinking"})
    assert DefaultCapabilities(mocks=True, harness=False).selected_names == (
        MOCK_NAMES | DEEP_THINKING_TOOLS | {"deep-thinking", "exit_agent"}
    )
    assert DefaultCapabilities(mocks=True).selected_names == MOCK_NAMES | HARNESS_NAMES
    selected = DefaultCapabilities(
        mocks=True,
        harness=True,
        include={"mock-tool"},
        exclude={"mock-skill", "read_tool_output"},
    )
    assert selected.includes("mock-tool")
    assert selected.mocks and selected.harness
    assert selected.include == {"mock-tool"}
    assert selected.exclude == {"mock-skill", "read_tool_output"}
    assert not selected.includes("mock-skill")
    assert not selected.includes("read_tool_output")
    assert DefaultCapabilities(
        harness=False, include={"read_tool_output"}
    ).selected_names == {"read_tool_output"}


def test_mock_main_loads_its_required_skill_without_full_harness(
    tmp_path: Path,
) -> None:
    node = Node(
        name_hds="main",
        default_capabilities=DefaultCapabilities(mocks=True, harness=False),
    )
    loaded = node.load_capabilities(work_root=tmp_path)
    assert {skill.name for skill in loaded.deep_skills} >= {"deep-thinking"}
    assert {tool.name for tool in loaded.tools} >= DEEP_THINKING_TOOLS


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
    assert {skill.name for skill in loaded.deep_skills} == {"deep-thinking"}
    assert {tool.name for tool in loaded.tools} == DEEP_THINKING_TOOLS
    assert node.run("hello").get_message() == "done"
    request = provider.requests[0]
    assert set(request.get_tools().list_tools()) == HARNESS_NAMES - {"call_sub_agent"}
    assert request.get_extra()["agentic"]["allowed_tools"] == [
        "get_execution_history",
        "search_tool_outputs",
        "read_tool_output",
        "control_tool_execution",
        "bash",
    ]


def test_mock_preset_works_without_application_paths_and_deduplicates_builtin_path(
    tmp_path: Path,
) -> None:
    defaults = DefaultCapabilities(mocks=True)
    assert defaults.capabilities_path == BUILTIN_CAPABILITIES_PATH
    first = Node(
        capabilities_paths=None, name_hds="main", default_capabilities=defaults
    )
    second = Node(
        [BUILTIN_CAPABILITIES_PATH], name_hds="main", default_capabilities=defaults
    )

    for node in (first, second):
        loaded = node.load_capabilities(work_root=tmp_path)
        assert {resource.name for resource in loaded.hds} == {"main", "mock-sub-agent"}
        assert {resource.name for resource in loaded.deep_skills} == {
            "deep-thinking",
            "mock-skill",
            "nested-skill",
        }
        assert len(loaded.tools) == 12


@pytest.mark.parametrize("paths", [None, []])
def test_node_uses_bundled_path_when_no_paths_are_supplied(
    tmp_path: Path, paths: list[Path] | None
) -> None:
    node = Node(capabilities_paths=paths, name_hds="main")

    loaded = node.load_capabilities(work_root=tmp_path)

    assert {resource.name for resource in loaded.hds} == {"main", "mock-sub-agent"}
    assert {tool.name for tool in loaded.tools} == {
        "mock-tool",
        "mock-sub-tool",
        "nested-sub-tool",
        "shared-tool",
    } | DEEP_THINKING_TOOLS


def test_direct_capabilities_loader_adds_bundled_path_once(tmp_path: Path) -> None:
    defaults = DefaultCapabilities(mocks=True)

    for paths in ([], [defaults.capabilities_path]):
        loaded = Capabilities(paths, default_capabilities=defaults).load(
            work_root=tmp_path
        )
        assert {hds.name for hds in loaded.hds} == {"main", "mock-sub-agent"}
        assert len(loaded.tools) == 12


def test_direct_capabilities_loader_uses_bundled_path_when_paths_are_none(
    tmp_path: Path,
) -> None:
    loaded = Capabilities(None).load(work_root=tmp_path)

    assert {hds.name for hds in loaded.hds} == {"main", "mock-sub-agent"}


def test_direct_capabilities_loader_validates_default_configuration() -> None:
    defaults = DefaultCapabilities()
    with pytest.raises(TypeError, match="default_capabilities must be"):
        Capabilities([], default_capabilities=cast(DefaultCapabilities, object()))
    with pytest.raises(TypeError, match="capabilities_paths must be a sequence"):
        Capabilities(cast(list[Path], "bad"), default_capabilities=defaults)
    with pytest.raises(TypeError, match="capabilities_paths must contain only Path"):
        Capabilities([cast(Path, "bad")], default_capabilities=defaults)


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
        next(skill for skill in loaded.deep_skills if skill.name == "mock-skill")
        .render_instructions("exit")
        .startswith("Replacement skill instructions")
    )
    assert next(
        skill for skill in loaded.deep_skills if skill.name == "mock-skill"
    ) is not next(
        skill for skill in other_loaded.deep_skills if skill.name == "mock-skill"
    )
    assert skill.catalog is None
    assert provider.requests == []
    response = node.run("hello")
    assert response.get_message() == "done"
    system_content = provider.requests[0].get_messages()[0].get_message()
    assert system_content.startswith("Replacement instructions\n\n")
    assert "<harness-output-guidance>" in system_content
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
                            "exit_agent", {"text": "done", "items": []}
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
            "search_tool_outputs",
            "read_tool_output",
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

    assert (
        set(provider.requests[0].get_tools().list_tools())
        == {
            "mock-tool",
            "exit_agent",
            "deep-thinking",
            "control_tool_execution",
            "bash",
        }
        | DEEP_THINKING_TOOLS
    )
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
        "from exmekh_lib.core.agents.tools.decorator import deep_tool\n"
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
    assert (
        next(tool for tool in loaded.tools if tool.name == "mock-tool").execute(
            text="ok"
        )
        == "Replacement: ok"
    )


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

    assert defaults.overrides["first-tool"] == ToolOverride("first-tool", "second-tool")

    tools = {tool.name: tool for tool in loaded.tools}
    assert set(tools) == {"first-tool", "second-tool"} | DEEP_THINKING_TOOLS
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

    assert {tool.name for tool in loaded.tools} == {
        "mock-tool",
        "replacement-tool",
    } | DEEP_THINKING_TOOLS
    assert (
        next(tool for tool in loaded.tools if tool.name == "mock-tool").execute(
            text="ok"
        )
        == "replacement-tool: ok"
    )


def test_named_tool_override_preserves_source_hooks(tmp_path: Path) -> None:
    path = write_hds(tmp_path, "application", tools=("target-tool",))
    write_tool(path, "target-tool")
    write_tool(path, "source-tool")
    source_entrypoint = path / "tools" / "source-tool-directory" / "main.py"
    source_entrypoint.write_text(
        "from exmekh_lib.core.agents.tools.decorator import deep_tool\n"
        "from exmekh_lib.interfaces.tool_hook import AfterTool\n"
        "class Suffix(AfterTool):\n"
        "    def handle(self, context, result):\n"
        "        return result + '!'\n"
        "@deep_tool(name='source-tool', description='Source', after=Suffix())\n"
        "def run(work_root, text: str) -> str:\n"
        "    return text\n"
    )
    node = Node(
        [path],
        name_hds="application",
        default_capabilities=DefaultCapabilities(
            overrides=ToolOverride("target-tool", "source-tool")
        ),
    )

    loaded = node.load_capabilities(work_root=tmp_path)
    target = next(tool for tool in loaded.tools if tool.name == "target-tool")

    assert (
        target.execute_with_hooks({"text": "ok"}, work_root=tmp_path, history=())
        == "ok!"
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
    assert set(skills) == {"target-skill", "replacement-skill", "deep-thinking"}
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
    monkeypatch.setattr(DefaultCapabilities, "CAPABILITIES_PATH", empty_bundle)

    node = Node(name_hds="main", default_capabilities=DefaultCapabilities())
    with pytest.raises(
        ValueError,
        match="Selected default capabilities are missing: calculate, date-time, deep-thinking, exit_agent",
    ):
        node.load_capabilities(work_root=tmp_path)


def test_legacy_node_still_uses_explicit_builtin_path(tmp_path: Path) -> None:
    loaded = Node([BUILTIN_CAPABILITIES_PATH], name_hds="main").load_capabilities(
        work_root=tmp_path
    )
    assert {resource.name for resource in loaded.hds} == {"main", "mock-sub-agent"}
