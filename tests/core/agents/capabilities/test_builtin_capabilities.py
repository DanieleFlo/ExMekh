from pathlib import Path

import pytest
from pydantic import BaseModel

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.storage import DEFAULT_STORAGE_SETTINGS
from exmekh_lib.config.web import WEB_NAVIGATION
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.cache.cache_manager import CacheManager
from exmekh_lib.core.cache.session_cache import SessionCache
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.openai_chat_completion_tool_mapper import (
    OpenAIChatCompletionToolMapper,
)
from exmekh_lib.core.providers.openai_response_tool_mapper import (
    OpenAIResponseToolMapper,
)
from exmekh_lib.interfaces.provider import Provider
from exmekh_lib.interfaces.tokenizer import Tokenizer

CAPABILITIES_PATH = (
    Path(__file__).resolve().parents[4] / "src" / "exmekh_lib" / "capabilities"
)


class _StubTokenizer(Tokenizer):
    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

    def initialize(self) -> None:
        pass

    def generate_id(self) -> str:
        return "stub-id"

    def get_session(self) -> SessionCache:
        return CacheManager.get_session(
            "stub-capabilities",
            "default",
            DEFAULT_STORAGE_SETTINGS.tokenizer_cache_directory,
        )


class RecordingProvider(Provider):
    def __init__(self) -> None:
        super().__init__(_StubTokenizer())
        self.requests: list[ProviderRequest] = []

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(AssistantMessage("done"))

    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        del request
        raise NotImplementedError


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_browser_action_schema_describes_valid_bounded_actions(tmp_path: Path, api):
    capabilities = Capabilities([CAPABILITIES_PATH]).load(work_root=tmp_path)
    tool = next(tool for tool in capabilities.tools if tool.name == "interact-web-page")
    if api == "responses":
        definition = OpenAIResponseToolMapper.map_tools(Tools([tool]))[0]
    else:
        definition = OpenAIChatCompletionToolMapper.map_tools(Tools([tool]))[0][
            "function"
        ]
    actions = definition["parameters"]["properties"]["actions"]
    assert actions["minItems"] == 1
    assert actions["maxItems"] == WEB_NAVIGATION.max_actions
    assert actions["items"]["required"] == ["kind"]
    assert "navigate" in actions["items"]["properties"]["kind"]["enum"]
    assert actions["items"]["properties"]["url"]["type"] == "string"


def test_loads_builtin_assistant_capabilities(tmp_path: Path) -> None:
    capabilities = Capabilities([CAPABILITIES_PATH]).load(work_root=tmp_path)

    assert tuple(hds.name for hds in capabilities.hds) == (
        "main",
        "mock-sub-agent",
    )
    assert {skill.name for skill in capabilities.deep_skills} == {
        "deep-thinking",
        "file-processing",
        "mock-skill",
        "nested-skill",
    }
    assert {tool.name for tool in capabilities.tools} == {
        "calculate",
        "date-time",
        "inspect-file",
        "read-file",
        "edit-file",
        "new-file",
        "delete-file",
        "interact-web-page",
        "mock-sub-tool",
        "mock-tool",
        "nested-sub-tool",
        "read-web-page",
        "search-web",
        "shared-tool",
        "logic-check",
        "logic-consistency",
        "logic-counterexample",
    }
    assert tuple(skill.name for skill in capabilities.exit_tools) == ("exit_agent",)

    main_hds = capabilities.hds[0]
    assert main_hds.get_sub_skill_names() == (
        "mock-skill",
        "deep-thinking",
        "file-processing",
    )
    assert main_hds.get_sub_tool_names() == ("mock-tool",)
    assert main_hds.get_exit_tool_name() == "exit_agent"


def test_builtin_mock_tool_returns_a_deterministic_result(tmp_path: Path) -> None:
    capabilities = Capabilities([CAPABILITIES_PATH]).load(work_root=tmp_path)
    mock_tool = next(tool for tool in capabilities.tools if tool.name == "mock-tool")

    result = mock_tool.execute(text="example")

    assert result == "Mock tool result: example"


def test_builtin_capabilities_configure_the_main_node(tmp_path: Path) -> None:
    capabilities = Node(
        capabilities_paths=[CAPABILITIES_PATH],
        name_hds="main",
    ).load_capabilities(work_root=tmp_path)

    assert capabilities.hds[0].get_hds().startswith("You are a clear, concise")


def test_builtin_client_receives_contextual_tool_descriptions(tmp_path: Path) -> None:
    provider = RecordingProvider()
    node = Node(capabilities_paths=[CAPABILITIES_PATH], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    node.run("test descriptions")

    request_tools = provider.requests[0].get_tools()
    assert request_tools is not None
    descriptions = {tool.name: tool.description for tool in request_tools.get_tools()}
    assert descriptions["inspect-file"] == "Available in skill 'file-processing'."
    assert descriptions == {
        "file-processing": "Inspects, reads, edits, creates and removes workspace files.",
        "inspect-file": "Available in skill 'file-processing'.",
        "read-file": "Available in skill 'file-processing'.",
        "edit-file": "Available in skill 'file-processing'.",
        "new-file": "Available in skill 'file-processing'.",
        "delete-file": "Available in skill 'file-processing'.",
        "deep-thinking": (
            "Investigates difficult questions with web evidence, "
            "mathematical and date/time computation, and formal Z3 reasoning."
        ),
        "calculate": "Available in skill 'deep-thinking'.",
        "date-time": "Available in skill 'deep-thinking'.",
        "interact-web-page": "Available in skill 'deep-thinking'.",
        "read-web-page": "Available in skill 'deep-thinking'.",
        "search-web": "Available in skill 'deep-thinking'.",
        "logic-check": "Available in skill 'deep-thinking'.",
        "logic-consistency": "Available in skill 'deep-thinking'.",
        "logic-counterexample": "Available in skill 'deep-thinking'.",
        "mock-skill": (
            "Produces a demonstration response without performing real operations."
        ),
        "nested-skill": "Available in skill 'mock-skill'.",
        "mock-sub-tool": "Available in skill 'mock-skill'.",
        "mock-tool": ("Returns the received text as a demonstration result."),
        "nested-sub-tool": "Available in skill 'nested-skill'.",
        "shared-tool": "Available in skills 'mock-skill', 'nested-skill'.",
        "exit_agent": (
            "Finishes the current skill and returns its result to the caller. "
            "Available in every skill."
        ),
        "get_execution_history": (
            "Return navigable XML history of capability calls and saved user steering. Omit "
            "start_position for a head-and-tail overview, or use a 1-based "
            f"position and limit from {RUNTIME.history_min_page_limit} to {RUNTIME.memory_max_page_limit} to read a page."
        ),
        "search_tool_outputs": (
            "Search this node's saved and runtime tool outputs, including "
            "background work and user steering saved inside skills. Return paginated XML previews with status "
            "and output IDs. All filters are optional."
        ),
        "read_tool_output": (
            "Read the current status and available output of a tool by output ID. "
            "Use an optional cursor to read only new output; omit it to read the "
            "complete output. Includes the final result when available. "
            "Also reads saved user steering and restores its attachments. "
            "summary_mode defaults to 'none'; 'brief' gives 3 to 5 bullet points, "
            f"'detailed' preserves essential details. Outputs below {RUNTIME.tool_output_summary_min_tokens} tokens stay complete."
        ),
        "control_tool_execution": (
            "Wait for a tool execution by its original tool_call_id, or request "
            "cancellation. Waiting returns as soon as the work finishes or "
            f"the wait limit expires; read its output with {RUNTIME.read_tool_name}."
        ),
        "bash": (
            "Run a command in Bash from the node workroot. Each call starts an "
            "independent process. Read incremental stdout/stderr with "
            f"{RUNTIME.read_tool_name}(output_id, cursor); wait or cancel with "
            f"{RUNTIME.execution_tool_name}(tool_call_id). Requires Bash on PATH."
        ),
    }
