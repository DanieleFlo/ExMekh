from pathlib import Path

import pytest
from pydantic import BaseModel

from exmekh_lib.config.storage import DEFAULT_STORAGE_SETTINGS
from exmekh_lib.config.web import WEB_NAVIGATION
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.node import DefaultCapabilities, Node
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
from tests.capabilities_support import MOCK_CAPABILITIES_PATH

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

    assert tuple(hds.name for hds in capabilities.hds) == ("main", "quick")
    assert {skill.name for skill in capabilities.deep_skills} == {
        "deep-thinking",
        "file-processing",
    }
    assert {tool.name for tool in capabilities.tools} == {
        "generate-image",
        "calculate",
        "date-time",
        "inspect-file",
        "read-file",
        "edit-file",
        "new-file",
        "delete-file",
        "interact-web-page",
        "read-web-page",
        "search-web",
        "logic-check",
        "logic-consistency",
        "logic-counterexample",
    }
    assert tuple(skill.name for skill in capabilities.exit_tools) == ("exit_agent",)

    main_hds = capabilities.hds[0]
    assert main_hds.get_sub_skill_names() == (
        "deep-thinking",
        "file-processing",
    )
    assert main_hds.get_sub_tool_names() == ("generate-image",)
    assert main_hds.get_exit_tool_name() == "exit_agent"


def test_mock_tool_is_loaded_only_from_test_fixtures(tmp_path: Path) -> None:
    capabilities = Capabilities(
        [MOCK_CAPABILITIES_PATH], default_capabilities=DefaultCapabilities()
    ).load(work_root=tmp_path)
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
    assert (
        descriptions["inspect-file"]
        == "Requires an open skill: 'deep-thinking', 'file-processing'."
    )
    scoped = {
        "inspect-file": "Requires an open skill: 'deep-thinking', 'file-processing'.",
        "read-file": "Requires an open skill: 'deep-thinking', 'file-processing'.",
        "edit-file": "Requires open skill 'file-processing'.",
        "new-file": "Requires open skill 'file-processing'.",
        "delete-file": "Requires open skill 'file-processing'.",
        "calculate": "Requires open skill 'deep-thinking'.",
        "date-time": "Requires open skill 'deep-thinking'.",
        "interact-web-page": "Requires open skill 'deep-thinking'.",
        "read-web-page": "Requires open skill 'deep-thinking'.",
        "search-web": "Requires open skill 'deep-thinking'.",
        "logic-check": "Requires open skill 'deep-thinking'.",
        "logic-consistency": "Requires open skill 'deep-thinking'.",
        "logic-counterexample": "Requires open skill 'deep-thinking'.",
    }
    assert {name: descriptions[name] for name in scoped} == scoped
    for name in ("file-processing", "deep-thinking"):
        assert descriptions[name].startswith("Open a scope")
    for name, description in descriptions.items():
        assert description.strip(), name
        assert "<if-tool" not in description
    assert "Available in every skill." in descriptions["exit_agent"]
    assert "output_id" in descriptions["read_tool_output"]
    assert "tool_call_id" in descriptions["control_tool_execution"]
