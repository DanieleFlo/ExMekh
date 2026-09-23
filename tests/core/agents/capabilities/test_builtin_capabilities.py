from pathlib import Path

from pydantic import BaseModel

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.node import Node
from qs.core.cache.cache_manager import CacheManager
from qs.core.cache.session_cache import SessionCache
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.interfaces.provider import Provider
from qs.interfaces.tokenizer import Tokenizer

CAPABILITIES_PATH = Path(__file__).resolve().parents[4] / "src" / "qs" / "capabilities"


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
            "stub-capabilities", "default", Path(".qs/tokenizers")
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


def test_loads_builtin_assistant_capabilities(tmp_path: Path) -> None:
    capabilities = Capabilities([CAPABILITIES_PATH]).load(work_root=tmp_path)

    assert tuple(hds.name for hds in capabilities.hds) == (
        "main",
        "mock-sub-agent",
    )
    assert {skill.name for skill in capabilities.deep_skills} == {
        "mock-skill",
        "nested-skill",
    }
    assert {tool.name for tool in capabilities.tools} == {
        "mock-sub-tool",
        "mock-tool",
        "nested-sub-tool",
        "shared-tool",
    }
    assert tuple(skill.name for skill in capabilities.exit_tools) == ("exit_agent",)

    main_hds = capabilities.hds[0]
    assert main_hds.get_sub_skill_names() == ("mock-skill",)
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

    assert capabilities.hds[0].get_hds().startswith("Sei un assistente semplice")


def test_builtin_client_receives_contextual_tool_descriptions(tmp_path: Path) -> None:
    provider = RecordingProvider()
    node = Node(capabilities_paths=[CAPABILITIES_PATH], name_hds="main")
    node.initialize(provider, work_root=tmp_path)

    node.run("test descriptions")

    request_tools = provider.requests[0].get_tools()
    assert request_tools is not None
    descriptions = {tool.name: tool.description for tool in request_tools.get_tools()}
    assert descriptions == {
        "mock-skill": (
            "Produce una risposta dimostrativa senza eseguire operazioni reali."
        ),
        "nested-skill": "Available in skill 'mock-skill'.",
        "mock-sub-tool": "Available in skill 'mock-skill'.",
        "mock-tool": ("Restituisce il testo ricevuto come risultato dimostrativo."),
        "nested-sub-tool": "Available in skill 'nested-skill'.",
        "shared-tool": "Available in skills 'mock-skill', 'nested-skill'.",
        "exit_agent": (
            "Conclude la skill corrente restituendo il risultato al chiamante. "
            "Available in every skill."
        ),
        "get_execution_history": (
            "Return navigable XML history of capability calls. Omit "
            "start_position for a head-and-tail overview, or use a 1-based "
            "position and limit to read a page."
        ),
        "search_saved_tool_outputs": (
            "Search node-scoped saved outputs and return paginated XML previews. "
            "All filters are optional."
        ),
        "read_saved_tool_output": (
            "Read one saved output by ID as an XML header followed by its "
            "complete content."
        ),
    }
