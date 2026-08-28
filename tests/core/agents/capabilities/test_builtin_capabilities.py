from pathlib import Path

from pydantic import BaseModel

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.node import Node
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.interfaces.provider import Provider

CAPABILITIES_PATH = Path(__file__).resolve().parents[4] / "src" / "qs" / "capabilities"


class RecordingProvider(Provider):
    def __init__(self) -> None:
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


def test_loads_builtin_assistant_capabilities() -> None:
    capabilities = Capabilities([CAPABILITIES_PATH]).load()

    assert tuple(hds.name for hds in capabilities.hds) == ("main",)
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
    assert tuple(skill.name for skill in capabilities.exit_skills) == ("finish",)

    main_hds = capabilities.hds[0]
    assert main_hds.get_sub_skill_names() == ("mock-skill",)
    assert main_hds.get_sub_tool_names() == ("mock-tool",)
    assert main_hds.get_exit_skill_name() == "finish"


def test_builtin_mock_tool_returns_a_deterministic_result() -> None:
    capabilities = Capabilities([CAPABILITIES_PATH]).load()
    mock_tool = next(tool for tool in capabilities.tools if tool.name == "mock-tool")

    result = mock_tool.execute(text="example")

    assert result == "Mock tool result: example"


def test_builtin_capabilities_configure_the_main_node() -> None:
    capabilities = Node(
        capabilities_paths=[CAPABILITIES_PATH],
        name_hds="main",
    ).load_capabilities()

    assert capabilities.hds[0].get_hds().startswith("Sei un assistente semplice")


def test_builtin_client_receives_contextual_tool_descriptions() -> None:
    provider = RecordingProvider()
    node = Node(capabilities_paths=[CAPABILITIES_PATH], name_hds="main")
    node.initialize(provider)

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
        "finish": "Termina l'esecuzione restituendo la risposta finale.",
        "exit_agent": "Available in skill 'finish'.",
    }
