"""Capture production prompt surfaces without contacting inference."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from pydantic import BaseModel

from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.openai_response_tool_mapper import (
    OpenAIResponseToolMapper,
)
from exmekh_lib.interfaces.provider import Provider
from exmekh_lib.tokenizers.qwen import QwenTokenizer
from exmekh_lib.utils.json import serialize_json
from exmekh_lib.utils.xml import append_text_element, read_text_element, render_xml


@dataclass(frozen=True)
class PromptBundle:
    system: str
    tools: list[dict[str, Any]]
    skills: dict[str, str]


class PromptCaptureProvider(Provider):
    def __init__(self) -> None:
        super().__init__(QwenTokenizer())
        self.request: ProviderRequest | None = None

    def call(self, request: ProviderRequest) -> ProviderResponse:
        self.request = request
        return ProviderResponse(AssistantMessage("Captured without inference."))

    def call_structured[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        raise AssertionError("Prompt capture must not request structured inference")

    def get_tools_text(self, tools: Tools | None) -> str:
        return (
            serialize_json(OpenAIResponseToolMapper.map_tools(tools))
            if tools is not None
            else ""
        )


def capture_prompts(
    work_root: Path, *, parameter_descriptions_in_schema: bool = False
) -> PromptBundle:
    """Use the real node composition, contextual descriptions and skill catalogs."""
    provider = PromptCaptureProvider()
    # Keep the tool set paired with the frozen 2026-10-20 baseline. New media
    # capabilities have separate acceptance tests, outside this prompt study.
    defaults = DefaultCapabilities(
        include={"main"},
        exclude={
            "generate-image",
        },
    )
    child = Node(
        name_hds="main", default_capabilities=defaults, agent_context_id="capture-child"
    )
    node = Node(
        name_hds="main",
        default_capabilities=defaults,
        children=[child],
        agent_context_id="capture-root",
    )
    agent = DeepAgent(
        provider,
        node,
        work_root,
        parameter_descriptions_in_schema=parameter_descriptions_in_schema,
    )
    agent.run("Capture the initial request.")
    assert provider.request is not None
    system = provider.request.get_messages()[0].get_message()
    assert isinstance(system, str)
    data = agent.capabilities_by_node[node]
    exit_tool = data.get_exit_tool(data.get_hds("main").get_exit_tool_name())
    return PromptBundle(
        system,
        json.loads(agent.get_tools_text()),
        {
            skill.name: skill.render_instructions(exit_tool.render_prompt_section())
            for skill in data.deep_skills
        },
    )


def save_bundle(bundle: PromptBundle, path: Path) -> None:
    root = ET.Element("harness-prompt-snapshot", {"version": "1"})
    append_text_element(root, "system", bundle.system)
    # Provider definitions retain their JSON protocol inside the XML snapshot.
    append_text_element(root, "tools-json", serialize_json(bundle.tools))
    for name, instructions in bundle.skills.items():
        append_text_element(root, "skill", instructions, attributes={"name": name})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(render_xml(root) + "\n")


def load_bundle(path: Path) -> PromptBundle:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    system = root.find("system")
    tools = root.find("tools-json")
    if system is None or tools is None:
        raise ValueError("Prompt snapshot requires system and tools-json.")
    return PromptBundle(
        read_text_element(system),
        json.loads(read_text_element(tools)),
        {
            item.attrib["name"]: read_text_element(item)
            for item in root.findall("skill")
        },
    )
