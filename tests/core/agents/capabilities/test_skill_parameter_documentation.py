from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from xml.etree import ElementTree as ET

import pytest

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.capabilities.catalog_attacher import SkillCatalogAttacher
from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.index import CapabilitiesIndex
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.deep_skill.metadata import DeepSkillMetadata
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.agents.hds.metadata import HDSMetadata
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.openai_chat_completion_tool_mapper import (
    OpenAIChatCompletionToolMapper,
)
from exmekh_lib.core.providers.openai_response_tool_mapper import (
    OpenAIResponseToolMapper,
)
from exmekh_lib.utils.xml import read_text_element


class SearchRequest(TypedDict):
    description: Annotated[str, 'Search <text> & "references".']
    mode: Annotated[Literal["exact", "prefix"], "Choose the matching rule."]


@dataclass
class SearchOptions:
    requests: list[Annotated[SearchRequest, "One search request."]]


def search(
    options: Annotated[SearchOptions | None, "Search options, or null for defaults."],
    labels: Annotated[dict[str, Annotated[str, "A label value."]], "Labels by key."],
) -> str:
    """Search indexed documents."""
    return "result"


def make_skill(tmp_path: Path, name: str, tools: tuple[str, ...]) -> DeepSkill:
    return DeepSkill(
        name=name,
        markdown_path=tmp_path / f"{name}.md",
        metadata=DeepSkillMetadata(
            description=name, sub_tools=tools, sub_skills=(), inherits_parent_tools=True
        ),
        body="Skill body",
        references=(),
    )


def make_hds(tmp_path: Path, tools: tuple[str, ...]) -> HDS:
    return HDS(
        name="root",
        markdown_path=tmp_path / "root.md",
        metadata=HDSMetadata(
            description="Root",
            sub_tools=tools,
            sub_skills=("research",),
            exit_tool="finish",
        ),
        body="Root instructions",
        references=(),
    )


@pytest.mark.parametrize(
    "mapper", [OpenAIChatCompletionToolMapper, OpenAIResponseToolMapper]
)
def test_contextual_schemas_hide_only_internal_documentation_and_preserve_contract(
    tmp_path, mapper
):
    tool = Tool(search, name="search")
    skill = make_skill(tmp_path, "research", ("search",))
    data = CapabilitiesData(deep_skills=(skill,), tools=(tool,))
    hds = make_hds(tmp_path, ())
    original = mapper.map_tools(Tools([tool]))[0]
    definitions = mapper.map_tools(Tools(list(data.get_contextual_tools(hds))))
    mapped = next(
        item.get("function", item)
        for item in definitions
        if item.get("function", item)["name"] == "search"
    )
    full = original.get("function", original)

    assert mapped["description"] == "Requires open skill 'research'."
    properties = mapped["parameters"]["properties"]
    options = properties["options"]["anyOf"][0]
    request = options["properties"]["requests"]["items"]
    assert request["properties"]["description"] == {"type": "string"}
    assert request["properties"]["mode"]["enum"] == ["exact", "prefix"]
    assert request["required"] == ["description", "mode"]
    assert properties["labels"]["additionalProperties"] == {"type": "string"}
    assert "description" not in properties["execution_mode"]
    assert mapped["parameters"]["required"] == full["parameters"]["required"]
    assert mapper.map_tools(Tools([tool]))[0] == original

    direct = data.get_contextual_tools(make_hds(tmp_path, ("search",)))
    assert (
        mapper.map_tools(Tools([next(t for t in direct if t.name == "search")]))[0]
        == original
    )


def test_catalog_loads_nested_parameter_documentation_without_repeating_execution_guidance(
    tmp_path,
):
    tool = Tool(search, name="search")
    first = make_skill(tmp_path, "research", ("search",))
    second = make_skill(tmp_path, "review", ("search",))
    data = CapabilitiesData(deep_skills=(first, second), tools=(tool,))
    SkillCatalogAttacher(CapabilitiesIndex.build(data)).attach_to_all(data.deep_skills)

    for skill in data.deep_skills:
        root = ET.fromstring(skill.catalog.render_prompt_section())
        assert root.findtext("./sub-tools/tool") == "Search indexed documents."
        parameters = root.findall("./parameter-descriptions/tool/parameter")
        docs = {p.attrib["path"]: read_text_element(p) for p in parameters}
        assert docs == {
            "/properties/options": "Search options, or null for defaults.",
            "/properties/options/anyOf/0/properties/requests/items": "One search request.",
            "/properties/options/anyOf/0/properties/requests/items/properties/description": 'Search <text> & "references".',
            "/properties/options/anyOf/0/properties/requests/items/properties/mode": "Choose the matching rule.",
            "/properties/labels": "Labels by key.",
            "/properties/labels/additionalProperties": "A label value.",
        }
        assert RUNTIME.execution_mode_guidance not in skill.render_instructions(
            "Exit instructions"
        )
        assert all(p.attrib["type"] == "string" for p in parameters)


def test_synchronous_business_parameters_with_execution_names_keep_their_documentation(
    tmp_path,
):
    def synchronous(execution_mode: Annotated[str, "Business option."]) -> str:
        return execution_mode

    tool = Tool(synchronous, name="synchronous", background_enabled=False)
    skill = make_skill(tmp_path, "research", ("synchronous",))
    data = CapabilitiesData(deep_skills=(skill,), tools=(tool,))
    SkillCatalogAttacher(CapabilitiesIndex.build(data)).attach_to_all(data.deep_skills)
    root = ET.fromstring(skill.catalog.render_prompt_section())
    assert (
        root.findtext("./parameter-descriptions/tool/parameter") == "Business option."
    )
