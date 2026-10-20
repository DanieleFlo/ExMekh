"""Prompt rendering keeps malformed literals and recursive catalogs bounded."""

import pytest

from exmekh_lib.core.agents.capabilities.contracts import RUNTIME_CONTRACTS
from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.index import CapabilitiesIndex
from exmekh_lib.core.agents.capabilities.prompts import CapabilityPrompts
from exmekh_lib.core.agents.capabilities.validator import CapabilitiesValidator
from exmekh_lib.core.agents.capability.markdown_literals import literal_spans
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.deep_skill.parser import SkillMarkdownParser
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.tool import Tool


def test_reachable_catalog_handles_repeated_skill_references(tmp_path):
    path = tmp_path / "recursive.md"
    path.write_text(
        "---\ndescription: Recursive catalog\nsub-skills:\n  - recursive\n---\nBody"
    )
    skill = DeepSkill.from_markdown(path, SkillMarkdownParser())
    data = CapabilitiesData((skill,), ())
    context = CapabilityPrompts(data, ()).context(skill)
    assert context is not None
    CapabilitiesValidator(CapabilitiesIndex.build(data)).validate((skill,))


def test_unclosed_inline_code_does_not_hide_the_rest_of_the_prompt():
    assert literal_spans("An unmatched ` then ordinary instructions") == ()


def test_runtime_contract_rejects_an_unstructured_inspection_result():
    with pytest.raises(TypeError, match="InspectionReport"):
        RUNTIME_CONTRACTS["inspect-file"].check_result(
            "inspect-file", "claimed success"
        )


def test_description_renderer_requires_text():
    with pytest.raises(TypeError, match="string"):
        Tool(lambda: None, name="work").render_description(lambda template: None)


def test_uninitialized_node_cannot_claim_an_inventory():
    with pytest.raises(RuntimeError):
        Node(name_hds="main").get_capability_inventory()


def test_runtime_contract_preserves_explicit_inspection_failure():
    failure = {"ok": False, "error": "Unreadable file"}
    assert (
        RUNTIME_CONTRACTS["inspect-file"].check_result("inspect-file", failure)
        is failure
    )
