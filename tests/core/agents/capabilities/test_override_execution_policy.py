"""A bundled target's execution contract must not depend on path ordering."""

import pytest

from exmekh_lib.config.file_editing import FILE_EDIT_FORMAT
from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.default_composer import (
    _DefaultCapabilitiesComposer,
)
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.models.annotation_schema import AnnotationSchemaBuilder
from exmekh_lib.core.models.interactive import InteractiveDefinition
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.openai_function_tool_definition_mapper import (
    OpenAIFunctionToolDefinitionMapper,
)


@pytest.mark.parametrize("bundled_first", [True, False])
def test_override_retains_the_bundled_execution_contract(tmp_path, bundled_first):
    contract = InteractiveDefinition(FILE_EDIT_FORMAT.behavior, {"document": "edits"})
    bundled = Tool(
        lambda edits: edits,
        name="edit-file",
        background_enabled=False,
        interactive=contract,
    )
    application = Tool(lambda text: text, name="edit-file")

    def edit(edits: str) -> str:
        return edits

    replacement = Tool(edit, name="edit-file")
    sources = [
        (tmp_path / "bundle/tools/edit-file", bundled),
        (tmp_path / "application/tools/edit-file", application),
    ]
    if not bundled_first:
        sources.reverse()
    paths, tools = zip(*sources)
    defaults = DefaultCapabilities(harness=False, overrides={"edit-file": replacement})
    composed = _DefaultCapabilitiesComposer(
        defaults, tmp_path / "bundle", paths
    ).compose(CapabilitiesData((), tuple(tools)))
    registered = Tools(list(composed.tools)).get_tool("edit-file")

    assert registered is not None
    assert not registered.background_enabled
    assert registered.interactive == contract
    mapper = OpenAIFunctionToolDefinitionMapper(AnnotationSchemaBuilder())
    assert set(mapper.map(registered).get_parameters()["properties"]) == {"edits"}
    assert replacement.background_enabled
    assert replacement.interactive is None
