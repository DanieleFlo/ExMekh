"""Capability composition and execution must agree on the public tool interface."""

import inspect

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.core.agents.capabilities.contracts import apply_runtime_contract
from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.default_composer import (
    _DefaultCapabilitiesComposer,
)
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.capabilities.index import CapabilitiesIndex
from exmekh_lib.core.agents.node.tool_call_resolver import ToolCallResolver
from exmekh_lib.core.models.assistant_message import FunctionCall, ToolCall
from exmekh_lib.core.models.tool import PreformattedToolResult, Tool
from exmekh_lib.core.models.tools import Tools


def test_capability_override_registration_and_execution_use_public_tool_name(tmp_path):
    class NamedTool(Tool):
        def get_name(self):
            return "public-tool"

    tool = NamedTool(lambda: "done", name="implementation-name")
    loaded = CapabilitiesData(deep_skills=(), tools=(tool,))
    defaults = DefaultCapabilities(harness=False, overrides={"public-tool": tool})
    composer = _DefaultCapabilitiesComposer(
        defaults, tmp_path / "builtin", [tmp_path / "application-tool.py"]
    )
    composed = composer.compose(loaded)
    index = CapabilitiesIndex.build(composed)
    registered = Tools(list(composed.get_all_tools()))
    call = ToolCall("call", FunctionCall("public-tool", {}))

    assert set(index.tools) == {"public-tool"}
    assert registered.list_tools() == ["public-tool"]
    assert (
        ToolCallResolver(composed).resolve(call).execute(work_root=tmp_path) == "done"
    )


def test_runtime_contract_and_call_validation_use_the_public_work_signature(tmp_path):
    class Reader(Tool):
        def get_work_signature(self):
            return inspect.signature(self.execute)

        def execute(self, output_id, cursor=None, summary_mode="none"):
            return PreformattedToolResult(output_id)

    reader = Reader(lambda payload: payload, name=DEFAULT_CAPABILITIES.read_tool_output)
    tool = apply_runtime_contract(reader)
    result = tool.execute_with_hooks(
        {"output_id": "saved"}, work_root=tmp_path, history=()
    )
    assert result.content == "saved"
