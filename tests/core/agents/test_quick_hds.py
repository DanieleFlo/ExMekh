from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.core.agents.capabilities.default_catalog import (
    DEFAULT_CAPABILITY_CATALOG,
)
from exmekh_lib.core.agents.node import DefaultCapabilities, Node
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool, ToolArguments


def test_quick_hds_has_only_browsing_file_reading_and_images(tmp_path):
    names = [
        "quick",
        "exit_agent",
        "search-web",
        "read-web-page",
        "inspect-file",
        "read-file",
        "generate-image",
    ]
    node = Node(
        name_hds=DEFAULT_CAPABILITIES.quick,
        default_capabilities=DefaultCapabilities(harness=False, include=names),
    )
    data = node.load_capabilities(work_root=tmp_path)
    assert [hds.name for hds in data.hds] == ["quick"]
    assert {tool.name for tool in data.tools} == set(names) - {"quick", "exit_agent"}
    assert data.deep_skills == ()
    assert DEFAULT_CAPABILITY_CATALOG["quick"].kind == "hds"
    assert set(data.hds[0].metadata.sub_tools) == set(names[2:])


def test_tool_hook_invocation_identity_survives_argument_override(tmp_path):
    seen = []

    class Before(BeforeTool):
        def handle(self, context):
            seen.append(context.call_id)
            return ToolArguments({"value": "updated"})

    class After(AfterTool):
        def handle(self, context, result):
            seen.append(context.call_id)
            assert context.arguments == {"value": "updated"}
            return result

    tool = Tool(lambda value: value, name="sample", before=Before(), after=After())
    assert (
        tool.execute_with_hooks(
            {"value": "original"}, call_id="call-17", work_root=tmp_path, history=()
        )
        == "updated"
    )
    assert seen == ["call-17", "call-17"]
