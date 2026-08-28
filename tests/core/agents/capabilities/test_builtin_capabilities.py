from pathlib import Path

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.node import Node

CAPABILITIES_PATH = Path(__file__).resolve().parents[4] / "src" / "qs" / "capabilities"


def test_loads_builtin_assistant_capabilities() -> None:
    capabilities = Capabilities([CAPABILITIES_PATH]).load()

    assert tuple(hds.name for hds in capabilities.hds) == ("main",)
    assert tuple(skill.name for skill in capabilities.deep_skills) == ("mock-skill",)
    assert tuple(tool.name for tool in capabilities.tools) == ("mock-tool",)
    assert tuple(skill.name for skill in capabilities.exit_skills) == ("finish",)

    main_hds = capabilities.hds[0]
    assert main_hds.metadata.sub_skills == ("mock-skill",)
    assert main_hds.metadata.sub_tools == ("mock-tool",)
    assert main_hds.metadata.exit_skill == "finish"


def test_builtin_mock_tool_returns_a_deterministic_result() -> None:
    capabilities = Capabilities([CAPABILITIES_PATH]).load()

    result = capabilities.tools[0].tool("example")

    assert result == "Mock tool result: example"


def test_builtin_capabilities_configure_the_main_node() -> None:
    capabilities = Node(
        capabilities_paths=[CAPABILITIES_PATH],
        name_hds="main",
    ).load_capabilities()

    assert capabilities.hds[0].get_hds().startswith("Sei un assistente semplice")
