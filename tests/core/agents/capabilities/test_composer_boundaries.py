import pytest

from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.capabilities.default_composer import (
    _DefaultCapabilitiesComposer,
)
from qs.core.agents.capabilities.defaults import DefaultCapabilities
from qs.core.agents.capabilities.tool_composer import _DefaultToolComposer
from qs.core.models.tool import Tool


def test_tool_composer_without_defaults_preserves_tool() -> None:
    tool = Tool(lambda: "done", name="work")
    assert _DefaultToolComposer(None).apply_hooks(tool) is tool


def test_capability_composer_rejects_unresolved_override(tmp_path) -> None:
    defaults = DefaultCapabilities(harness=False)
    defaults._overrides = {"mock-tool": object()}
    composer = _DefaultCapabilitiesComposer(defaults, tmp_path, ())

    with pytest.raises(TypeError, match="must be resolved before composition"):
        composer._apply_overrides(CapabilitiesData((), ()))
