import logging
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.capabilities.index import CapabilitiesIndex
from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.deep_skill.metadata import DeepSkillMetadata
from qs.core.agents.exit_tool.data import ExitToolData
from qs.core.agents.exit_tool.exit_tool import ExitTool
from qs.core.agents.hds.hds import HDS
from qs.core.agents.hds.metadata import HDSMetadata
from qs.core.models.tool import Tool


def _hds() -> HDS:
    return HDS(
        name="hds",
        markdown_path=Path("hds.md"),
        metadata=HDSMetadata(
            description="Description",
            sub_skills=(),
            sub_tools=(),
            exit_tool="exit_agent",
        ),
        body="Body",
        references=(),
    )


def _deep_skill() -> DeepSkill:
    return DeepSkill(
        name="skill",
        markdown_path=Path("skill.md"),
        metadata=DeepSkillMetadata(
            description="Description",
            inherits_parent_tools=False,
            sub_skills=(),
            sub_tools=(),
        ),
        body="Body",
        references=(),
    )


def _exit_tool() -> ExitTool:
    def finish() -> ExitToolData:
        return ExitToolData("done", ["item"])

    return ExitTool(
        finish,
        name="exit_agent",
        description="Exit",
        instructions_path=Path("finish/instructions.md"),
        instructions="Finish",
        folder_path=Path("finish"),
    )


def test_rejects_string_as_capabilities_path_collection() -> None:
    with pytest.raises(
        TypeError,
        match="capabilities_paths must be a sequence of Path",
    ):
        Capabilities(cast(Sequence[Path], "capabilities"))


def test_rejects_non_path_collection_item() -> None:
    with pytest.raises(
        TypeError,
        match="capabilities_paths must contain only Path",
    ):
        Capabilities([cast(Path, "capabilities")])


def test_rejects_empty_capabilities_paths() -> None:
    with pytest.raises(ValueError, match="capabilities_paths cannot be empty"):
        Capabilities([])


def test_rejects_duplicate_capabilities_paths(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"

    with pytest.raises(ValueError, match="Duplicate capabilities path"):
        Capabilities([capabilities_path, capabilities_path])


def test_rejects_symbolic_link_capabilities_path(tmp_path: Path) -> None:
    target_path = tmp_path / "target" / "capabilities"
    target_path.mkdir(parents=True)
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.symlink_to(target_path, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_rejects_symbolic_link_capability_type_folder(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.mkdir()
    external_tools_path = tmp_path / "external-tools"
    external_tools_path.mkdir()
    (capabilities_path / "tools").symlink_to(
        external_tools_path,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_validates_all_roots_before_creating_type_folders(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "first" / "capabilities"
    valid_path.mkdir(parents=True)
    invalid_path = tmp_path / "second" / "invalid"
    invalid_path.mkdir(parents=True)

    with pytest.raises(ValueError, match="must be named 'capabilities'"):
        Capabilities([valid_path, invalid_path]).load(work_root=tmp_path)

    assert tuple(valid_path.iterdir()) == ()


def test_rejects_wrong_capabilities_folder_name(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capability"
    capabilities_path.mkdir()

    with pytest.raises(
        ValueError,
        match="Capabilities folder must be named 'capabilities'",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_rejects_missing_capabilities_path(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_rejects_capabilities_path_that_is_not_directory(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.write_text("not a directory")

    with pytest.raises(ValueError, match="must be a directory"):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_rejects_unexpected_capabilities_entry(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.mkdir()
    (capabilities_path / "instructions.txt").write_text("unexpected")

    with pytest.raises(
        ValueError,
        match="Unexpected entry inside capabilities folder",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_rejects_capability_type_that_is_not_directory(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.mkdir()
    (capabilities_path / "tools").write_text("not a directory")

    with pytest.raises(ValueError, match="entry 'tools' must be a directory"):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_returns_exit_tool_and_hds_children(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    exit_tool_path = capabilities_path / "exit-tool" / "finish"
    hds_path = capabilities_path / "hds" / "document"
    exit_tool_path.mkdir(parents=True)
    hds_path.mkdir(parents=True)
    capabilities = Capabilities([capabilities_path])

    assert capabilities.get_hds_paths() == [hds_path]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("deep_skills", [], "deep_skills must be a tuple"),
        ("deep_skills", (object(),), "deep_skills must contain only"),
        ("tools", [], "tools must be a tuple"),
        ("tools", (object(),), "tools must contain only"),
        ("tools", (_hds(),), "tools must contain only"),
        ("tools", (_exit_tool(),), "tools must contain only"),
        ("hds", [], "hds must be a tuple"),
        ("hds", (object(),), "hds must contain only"),
        ("hds", (_deep_skill(),), "hds must contain only"),
        ("exit_tools", [], "exit_tools must be a tuple"),
        ("exit_tools", (object(),), "exit_tools must contain only"),
    ],
)
def test_capabilities_data_rejects_invalid_collections(field, value, message):
    values = {"deep_skills": (), "tools": (), field: value}
    with pytest.raises(TypeError, match=message):
        CapabilitiesData(**values)


def test_capabilities_data_returns_all_tool_types() -> None:
    skill, tool, exit_tool = (
        _deep_skill(),
        Tool(lambda: None, name="tool"),
        _exit_tool(),
    )
    data = CapabilitiesData((skill,), (tool,), exit_tools=(exit_tool,))
    assert data.get_all_tools() == (skill, tool, exit_tool)


def test_capabilities_data_resolves_runtime_capabilities() -> None:
    hds, exit_tool = _hds(), _exit_tool()
    data = CapabilitiesData((), (), hds=(hds,), exit_tools=(exit_tool,))
    assert data.get_hds("hds") is hds
    assert data.get_exit_tool("exit_agent") is exit_tool
    with pytest.raises(ValueError, match="HDS 'missing' not found"):
        data.get_hds("missing")
    with pytest.raises(ValueError, match="Exit tool 'missing' not found"):
        data.get_exit_tool("missing")


def test_capabilities_data_indexes_skill_availability() -> None:
    parent_skill = DeepSkill(
        name="parent",
        markdown_path=Path("parent.md"),
        metadata=DeepSkillMetadata(
            description="Parent",
            inherits_parent_tools=False,
            sub_skills=("child",),
            sub_tools=("shared",),
        ),
        body="Parent",
        references=(),
    )
    child_skill = DeepSkill(
        name="child",
        markdown_path=Path("child.md"),
        metadata=DeepSkillMetadata(
            description="Child",
            inherits_parent_tools=False,
            sub_skills=(),
            sub_tools=("shared",),
        ),
        body="Child",
        references=(),
    )
    data = CapabilitiesData(
        deep_skills=(parent_skill, child_skill),
        tools=(),
        exit_tools=(_exit_tool(),),
    )

    assert data.get_skill_availability() == {
        "child": ("parent",),
        "shared": ("parent", "child"),
    }


def test_index_rejects_non_capabilities_data() -> None:
    with pytest.raises(TypeError, match="data must be CapabilitiesData"):
        CapabilitiesIndex.build(cast(CapabilitiesData, object()))


@pytest.mark.parametrize("factory", [_hds, _deep_skill, _exit_tool])
def test_debug_representations_preserve_capability_identity(factory):
    capability = factory()
    assert capability.name in str(capability)
    assert type(capability).__name__ in str(capability)
    assert str(capability) in repr(capability)


def test_debug_disabled_does_not_render_tools(caplog):
    class UnrenderableTool(Tool):
        def __repr__(self):
            raise AssertionError("debug representation should not be evaluated")

    tool = UnrenderableTool(lambda: None, name="work")
    data = CapabilitiesData((), (tool,))
    with caplog.at_level(logging.INFO, logger="qs.core.agents.capabilities.data"):
        assert data.get_all_tools() == (tool,)
        assert data.get_contextual_tools(_hds())[0].name == "work"


def test_contextual_debug_includes_every_tool_and_original_type(caplog):
    hds, skill, exit_tool = _hds(), _deep_skill(), _exit_tool()
    data = CapabilitiesData(
        (skill,),
        (Tool(lambda: None, name="regular"),),
        hds=(hds,),
        exit_tools=(exit_tool,),
    )
    with caplog.at_level(logging.DEBUG, logger="qs.core.agents.capabilities.data"):
        tools = data.get_contextual_tools(hds)
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Contextual capability:")
    ]
    assert len(messages) == len(tools)
    assert any("type=DeepSkill name=skill" in message for message in messages)
    assert any("type=ExitTool name=exit_agent" in message for message in messages)
    assert all(
        any(f"name={tool.name} " in message for message in messages) for tool in tools
    )
    assert "HDS:" in str(data) and "ExitTools:" in str(data)
