from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.capabilities.index import CapabilitiesIndex
from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.deep_skill.metadata import DeepSkillMetadata
from qs.core.agents.exit_skill.exit_skill import ExitSkill
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
            exit_skill="finish",
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
        exit_skill_name="finish",
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
        Capabilities([capabilities_path]).load()


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
        Capabilities([capabilities_path]).load()


def test_validates_all_roots_before_creating_type_folders(
    tmp_path: Path,
) -> None:
    valid_path = tmp_path / "first" / "capabilities"
    valid_path.mkdir(parents=True)
    invalid_path = tmp_path / "second" / "invalid"
    invalid_path.mkdir(parents=True)

    with pytest.raises(ValueError, match="must be named 'capabilities'"):
        Capabilities([valid_path, invalid_path]).load()

    assert tuple(valid_path.iterdir()) == ()


def test_rejects_wrong_capabilities_folder_name(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capability"
    capabilities_path.mkdir()

    with pytest.raises(
        ValueError,
        match="Capabilities folder must be named 'capabilities'",
    ):
        Capabilities([capabilities_path]).load()


def test_rejects_missing_capabilities_path(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        Capabilities([capabilities_path]).load()


def test_rejects_capabilities_path_that_is_not_directory(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.write_text("not a directory")

    with pytest.raises(ValueError, match="must be a directory"):
        Capabilities([capabilities_path]).load()


def test_rejects_unexpected_capabilities_entry(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.mkdir()
    (capabilities_path / "instructions.txt").write_text("unexpected")

    with pytest.raises(
        ValueError,
        match="Unexpected entry inside capabilities folder",
    ):
        Capabilities([capabilities_path]).load()


def test_rejects_capability_type_that_is_not_directory(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.mkdir()
    (capabilities_path / "tools").write_text("not a directory")

    with pytest.raises(ValueError, match="entry 'tools' must be a directory"):
        Capabilities([capabilities_path]).load()


def test_returns_exit_skill_and_hds_children(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    exit_skill_path = capabilities_path / "exit-skill" / "finish"
    hds_path = capabilities_path / "hds" / "document"
    exit_skill_path.mkdir(parents=True)
    hds_path.mkdir(parents=True)
    capabilities = Capabilities([capabilities_path])

    assert capabilities.get_hds_paths() == [hds_path]


@pytest.mark.parametrize(
    ("deep_skills", "tools", "hds", "exit_skills", "exit_tools", "message"),
    [
        (
            cast(tuple[DeepSkill, ...], []),
            (),
            (),
            (),
            (),
            "deep_skills must be a tuple",
        ),
        ((cast(DeepSkill, object()),), (), (), (), (), "deep_skills must contain only"),
        ((), cast(tuple[Tool, ...], []), (), (), (), "tools must be a tuple"),
        ((), (cast(Tool, object()),), (), (), (), "tools must contain only"),
        ((), (_hds(),), (), (), (), "tools must contain only"),
        ((), (), cast(tuple[HDS, ...], []), (), (), "hds must be a tuple"),
        ((), (), (cast(HDS, object()),), (), (), "hds must contain only"),
        ((), (), (_deep_skill(),), (), (), "hds must contain only"),
        (
            (),
            (),
            (),
            cast(tuple[ExitSkill, ...], []),
            (),
            "exit_skills must be a tuple",
        ),
        ((), (), (), (cast(ExitSkill, object()),), (), "exit_skills must contain only"),
        ((), (), (), (), cast(tuple[ExitTool, ...], []), "exit_tools must be a tuple"),
        ((), (), (), (), (cast(ExitTool, object()),), "exit_tools must contain only"),
    ],
)
def test_capabilities_data_rejects_invalid_collections(
    deep_skills: tuple[DeepSkill, ...],
    tools: tuple[Tool, ...],
    hds: tuple[HDS, ...],
    exit_skills: tuple[ExitSkill, ...],
    exit_tools: tuple[ExitTool, ...],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        CapabilitiesData(
            deep_skills=deep_skills,
            tools=tools,
            hds=hds,
            exit_skills=exit_skills,
            exit_tools=exit_tools,
        )


def test_capabilities_data_returns_all_tool_types() -> None:
    deep_skill = _deep_skill()
    tool = Tool(lambda: None, name="tool")
    exit_skill = ExitSkill(
        name="finish",
        markdown_path=Path("finish.md"),
        metadata=deep_skill.metadata,
        body="Finish",
        references=(),
    )
    exit_tool = _exit_tool()

    data = CapabilitiesData((deep_skill,), (tool,), (), (exit_skill,), (exit_tool,))

    assert data.get_all_tools() == (deep_skill, tool, exit_skill, exit_tool)


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
        "child": ["parent"],
        "shared": ["parent", "child"],
        "exit_agent": ["finish"],
    }


def test_index_rejects_non_capabilities_data() -> None:
    with pytest.raises(TypeError, match="data must be CapabilitiesData"):
        CapabilitiesIndex.build(cast(CapabilitiesData, object()))
