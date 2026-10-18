import sys
from pathlib import Path

import pytest

from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.capability.markdown_capability import MarkdownCapability
from exmekh_lib.core.agents.deep_skill.deep_skills import DeepSkills
from exmekh_lib.core.agents.hds.hds import HDS

_DYNAMIC_PACKAGE_PREFIX = "_exmekh_lib_deep_tool_"


def _dynamic_tool_modules() -> set[str]:
    return {
        module_name
        for module_name in sys.modules
        if module_name.startswith(_DYNAMIC_PACKAGE_PREFIX)
    }


def _write_skill(
    folder_path: Path,
    name: str,
    *,
    body: str = "Skill body",
    sub_skills: tuple[str, ...] = (),
    sub_tools: tuple[str, ...] = (),
) -> None:
    metadata_collections = _metadata_collection(
        "sub-skills", sub_skills
    ) + _metadata_collection("sub-tools", sub_tools)
    (folder_path / f"{name}.md").write_text(
        "\n".join(
            (
                "---",
                f"description: {name} description",
                *metadata_collections,
                "---",
                body,
            )
        )
    )


def _write_exit_tool(capabilities_path: Path, name: str) -> None:
    exit_tool_path = capabilities_path / "exit-tool" / name
    if exit_tool_path.exists():
        return
    exit_tool_path.mkdir(parents=True)
    (exit_tool_path / "instructions.md").write_text(f"{name} instructions")
    (exit_tool_path / "main.py").write_text(
        "from exmekh_lib.core.agents.exit_tool.data import ExitToolData\n"
        "from exmekh_lib.core.agents.exit_tool.decorator import exit_tool\n\n"
        f"@exit_tool(instructions_path='instructions.md', name={name!r}, description='Exit agent')\n"
        "def exit_agent(text: str, items: list[str]) -> ExitToolData:\n"
        "    return ExitToolData(text=text, items=items)\n"
    )


def _write_hds(
    folder_path: Path,
    name: str,
    *,
    body: str = "HDS body",
    sub_skills: tuple[str, ...] = (),
    sub_tools: tuple[str, ...] = (),
    exit_tool: str = "finish",
) -> None:
    _write_exit_tool(folder_path.parent, exit_tool)
    metadata_collections = _metadata_collection(
        "sub-skills", sub_skills
    ) + _metadata_collection("sub-tools", sub_tools)
    (folder_path / f"{name}.md").write_text(
        "\n".join(
            (
                "---",
                f"description: {name} description",
                f"exit-tool: {exit_tool}",
                *metadata_collections,
                "---",
                body,
            )
        )
    )


def _metadata_collection(
    field: str,
    values: tuple[str, ...],
) -> tuple[str, ...]:
    if not values:
        return ()

    return (
        f"{field}:",
        *(f"  - {value}" for value in values),
    )


def _write_tool(
    capabilities_path: Path,
    name: str,
    *,
    description: str = "Tool description.",
) -> None:
    tool_path = capabilities_path / "tools" / name
    tool_path.mkdir(parents=True)
    (tool_path / "main.py").write_text(
        "from exmekh_lib.core.agents.tools.decorator import deep_tool\n\n"
        f"@deep_tool(description={description!r})\n"
        "def run(work_root) -> str:\n"
        "    del work_root\n"
        "    return 'result'\n"
    )


def test_load_validates_references_after_loading_all_deep_skills(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir(parents=True)
    _write_skill(deep_skill_path, "first", body="Use {{!second}}")
    _write_skill(deep_skill_path, "second")

    data = Capabilities([capabilities_path]).load(work_root=tmp_path)

    assert tuple(skill.name for skill in data.deep_skills) == (
        "first",
        "second",
    )
    assert data.deep_skills[0].references == ("second",)
    assert not hasattr(data.deep_skills[0].metadata, "exit_tool")
    assert data.exit_tools == ()


def test_load_validates_hds_exit_tool_reference(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    _write_hds(hds_path, "instructions", exit_tool="custom-exit")

    data = Capabilities([capabilities_path]).load(work_root=tmp_path)

    assert data.hds[0].metadata.exit_tool == "custom-exit"
    assert data.exit_tools[0].name == "custom-exit"


def test_load_rejects_missing_hds_exit_tool(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    _write_hds(hds_path, "instructions")
    hds_file = hds_path / "instructions.md"
    hds_file.write_text(
        hds_file.read_text().replace("exit-tool: finish", "exit-tool: missing")
    )

    with pytest.raises(
        ValueError,
        match="HDS 'instructions' has invalid exit-tool: missing",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_free_reference_can_resolve_exit_tool(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    _write_hds(hds_path, "instructions", body="Use {{!finish}}")
    _write_exit_tool(capabilities_path, "finish")

    data = Capabilities([capabilities_path]).load(work_root=tmp_path)

    assert data.hds[0].references == ("finish",)


def test_loads_hds_with_exit_tool_and_returns_only_body(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    _write_hds(hds_path, "instructions", body="Instruction body")

    data = Capabilities([capabilities_path]).load(work_root=tmp_path)

    assert tuple(hds.name for hds in data.hds) == ("instructions",)
    assert data.hds[0].description == "instructions description"
    assert data.hds[0].metadata.exit_tool == "finish"
    assert data.hds[0].tool() == "Instruction body"
    assert not hasattr(data.hds[0], "catalog")


def test_load_validates_hds_sub_resources_after_loading_everything(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir()
    _write_hds(
        hds_path,
        "instructions",
        body="Use {{!child}} and {{!search}}",
        sub_skills=("child",),
        sub_tools=("search",),
    )
    _write_skill(deep_skill_path, "child")
    _write_tool(capabilities_path, "search")

    data = Capabilities([capabilities_path]).load(work_root=tmp_path)

    assert data.hds[0].references == ("child", "search")
    assert data.hds[0].tool() == "Use child and search"


def test_markdown_capability_siblings_can_reference_each_other(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir()
    _write_hds(hds_path, "instructions")
    _write_skill(
        deep_skill_path,
        "parent",
        body="Use {{!instructions}}",
    )

    data = Capabilities([capabilities_path]).load(work_root=tmp_path)

    assert isinstance(data.deep_skills[0], MarkdownCapability)
    assert not isinstance(data.deep_skills[0], HDS)
    assert isinstance(data.hds[0], MarkdownCapability)
    assert data.deep_skills[0].references == ("instructions",)


def test_load_rejects_unknown_hds_reference(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    _write_hds(hds_path, "instructions", body="Use {{!missing}}")

    with pytest.raises(
        ValueError,
        match="HDS 'instructions' has invalid references: missing",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_load_rejects_hds_sub_skill_that_resolves_to_tool(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    _write_hds(hds_path, "instructions", sub_skills=("search",))
    _write_tool(capabilities_path, "search")

    with pytest.raises(
        ValueError,
        match="HDS 'instructions' has invalid sub-skills: search",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_load_rejects_hds_sub_tool_that_resolves_to_skill(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir()
    _write_hds(hds_path, "instructions", sub_tools=("child",))
    _write_skill(deep_skill_path, "child")

    with pytest.raises(
        ValueError,
        match="HDS 'instructions' has invalid sub-tools: child",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_load_builds_deep_skill_tool_with_sub_skill_catalog(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir(parents=True)
    _write_skill(
        deep_skill_path,
        "parent",
        body="Parent body",
        sub_skills=("child",),
    )
    _write_skill(deep_skill_path, "child")

    raw_parent = DeepSkills([deep_skill_path]).load()[1]
    parent = Capabilities([capabilities_path]).load(work_root=tmp_path).deep_skills[1]

    assert raw_parent.catalog is None
    assert parent.catalog is not None
    assert parent.name == "parent"
    assert parent.description == "parent description"
    assert parent.tool() == (
        "Parent body\n\n<catalog>\n  <sub-skills>\n"
        '    <skill name="child">child description</skill>\n'
        "  </sub-skills>\n</catalog>"
    )


def test_load_rejects_unknown_deep_skill_references(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir(parents=True)
    _write_skill(deep_skill_path, "first", body="Use {{!missing}}")

    with pytest.raises(
        ValueError,
        match="Deep skill 'first' has invalid references: missing",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_load_unloads_tools_when_reference_validation_fails(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir(parents=True)
    _write_skill(deep_skill_path, "first", body="Use {{!missing}}")
    _write_tool(capabilities_path, "search")
    modules_before_load = _dynamic_tool_modules()

    with pytest.raises(ValueError, match="invalid references"):
        Capabilities([capabilities_path]).load(work_root=tmp_path)

    assert _dynamic_tool_modules() == modules_before_load


def test_load_resolves_tool_references_and_builds_sub_tool_catalog(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir(parents=True)
    _write_skill(
        deep_skill_path,
        "parent",
        body="Use {{!search}}",
        sub_tools=("search",),
    )

    _write_tool(
        capabilities_path,
        "search",
        description="Search indexed documents.",
    )

    data = Capabilities([capabilities_path]).load(work_root=tmp_path)

    assert tuple(tool.name for tool in data.tools) == ("search",)
    assert data.deep_skills[0].tool() == (
        "Use search\n\n<catalog>\n  <sub-tools>\n"
        '    <tool name="search">Search indexed documents.</tool>\n'
        "  </sub-tools>\n</catalog>"
    )


def test_load_rejects_sub_skill_that_resolves_to_tool(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir(parents=True)
    _write_skill(
        deep_skill_path,
        "parent",
        sub_skills=("search",),
    )
    _write_tool(capabilities_path, "search")

    with pytest.raises(
        ValueError,
        match="Deep skill 'parent' has invalid sub-skills: search",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_load_rejects_sub_tool_that_resolves_to_deep_skill(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir(parents=True)
    _write_skill(
        deep_skill_path,
        "parent",
        sub_tools=("child",),
    )
    _write_skill(deep_skill_path, "child")

    with pytest.raises(
        ValueError,
        match="Deep skill 'parent' has invalid sub-tools: child",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_load_rejects_duplicate_deep_skill_names_across_capabilities(
    tmp_path: Path,
) -> None:
    capabilities_paths: list[Path] = []

    for parent_name in ("first", "second"):
        capabilities_path = tmp_path / parent_name / "capabilities"
        deep_skill_path = capabilities_path / "deep-skill"
        deep_skill_path.mkdir(parents=True)
        _write_skill(deep_skill_path, "shared")
        capabilities_paths.append(capabilities_path)

    with pytest.raises(
        ValueError,
        match="Duplicate deep-skill name: 'shared'",
    ):
        Capabilities(capabilities_paths).load(work_root=tmp_path)


def test_load_rejects_name_shared_by_skill_and_tool(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir(parents=True)
    _write_skill(deep_skill_path, "search")
    _write_tool(capabilities_path, "search")

    with pytest.raises(
        ValueError,
        match="Names shared by deep skills and tools: search",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_load_rejects_duplicate_hds_names_across_capabilities(
    tmp_path: Path,
) -> None:
    capabilities_paths: list[Path] = []

    for parent_name in ("first", "second"):
        capabilities_path = tmp_path / parent_name / "capabilities"
        hds_path = capabilities_path / "hds"
        hds_path.mkdir(parents=True)
        _write_hds(hds_path, "shared")
        capabilities_paths.append(capabilities_path)

    with pytest.raises(ValueError, match="Duplicate HDS name: 'shared'"):
        Capabilities(capabilities_paths).load(work_root=tmp_path)


def test_load_rejects_name_shared_by_hds_and_tool(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    _write_hds(hds_path, "search")
    _write_tool(capabilities_path, "search")

    with pytest.raises(
        ValueError,
        match="Names shared by HDS and other capabilities: search",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_load_rejects_name_shared_by_hds_and_deep_skill(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    deep_skill_path = capabilities_path / "deep-skill"
    deep_skill_path.mkdir()
    _write_hds(hds_path, "shared")
    _write_skill(deep_skill_path, "shared")

    with pytest.raises(
        ValueError,
        match="Names shared by HDS and other capabilities: shared",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


def test_load_rejects_duplicate_exit_tool_names_across_capabilities(
    tmp_path: Path,
) -> None:
    capabilities_paths: list[Path] = []

    for parent_name in ("first", "second"):
        capabilities_path = tmp_path / parent_name / "capabilities"
        capabilities_path.mkdir(parents=True)
        _write_exit_tool(capabilities_path, "finish")
        capabilities_paths.append(capabilities_path)

    with pytest.raises(ValueError, match="Duplicate exit-tool name: 'finish'"):
        Capabilities(capabilities_paths).load(work_root=tmp_path)


def test_load_rejects_name_shared_by_exit_tool_and_tool(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.mkdir()
    _write_exit_tool(capabilities_path, "finish")
    _write_tool(capabilities_path, "finish")

    with pytest.raises(
        ValueError,
        match="Names shared by exit tools and other capabilities: finish",
    ):
        Capabilities([capabilities_path]).load(work_root=tmp_path)


@pytest.mark.parametrize("kind", ["deep-skill", "hds"])
def test_exit_tool_name_collisions_include_markdown_capabilities(tmp_path, kind):
    root = tmp_path / "capabilities"
    folder = root / kind
    folder.mkdir(parents=True)
    _write_exit_tool(root, "finish")
    if kind == "hds":
        _write_hds(folder, "finish")
    else:
        _write_skill(folder, "finish")
    with pytest.raises(ValueError, match="Names shared by exit tools"):
        Capabilities([root]).load(work_root=tmp_path)


def test_exit_tool_collision_uses_decorated_name_not_folder(tmp_path):
    root = tmp_path / "capabilities"
    _write_exit_tool(root, "first-folder")
    _write_exit_tool(root, "second-folder")
    for folder in ("first-folder", "second-folder"):
        entrypoint = root / "exit-tool" / folder / "main.py"
        entrypoint.write_text(
            entrypoint.read_text().replace(f"name='{folder}'", "name='shared-exit'")
        )
    with pytest.raises(ValueError, match="Duplicate exit-tool name: 'shared-exit'"):
        Capabilities([root]).load(work_root=tmp_path)


def test_multiple_hds_can_share_one_exit_tool(tmp_path):
    root = tmp_path / "capabilities"
    hds = root / "hds"
    hds.mkdir(parents=True)
    _write_hds(hds, "first")
    _write_hds(hds, "second")
    data = Capabilities([root]).load(work_root=tmp_path)
    assert len(data.exit_tools) == 1
    assert all(
        data.get_exit_tool(item.get_exit_tool_name()) is data.exit_tools[0]
        for item in data.hds
    )


@pytest.mark.parametrize("kind", ["tools", "deep-skill"])
def test_hds_exit_reference_must_resolve_to_an_exit_tool(tmp_path, kind):
    root = tmp_path / "capabilities"
    hds = root / "hds"
    hds.mkdir(parents=True)
    _write_hds(hds, "main")
    if kind == "tools":
        _write_tool(root, "ordinary")
    else:
        (root / kind).mkdir()
        _write_skill(root / kind, "ordinary")
    file = hds / "main.md"
    file.write_text(
        file.read_text().replace("exit-tool: finish", "exit-tool: ordinary")
    )
    with pytest.raises(ValueError, match="invalid exit-tool: ordinary"):
        Capabilities([root]).load(work_root=tmp_path)


def test_validates_exit_instruction_references_after_all_roots_are_loaded(tmp_path):
    first = tmp_path / "first" / "capabilities"
    second = tmp_path / "second" / "capabilities"
    _write_exit_tool(first, "finish")
    instructions = first / "exit-tool" / "finish" / "instructions.md"
    instructions.write_text("Use {{!later}} then {{!finish}} and {{!later}}")
    _write_tool(second, "later")
    data = Capabilities([first, second]).load(work_root=tmp_path)
    assert data.exit_tools[0].references == ("later", "finish")
    assert data.exit_tools[0].instructions == "Use later then finish and later"


def test_missing_exit_instruction_reference_unloads_all_modules(tmp_path):
    root = tmp_path / "capabilities"
    _write_tool(root, "ordinary")
    _write_exit_tool(root, "finish")
    (root / "exit-tool" / "finish" / "instructions.md").write_text("Use {{!missing}}")
    before = _dynamic_tool_modules()
    with pytest.raises(
        ValueError, match="Exit tool 'finish' has invalid references: missing"
    ):
        Capabilities([root]).load(work_root=tmp_path)
    assert _dynamic_tool_modules() == before


def test_rejects_legacy_exit_skill_directory_with_migration_hint(tmp_path):
    root = tmp_path / "capabilities"
    (root / "exit-skill").mkdir(parents=True)
    with pytest.raises(ValueError, match="move instructions into exit-tool"):
        Capabilities([root]).load(work_root=tmp_path)
