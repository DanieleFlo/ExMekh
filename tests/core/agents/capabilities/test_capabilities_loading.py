import sys
from pathlib import Path

import pytest

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.capability.markdown_capability import MarkdownCapability
from qs.core.agents.deep_skill.deep_skills import DeepSkills
from qs.core.agents.hds.hds import HDS

_DYNAMIC_PACKAGE_PREFIX = "_qs_deep_tool_"


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


def _write_exit_skill(capabilities_path: Path, name: str) -> None:
    exit_skill_path = capabilities_path / "exit-skill" / f"{name}.md"

    if exit_skill_path.exists():
        return

    exit_skill_path.parent.mkdir(parents=True)
    exit_skill_path.write_text(
        f"---\ndescription: {name} description\n---\n{name} instructions"
    )
    exit_tool_path = capabilities_path / "exit-tool" / name
    exit_tool_path.mkdir(parents=True)
    (exit_tool_path / "main.py").write_text(
        "from qs.core.agents.exit_tool.data import ExitToolData\n"
        "from qs.core.agents.exit_tool.decorator import exit_tool\n\n"
        "@exit_tool(name='exit_agent', description='Exit agent')\n"
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
    exit_skill: str = "finish",
) -> None:
    _write_exit_skill(folder_path.parent, exit_skill)
    metadata_collections = _metadata_collection(
        "sub-skills", sub_skills
    ) + _metadata_collection("sub-tools", sub_tools)
    (folder_path / f"{name}.md").write_text(
        "\n".join(
            (
                "---",
                f"description: {name} description",
                f"exit-skill: {exit_skill}",
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
        "from qs.core.agents.tools.decorator import deep_tool\n\n"
        f"@deep_tool(description={description!r})\n"
        "def run() -> str:\n"
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

    data = Capabilities([capabilities_path]).load()

    assert tuple(skill.name for skill in data.deep_skills) == (
        "first",
        "second",
    )
    assert data.deep_skills[0].references == ("second",)
    assert not hasattr(data.deep_skills[0].metadata, "exit_skill")
    assert data.exit_skills == ()


def test_load_validates_hds_exit_skill_reference(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    _write_hds(hds_path, "instructions", exit_skill="custom-exit")

    data = Capabilities([capabilities_path]).load()

    assert data.hds[0].metadata.exit_skill == "custom-exit"
    assert data.exit_skills[0].name == "custom-exit"


def test_load_rejects_missing_hds_exit_skill(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    _write_hds(hds_path, "instructions")
    exit_skill_path = capabilities_path / "exit-skill" / "finish.md"
    exit_skill_path.unlink()

    with pytest.raises(
        ValueError,
        match="HDS 'instructions' has invalid exit-skill: finish",
    ):
        Capabilities([capabilities_path]).load()


def test_free_reference_can_resolve_exit_skill(tmp_path: Path) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    _write_hds(hds_path, "instructions", body="Use {{!finish}}")
    _write_exit_skill(capabilities_path, "finish")

    data = Capabilities([capabilities_path]).load()

    assert data.hds[0].references == ("finish",)


def test_loads_hds_with_exit_skill_and_returns_only_body(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    hds_path = capabilities_path / "hds"
    hds_path.mkdir(parents=True)
    _write_hds(hds_path, "instructions", body="Instruction body")

    data = Capabilities([capabilities_path]).load()

    assert tuple(hds.name for hds in data.hds) == ("instructions",)
    assert data.hds[0].description == "instructions description"
    assert data.hds[0].metadata.exit_skill == "finish"
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

    data = Capabilities([capabilities_path]).load()

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

    data = Capabilities([capabilities_path]).load()

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
        Capabilities([capabilities_path]).load()


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
        Capabilities([capabilities_path]).load()


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
        Capabilities([capabilities_path]).load()


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
    parent = Capabilities([capabilities_path]).load().deep_skills[1]

    assert raw_parent.catalog is None
    assert parent.catalog is not None
    assert parent.name == "parent"
    assert parent.description == "parent description"
    assert parent.tool() == (
        "Parent body\n\n<catalog><sub-skills>\n"
        '<skill name="child">child description</skill>\n'
        "</sub-skills></catalog>"
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
        Capabilities([capabilities_path]).load()


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
        Capabilities([capabilities_path]).load()

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

    data = Capabilities([capabilities_path]).load()

    assert tuple(tool.name for tool in data.tools) == ("search",)
    assert data.deep_skills[0].tool() == (
        "Use search\n\n<catalog><sub-tools>\n"
        '<tool name="search">Search indexed documents.</tool>\n'
        "</sub-tools></catalog>"
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
        Capabilities([capabilities_path]).load()


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
        Capabilities([capabilities_path]).load()


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
        Capabilities(capabilities_paths).load()


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
        Capabilities([capabilities_path]).load()


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
        Capabilities(capabilities_paths).load()


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
        Capabilities([capabilities_path]).load()


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
        Capabilities([capabilities_path]).load()


def test_load_rejects_duplicate_exit_skill_names_across_capabilities(
    tmp_path: Path,
) -> None:
    capabilities_paths: list[Path] = []

    for parent_name in ("first", "second"):
        capabilities_path = tmp_path / parent_name / "capabilities"
        capabilities_path.mkdir(parents=True)
        _write_exit_skill(capabilities_path, "finish")
        capabilities_paths.append(capabilities_path)

    with pytest.raises(ValueError, match="Duplicate exit-skill name: 'finish'"):
        Capabilities(capabilities_paths).load()


def test_load_rejects_name_shared_by_exit_skill_and_tool(
    tmp_path: Path,
) -> None:
    capabilities_path = tmp_path / "capabilities"
    capabilities_path.mkdir()
    _write_exit_skill(capabilities_path, "finish")
    _write_tool(capabilities_path, "finish")

    with pytest.raises(
        ValueError,
        match="Names shared by exit skills and other capabilities: finish",
    ):
        Capabilities([capabilities_path]).load()
