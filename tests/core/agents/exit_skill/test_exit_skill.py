from pathlib import Path
from typing import cast

import pytest

from qs.core.agents.deep_skill.metadata import DeepSkillMetadata
from qs.core.agents.exit_skill.exit_skill import CalledExitSkill, ExitSkill
from qs.core.models.assistant_message import FunctionCall, ToolCall


def _metadata() -> DeepSkillMetadata:
    return DeepSkillMetadata(
        description="Finish",
        sub_skills=(),
        sub_tools=(),
        inherits_parent_tools=False,
    )


def _exit_skill(tmp_path: Path) -> ExitSkill:
    return ExitSkill(
        name="finish",
        markdown_path=tmp_path / "finish.md",
        metadata=_metadata(),
        body="  Finish now  ",
        references=(),
    )


def test_renders_exit_skill_content(tmp_path: Path) -> None:
    exit_skill = _exit_skill(tmp_path)

    assert exit_skill.get_body() == "Finish now"
    assert exit_skill.render_prompt_section() == ("<exit-skill>Finish now</exit-skill>")


def test_rejects_invalid_metadata(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="metadata must be DeepSkillMetadata"):
        ExitSkill(
            name="finish",
            markdown_path=tmp_path / "finish.md",
            metadata=cast(DeepSkillMetadata, object()),
            body="Finish",
            references=(),
        )


def test_called_exit_skill_validates_types(tmp_path: Path) -> None:
    skill = _exit_skill(tmp_path)
    tool_call = ToolCall("call-1", FunctionCall("finish", {}))
    called = CalledExitSkill(skill, tool_call)

    assert called.exit_skill is skill
    assert called.tool_call is tool_call

    with pytest.raises(TypeError, match="exit_skill must be an ExitSkill"):
        CalledExitSkill(cast(ExitSkill, object()), tool_call)
    with pytest.raises(TypeError, match="tool_call must be a ToolCall"):
        CalledExitSkill(skill, cast(ToolCall, object()))
