from pathlib import Path

import pytest

from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.deep_skill.execution import DeepSkillExecutionTracker
from qs.core.agents.deep_skill.metadata import DeepSkillMetadata
from qs.core.models.assistant_message import AssistantMessage


def _skill(name: str) -> DeepSkill:
    return DeepSkill(
        name=name,
        markdown_path=Path(f"{name}.md"),
        metadata=DeepSkillMetadata(
            description=name,
            sub_skills=(),
            sub_tools=(),
            inherits_parent_tools=False,
        ),
        body=f"{name} body",
        references=(),
    )


def test_tracks_nested_executions_and_copies_skills() -> None:
    parent_skill = _skill("parent")
    child_skill = _skill("child")
    tracker = DeepSkillExecutionTracker()

    parent = tracker.open(parent_skill, "parent-call", 2)
    child = tracker.open(child_skill, "child-call", 4)

    assert tracker.depth == 2
    assert tracker.current is child
    assert tracker.roots == (parent,)
    assert parent.children == [child]
    assert parent.skill is not parent_skill
    assert child.skill is not child_skill


def test_closes_nested_executions_in_reverse_opening_order() -> None:
    tracker = DeepSkillExecutionTracker()
    parent = tracker.open(_skill("parent"), "parent-call", 2)
    child = tracker.open(_skill("child"), "child-call", 4)
    child_history = [AssistantMessage("child transcript")]

    closed_child = tracker.close(
        end_index=6,
        output="child result",
        history=child_history,
    )

    assert closed_child is child
    assert child.end_index == 6
    assert child.output == "child result"
    assert child.history == child_history
    assert tracker.current is parent
    assert tracker.depth == 1


def test_rejects_closing_when_no_execution_is_active() -> None:
    tracker = DeepSkillExecutionTracker()

    assert tracker.depth == 0
    assert tracker.roots == ()
    with pytest.raises(RuntimeError, match="No deep skill execution is active"):
        _ = tracker.current
    with pytest.raises(RuntimeError, match="none is active"):
        tracker.close(end_index=0, output="result", history=[])


def test_rejects_closing_execution_twice() -> None:
    tracker = DeepSkillExecutionTracker()
    execution = tracker.open(_skill("skill"), "call", 0)
    execution.close(end_index=1, output="result", history=[])

    with pytest.raises(RuntimeError, match="already closed"):
        execution.close(end_index=2, output="other", history=[])


def test_serializes_nested_history_tree() -> None:
    tracker = DeepSkillExecutionTracker()
    parent = tracker.open(_skill("parent"), "parent-call", 2)
    child = tracker.open(_skill("child"), "child-call", 4)
    child_message = AssistantMessage("child transcript")
    child.close(
        end_index=5,
        output="child result",
        history=[child_message],
    )

    assert parent.history_tree() == {
        "skill": "parent",
        "output": None,
        "history": [],
        "children": [
            {
                "skill": "child",
                "output": "child result",
                "history": [child_message.message_dump()],
                "children": [],
            }
        ],
    }
