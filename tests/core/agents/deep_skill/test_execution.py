from pathlib import Path

import pytest

from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.deep_skill.execution import DeepSkillExecutionTracker
from exmekh_lib.core.agents.deep_skill.metadata import DeepSkillMetadata
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference


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
    assert tracker.active_path == "/parent/child"
    assert tracker.current is child
    assert tracker.get_current_start_index() == 4
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
    assert tracker.active_path == "/"
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


def test_failed_closure_preserves_the_active_execution() -> None:
    tracker = DeepSkillExecutionTracker()
    execution = tracker.open(_skill("skill"), "call", 0)
    execution.close(end_index=1, output="result", history=[])

    with pytest.raises(RuntimeError, match="already closed"):
        tracker.close(end_index=2, output="other", history=[])

    assert tracker.current is execution
    assert tracker.depth == 1


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
    assert tracker.history_tree() == [parent.history_tree()]


def test_tracks_tool_outputs_in_active_skills_and_their_ancestors() -> None:
    tracker = DeepSkillExecutionTracker()
    tracker.record_tool_output("root-tool", "root-output")
    parent = tracker.open(_skill("parent"), "parent-call", 0)
    tracker.record_tool_output("work", "parent-output")
    child = tracker.open(_skill("child"), "child-call", 2)
    tracker.record_tool_output("work", "child-output")
    leaf = tracker.open(_skill("leaf"), "leaf-call", 4)
    tracker.record_tool_output("leaf-tool", "leaf-output")
    tracker.close(end_index=5, output="leaf result", history=[])
    tracker.close(end_index=6, output="child result", history=[])
    sibling = tracker.open(_skill("child"), "child-call", 7)
    tracker.record_tool_output("work", "sibling-output")

    parent_output = ToolOutputReference("work", "parent-output")
    child_output = ToolOutputReference("work", "child-output")
    leaf_output = ToolOutputReference("leaf-tool", "leaf-output")
    sibling_output = ToolOutputReference("work", "sibling-output")
    assert parent.used_tools == (
        parent_output,
        child_output,
        leaf_output,
        sibling_output,
    )
    assert child.used_tools == (child_output, leaf_output)
    assert leaf.used_tools == (leaf_output,)
    assert sibling.used_tools == (sibling_output,)


def test_used_tool_references_are_snapshots_and_closed_skills_reject_new_work():
    tracker = DeepSkillExecutionTracker()
    execution = tracker.open(_skill("skill"), "call", 0)
    tracker.record_tool_output("work", "first")
    snapshot = execution.used_tools
    tracker.record_tool_output("work", "second")

    assert snapshot == (ToolOutputReference("work", "first"),)
    assert execution.used_tools == (
        ToolOutputReference("work", "first"),
        ToolOutputReference("work", "second"),
    )
    tracker.close(end_index=1, output="result", history=[])
    with pytest.raises(RuntimeError, match="closed deep skill"):
        execution.record_tool_output(ToolOutputReference("work", "late"))
    with pytest.raises(RuntimeError, match="closed deep skill"):
        execution.add_child(execution)
