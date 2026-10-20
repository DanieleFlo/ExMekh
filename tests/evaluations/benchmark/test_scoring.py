"""Grading policy stays fixed even when judge and executable evidence disagree."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluations.benchmark.models import (
    Check,
    Criterion,
    JudgeMark,
    Observation,
    Suite,
    Task,
)
from evaluations.benchmark.scoring import CheckerRegistry, combine


def criterion(verification="exact", **changes):
    return Criterion(
        id="result",
        description="Deliver the requested result",
        verification=verification,
        verification_reason="Fixed before execution",
        checker=None if verification == "semantic" else "custom",
        **changes,
    )


def task(item):
    return Task(id="example", domain="custom", prompt="Do the task", criteria=(item,))


@pytest.mark.parametrize(
    "verification, expected", [("strong", 84), ("partial", 48), ("semantic", 20)]
)
def test_blend_depends_on_predeclared_verifiability(verification, expected):
    result = combine(
        task(criterion(verification)),
        {"result": Check(100)},
        (JudgeMark("result", 20, ("answer",), "Partially supported"),),
    )
    assert result.quality == pytest.approx(expected)


def test_exact_failure_cannot_be_overridden_by_a_judge():
    definition = task(criterion(required=True, passing_score=100))
    assert combine(definition, {"result": Check(0)}, ()).success is False
    with pytest.raises(ValueError, match="non-exact"):
        combine(
            definition, {"result": Check(0)}, (JudgeMark("result", 100, (), "Claim"),)
        )


def test_missing_judge_does_not_redistribute_its_weight():
    result = combine(task(criterion("partial")), {"result": Check(100)}, ())
    assert result.quality is None
    assert result.success is None
    assert result.criteria[0]["analytic"] == 100


def test_required_failure_remains_failed_with_high_semantic_quality():
    exact = criterion(required=True, passing_score=100)
    semantic = criterion("semantic").model_copy(
        update={"id": "explanation", "importance": 10}
    )
    definition = task(exact).model_copy(update={"criteria": (exact, semantic)})
    result = combine(
        definition,
        {"result": Check(0)},
        (JudgeMark("explanation", 100, ("answer",), "Good"),),
    )
    assert result.quality > 90
    assert result.success is False


def test_missing_optional_evidence_does_not_hide_a_known_required_success():
    required = criterion(required=True, passing_score=100)
    optional = criterion("semantic").model_copy(update={"id": "optional"})
    definition = task(required).model_copy(update={"criteria": (required, optional)})
    result = combine(definition, {"result": Check(100)}, ())
    assert result.quality is None
    assert result.success is True


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -1, 101, True])
def test_invalid_scores_are_rejected(score):
    with pytest.raises(ValueError):
        combine(task(criterion()), {"result": Check(score)}, ())


def test_registry_accepts_new_checks_without_suite_size_assumptions(tmp_path):
    registry = CheckerRegistry()
    registry.register(
        "custom",
        lambda task, observation, workspace, parameters: Check(
            100 if observation.answer == "ok" else 0
        ),
    )
    suite = Suite(name="one", tasks=(task(criterion()),))
    assert (
        registry.evaluate(
            suite.tasks[0], Observation("completed", "ok"), Path(tmp_path)
        )["result"].score
        == 100
    )
    with pytest.raises(ValueError, match="registered"):
        registry.register("custom", lambda *args: Check(0))


def test_task_contract_rejects_inconsistent_verification_and_duplicate_ids():
    with pytest.raises(ValidationError):
        Criterion(
            id="bad",
            description="bad",
            verification="semantic",
            verification_reason="bad",
            checker="custom",
        )
    with pytest.raises(ValidationError):
        Suite(name="duplicate", tasks=(task(criterion()), task(criterion())))
