"""Combine independent evidence without letting a judge override exact checks."""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evaluations.benchmark.models import Assessment, Check, JudgeMark, Observation, Task

type Checker = Callable[[Task, Observation, Path, dict[str, Any]], Check]


class CheckerRegistry:
    def __init__(self) -> None:
        self._checks: dict[str, Checker] = {}

    def register(self, name: str, checker: Checker) -> None:
        if name in self._checks:
            raise ValueError(f"Checker already registered: {name}")
        self._checks[name] = checker

    def validate(self, task: Task) -> None:
        unknown = {
            item.checker for item in task.criteria if item.checker
        } - self._checks.keys()
        if unknown:
            raise ValueError(f"Unknown checkers: {sorted(unknown)}")

    def evaluate(
        self, task: Task, observation: Observation, workspace: Path
    ) -> dict[str, Check]:
        self.validate(task)
        results = {}
        for criterion in task.criteria:
            if criterion.checker:
                result = self._checks[criterion.checker](
                    task, observation, workspace, criterion.parameters
                )
                validate_score(result.score)
                results[criterion.id] = result
        return results


def validate_score(value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 100
    ):
        raise ValueError("Scores must be finite numbers from 0 to 100, or missing")


def combine(
    task: Task, checks: dict[str, Check], marks: tuple[JudgeMark, ...]
) -> Assessment:
    judge = {mark.criterion: mark for mark in marks}
    if len(judge) != len(marks):
        raise ValueError("Duplicate judge criterion")
    allowed = {
        criterion.id for criterion in task.criteria if criterion.analytic_weight < 1
    }
    if judge.keys() - allowed:
        raise ValueError("Judge may only score the requested non-exact criteria")
    rows: list[dict[str, Any]] = []
    for criterion in task.criteria:
        weight = criterion.analytic_weight
        analytic = checks.get(criterion.id)
        mark = judge.get(criterion.id)
        left = analytic.score if analytic else None
        right = mark.score if mark else None
        validate_score(left)
        validate_score(right)
        present = (weight == 0 or left is not None) and (
            weight == 1 or right is not None
        )
        score = weight * (left or 0) + (1 - weight) * (right or 0) if present else None
        rows.append(
            {
                "id": criterion.id,
                "importance": criterion.importance,
                "analytic_weight": weight,
                "judge_weight": 1 - weight,
                "verification_reason": criterion.verification_reason,
                "analytic": left,
                "judge": right,
                "score": score,
                "evidence": analytic.evidence if analytic else {},
                "judge_evidence": mark.evidence if mark else (),
                "rationale": mark.rationale if mark else None,
                "error": analytic.error if analytic else None,
                "required": criterion.required,
                "passing_score": criterion.passing_score,
            }
        )
    complete = all(row["score"] is not None for row in rows)
    quality = (
        sum(row["importance"] * row["score"] for row in rows)
        / sum(row["importance"] for row in rows)
        if complete
        else None
    )
    return Assessment(
        quality,
        required_success(rows),
        tuple(rows),
        None
        if complete
        else "Grading evidence is missing; weights were not redistributed",
    )


def required_success(rows: list[dict[str, Any]]) -> bool | None:
    required = [row for row in rows if row["required"]]
    if any(
        row["score"] is not None and row["score"] < row["passing_score"]
        for row in required
    ):
        return False
    if not required or any(row["score"] is None for row in required):
        return None
    return True
