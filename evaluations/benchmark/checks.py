"""Reusable checks observe outcomes, not a prescribed reasoning trajectory."""

import math
import re
from pathlib import Path
from typing import Any

from evaluations.benchmark.models import Check, Observation, Task
from evaluations.benchmark.probes import run_probes
from evaluations.tool_argument_grading import parse_arguments


def values_equal(actual: Any, expected: Any, tolerance: float = 0) -> bool:
    if type(expected) in (int, float):
        return type(actual) in (int, float) and math.isclose(
            actual, expected, rel_tol=0, abs_tol=tolerance
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(
                values_equal(actual[key], value, tolerance)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                values_equal(left, right, tolerance)
                for left, right in zip(actual, expected)
            )
        )
    return type(actual) is type(expected) and actual == expected


def check_answer(
    task: Task, observation: Observation, workspace: Path, parameters: dict[str, Any]
) -> Check:
    conditions = {
        "contains:" + token: token in observation.answer
        for token in parameters.get("contains", ())
    }
    conditions.update(
        {
            "omits:" + token: token not in observation.answer
            for token in parameters.get("omits", ())
        }
    )
    if "exact" in parameters:
        conditions["exact_answer"] = observation.answer.strip() == parameters["exact"]
    if "number" in parameters:
        numbers = re.findall(r"(?<![\w.-])-?\d+(?:\.\d+)?", observation.answer)
        conditions["requested_number"] = any(
            math.isclose(float(value), float(parameters["number"]), abs_tol=1e-8)
            for value in numbers
        )
    if "max_words" in parameters:
        conditions["word_limit"] = (
            len(observation.answer.split()) <= parameters["max_words"]
        )
    if not conditions:
        raise ValueError("Answer checker requires at least one condition")
    return Check(100 * sum(conditions.values()) / len(conditions), conditions)


def check_json(
    task: Task, observation: Observation, workspace: Path, parameters: dict[str, Any]
) -> Check:
    source = (
        observation.artifacts.get(parameters["artifact"])
        if "artifact" in parameters
        else observation.answer
    )
    try:
        actual = parse_arguments(source or "")
    except (ValueError, TypeError) as error:
        return Check(0, {"valid_json": False}, str(error))
    expected = parameters.get("expected", parameters.get("expected_subset"))
    if expected is None:
        raise ValueError("JSON checks require expected or expected_subset")
    selected = (
        {key: actual[key] for key in expected if key in actual}
        if "expected_subset" in parameters
        else actual
    )
    return Check(
        100 if values_equal(selected, expected) else 0,
        {"actual": actual, "expected": expected},
    )


def check_calls(
    task: Task, observation: Observation, workspace: Path, parameters: dict[str, Any]
) -> Check:
    names = [item["name"] for item in observation.calls]
    conditions = {}
    if "exact_order" in parameters:
        conditions["requested_call_order"] = names == parameters["exact_order"]
    for name, count in parameters.get("counts", {}).items():
        conditions["count:" + name] = names.count(name) == count
    for name in parameters.get("forbidden", ()):
        conditions["forbidden:" + name] = name not in names
    if not conditions:
        raise ValueError("Call checker requires at least one condition")
    return Check(
        100 * sum(conditions.values()) / len(conditions),
        {"checks": conditions, "calls": names},
    )


def check_events(
    task: Task, observation: Observation, workspace: Path, parameters: dict[str, Any]
) -> Check:
    count = sum(event["kind"] == parameters["kind"] for event in observation.events)
    return Check(
        100 if count >= parameters.get("minimum", 1) else 0,
        {"event": parameters["kind"], "count": count},
    )


def check_unchanged(
    task: Task, observation: Observation, workspace: Path, parameters: dict[str, Any]
) -> Check:
    actual = {
        str(path.relative_to(workspace)): path.read_text(encoding="utf-8")
        for path in workspace.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and ".exmekh_lib" not in path.relative_to(workspace).parts
    }
    return Check(
        100 if actual == task.files else 0,
        {
            "expected_paths": sorted(task.files),
            "actual_paths": sorted(actual),
            "unchanged": actual == task.files,
        },
    )


def check_python(
    task: Task, observation: Observation, workspace: Path, parameters: dict[str, Any]
) -> Check:
    if "expected" in parameters and len(parameters["expected"]) != len(
        parameters["inputs"]
    ):
        raise ValueError("Expected results must match the number of program inputs")
    batch = run_probes(workspace, parameters)
    if batch.infrastructure_error:
        return Check(None, batch.evidence(), batch.infrastructure_error)
    if "expected" in parameters:
        passing = [
            item.succeeded
            and values_equal(
                item.value, expected, parameters.get("absolute_tolerance", 0)
            )
            for item, expected in zip(
                batch.results, parameters["expected"], strict=True
            )
        ]
    else:
        passing = [item.succeeded for item in batch.results]
    return Check(100 * sum(passing) / len(passing), batch.evidence())


def _grid(value: Any, size: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == size
        and all(
            isinstance(row, list)
            and len(row) == size
            and all(type(cell) in (int, float) and math.isfinite(cell) for cell in row)
            for row in value
        )
    )


def check_python_raises(
    task: Task, observation: Observation, workspace: Path, parameters: dict[str, Any]
) -> Check:
    batch = run_probes(workspace, parameters, mode="raises")
    if batch.infrastructure_error:
        return Check(None, batch.evidence(), batch.infrastructure_error)
    passing = [
        item.succeeded
        and isinstance(item.value, dict)
        and item.value.get("raised") is True
        and (
            not parameters.get("exceptions")
            or item.value.get("exception") in parameters["exceptions"]
        )
        for item in batch.results
    ]
    return Check(
        100 * sum(passing) / len(parameters["inputs"]),
        batch.evidence(),
    )


def check_fluid(
    task: Task, observation: Observation, workspace: Path, parameters: dict[str, Any]
) -> Check:
    batch = run_probes(workspace, parameters)
    if batch.infrastructure_error:
        return Check(None, batch.evidence(), batch.infrastructure_error)
    conditions = {
        f"{name}:{index}": False
        for index in range(len(parameters["inputs"]))
        for name in (
            "finite_shape",
            "nonnegative_density",
            "nonzero_initial_mass",
            "closed_walls",
        )
    }
    conditions.update(bounded_mass_change=False, nontrivial_evolution=False)
    grids = []
    for index, item in enumerate(batch.results):
        value, size = item.value, item.arguments[0]
        valid = (
            item.succeeded
            and isinstance(value, dict)
            and all(
                _grid(value.get(key), size)
                for key in ("density", "velocity_x", "velocity_y")
            )
        )
        conditions[f"finite_shape:{index}"] = valid
        if valid:
            density, vx, vy = value["density"], value["velocity_x"], value["velocity_y"]
            conditions[f"nonnegative_density:{index}"] = (
                min(cell for row in density for cell in row) >= 0
            )
            conditions[f"nonzero_initial_mass:{index}"] = sum(map(sum, density)) > 0
            conditions[f"closed_walls:{index}"] = all(
                abs(vx[row][col]) < 1e-6 for row in range(size) for col in (0, size - 1)
            ) and all(
                abs(vy[row][col]) < 1e-6 for row in (0, size - 1) for col in range(size)
            )
            grids.append(value)
    conditions["all_probes_completed"] = all(item.succeeded for item in batch.results)
    if len(grids) == len(parameters["inputs"]) and len(grids) >= 2:
        initial, final = grids[0]["density"], grids[1]["density"]
        mass_initial, mass_final = sum(map(sum, initial)), sum(map(sum, final))
        conditions["bounded_mass_change"] = (
            mass_initial > 0 and abs(mass_final - mass_initial) / mass_initial < 0.4
        )
        conditions["nontrivial_evolution"] = any(
            abs(left - right) > 1e-7
            for row_a, row_b in zip(initial, final)
            for left, right in zip(row_a, row_b)
        )
    return Check(
        100 * sum(conditions.values()) / len(conditions),
        {"checks": conditions, **batch.evidence()},
    )
