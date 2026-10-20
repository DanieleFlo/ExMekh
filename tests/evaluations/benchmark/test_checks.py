"""Outcome checks reject plausible-looking but unusable deliverables."""

import pytest

from evaluations.benchmark.checks import (
    check_answer,
    check_fluid,
    check_json,
    check_python,
    values_equal,
)
from evaluations.benchmark.models import Criterion, Observation, Task


def example():
    return Task(
        id="example",
        domain="any",
        prompt="Return a result",
        criteria=(
            Criterion(
                id="result",
                description="Correct result",
                verification="exact",
                verification_reason="Observable",
                checker="answer",
            ),
        ),
    )


def test_exact_answer_does_not_accept_contradictions(tmp_path):
    assert (
        check_answer(
            example(), Observation("completed", "42 or 41"), tmp_path, {"exact": "42"}
        ).score
        == 0
    )
    assert (
        check_answer(
            example(), Observation("completed", "\n42\n"), tmp_path, {"exact": "42"}
        ).score
        == 100
    )


@pytest.mark.parametrize("source", ['{"x": 1, "x": 2}', '{"x": NaN}', '{"x": true}'])
def test_json_checks_do_not_repair_duplicate_keys_or_coerce_booleans(tmp_path, source):
    assert (
        check_json(
            example(),
            Observation("completed", source),
            tmp_path,
            {"expected": {"x": 1}},
        ).score
        == 0
    )


def test_numeric_comparison_uses_an_explicit_tolerance():
    assert values_equal([1.00001], [1], 0.001)
    assert not values_equal([True], [1])


def test_json_subset_requires_present_keys_even_when_the_expected_value_is_null(
    tmp_path,
):
    result = check_json(
        example(),
        Observation("completed", "{}"),
        tmp_path,
        {"expected_subset": {"required": None}},
    )
    assert result.score == 0


def test_python_probes_evaluate_all_cases_after_an_independent_failure(tmp_path):
    (tmp_path / "program.py").write_text(
        "def run(value):\n    if value == 0:\n        raise ValueError('bad')\n    return value * 2\n"
    )
    result = check_python(
        example(),
        Observation("completed"),
        tmp_path,
        {
            "module": "program.py",
            "function": "run",
            "inputs": [[0], [2]],
            "expected": [0, 4],
        },
    )
    if result.score is None:
        pytest.skip(result.error)
    assert result.score == 50
    assert len(result.evidence["executions"]) == 2


def test_stationary_fake_fluid_fails_the_evolution_check(tmp_path):
    (tmp_path / "fluid.py").write_text(
        "def simulate(size, steps):\n    return {name: [[1 if name == 'density' else 0 for x in range(size)] for y in range(size)] for name in ('density', 'velocity_x', 'velocity_y')}\n"
    )
    result = check_fluid(
        example(),
        Observation("completed"),
        tmp_path,
        {"module": "fluid.py", "function": "simulate", "inputs": [[4, 0], [4, 5]]},
    )
    if result.score is None:
        pytest.skip(result.error)
    assert result.score < 100
    assert result.evidence["checks"]["nontrivial_evolution"] is False
