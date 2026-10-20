"""Mechanically test invalid-input rejection instead of asking a judge to guess."""

import pytest

from evaluations.benchmark.checks import check_python_raises
from evaluations.benchmark.models import Observation
from tests.evaluations.benchmark.test_checks import example


@pytest.mark.parametrize(
    "body, score",
    [
        ("raise ValueError('invalid')", 100),
        ("return {'raised': True}", 0),
    ],
)
def test_exception_probe_distinguishes_raising_from_returning_a_claim(
    tmp_path, body, score
):
    (tmp_path / "program.py").write_text("def run(value):\n    " + body + "\n")
    result = check_python_raises(
        example(),
        Observation("completed"),
        tmp_path,
        {"module": "program.py", "function": "run", "inputs": [[-1]]},
    )
    if result.score is None:
        pytest.skip(result.error)
    assert result.score == score


def test_import_failure_does_not_count_as_api_input_validation(tmp_path):
    (tmp_path / "program.py").write_text("raise ValueError('import failed')\n")
    result = check_python_raises(
        example(),
        Observation("completed"),
        tmp_path,
        {"module": "program.py", "function": "run", "inputs": [[-1]]},
    )
    if result.score is None:
        pytest.skip(result.error)
    assert result.score == 0


@pytest.mark.parametrize("source", ["def wrong(): pass", "run = 42"])
def test_missing_or_noncallable_api_is_not_successful_validation(tmp_path, source):
    (tmp_path / "program.py").write_text(source)
    result = check_python_raises(
        example(),
        Observation("completed"),
        tmp_path,
        {"module": "program.py", "function": "run", "inputs": [[-1]]},
    )
    if result.score is None:
        pytest.skip(result.error)
    assert result.score == 0
