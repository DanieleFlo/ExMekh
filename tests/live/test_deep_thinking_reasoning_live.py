"""Real SymPy and Z3 checks through the bundled deep-thinking tools."""

from pathlib import Path

import pytest

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.capabilities.defaults import BUILTIN_CAPABILITIES_PATH

pytestmark = pytest.mark.live


def _tool(work_root: Path, name: str):
    loaded = Capabilities([BUILTIN_CAPABILITIES_PATH]).load(work_root=work_root)
    return next(tool for tool in loaded.tools if tool.name == name)


def test_live_calculate_solves_and_evaluates_with_sympy(tmp_path: Path) -> None:
    calculate = _tool(tmp_path, "calculate")

    roots = calculate.execute(operation="solve", expression="x**2 - 5*x + 6 = 0")
    assert roots == {"operation": "solve", "result": "[2, 3]"}

    derivative = calculate.execute(
        operation="differentiate", expression="sin(x) + x**3"
    )
    assert derivative["result"] == "3*x**2 + cos(x)"

    determinant = calculate.execute(operation="determinant", matrix=[[1, 2], [3, 4]])
    assert determinant["result"] == "-2"

    numeric = calculate.execute(
        operation="evaluate", expression="sqrt(2)", precision=20
    )
    assert str(numeric["result"]).startswith("1.4142135623730950488")


def test_live_logic_check_proves_modus_ponens(tmp_path: Path) -> None:
    p, q = {"var": "P"}, {"var": "Q"}
    result = _tool(tmp_path, "logic-check").execute(
        variables={"P": "Bool", "Q": "Bool"},
        premises=[{"op": "implies", "left": p, "right": q}, p],
        conclusion=q,
    )
    assert result["status"] == "valid"
    assert result["backend"] == "z3"
    assert result["entailed"] is True


def test_live_logic_consistency_reports_an_unsat_core(tmp_path: Path) -> None:
    x = {"var": "x"}
    premises = [
        {"op": ">", "left": x, "right": {"const": 10}},
        {"op": "<", "left": x, "right": {"const": 5}},
    ]
    result = _tool(tmp_path, "logic-consistency").execute(
        variables={"x": "Real"}, premises=premises
    )
    assert result["status"] == "inconsistent"
    assert result["unsat_core"] == premises


def test_live_logic_counterexample_refutes_affirming_the_consequent(
    tmp_path: Path,
) -> None:
    p, q = {"var": "P"}, {"var": "Q"}
    result = _tool(tmp_path, "logic-counterexample").execute(
        variables={"P": "Bool", "Q": "Bool"},
        premises=[{"op": "implies", "left": p, "right": q}, q],
        conclusion=p,
    )
    assert result["status"] == "found"
    assert result["counterexample"] == {"P": False, "Q": True}
