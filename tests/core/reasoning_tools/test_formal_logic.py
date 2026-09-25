"""Checks for formal Z3 reasoning and its bounded JSON formula language."""

from types import SimpleNamespace

import pytest

from qs.core.reasoning_tools import formal_logic
from qs.core.reasoning_tools.formal_logic import (
    FormalLogicError,
    check_logic,
    logic_tool_result,
)


def _var(name: str) -> dict[str, str]:
    return {"var": name}


def _const(value: int | bool) -> dict[str, int | bool]:
    return {"const": value}


def _binary(op: str, left: object, right: object) -> dict[str, object]:
    return {"op": op, "left": left, "right": right}


def test_modus_ponens_is_valid_but_affirming_the_consequent_is_not() -> None:
    p, q = _var("P"), _var("Q")
    implication = _binary("implies", p, q)

    valid = check_logic("entailment", {"P": "Bool", "Q": "Bool"}, [implication, p], q)
    assert valid["status"] == "valid"
    assert valid["entailed"] is True
    assert valid["solver_result"] == "unsat"

    invalid = check_logic(
        "counterexample", {"P": "Bool", "Q": "Bool"}, [implication, q], p
    )
    assert invalid["status"] == "found"
    assert invalid["counterexample"] == {"P": False, "Q": True}


def test_inconsistent_premises_are_not_reported_as_valid_entailment() -> None:
    x = _var("x")
    premises = [_binary(">", x, _const(10)), _binary("<", x, _const(5))]
    consistent = check_logic("consistency", {"x": "Real"}, premises)
    assert consistent["status"] == "inconsistent"
    assert consistent["unsat_core"] == premises

    entailment = check_logic("entailment", {"x": "Real"}, premises, _const(True))
    assert entailment["status"] == "inconsistent_premises"
    assert "entailed" not in entailment


def test_linear_arithmetic_entailment_and_countermodel() -> None:
    x, y = _var("x"), _var("y")
    premises = [
        _binary(">", x, _const(10)),
        _binary(">", y, x),
        _binary("<", y, _const(20)),
    ]
    result = check_logic(
        "entailment", {"x": "Real", "y": "Real"}, premises, _binary("<", x, _const(20))
    )
    assert result["status"] == "valid"

    weaker = check_logic(
        "entailment",
        {"x": "Int"},
        [_binary(">", x, _const(0))],
        _binary(">", x, _const(10)),
    )
    assert weaker["status"] == "invalid"
    assert 0 < int(weaker["counterexample"]["x"]) <= 10


def test_boolean_composition_and_linear_numeric_operations() -> None:
    p, q = _var("P"), _var("Q")
    de_morgan = _binary(
        "iff",
        {"op": "not", "arg": {"op": "and", "args": [p, q]}},
        {"op": "or", "args": [{"op": "not", "arg": p}, {"op": "not", "arg": q}]},
    )
    assert (
        check_logic("entailment", {"P": "Bool", "Q": "Bool"}, [], de_morgan)["status"]
        == "valid"
    )

    x = _var("x")
    doubled = _binary("*", _const(2), x)
    assert (
        check_logic(
            "entailment",
            {"x": "Real"},
            [_binary("=", x, _const(3))],
            _binary("=", doubled, _const(6)),
        )["status"]
        == "valid"
    )


def test_unsupported_or_unsafe_formulas_return_stable_errors() -> None:
    x = _var("x")
    nonlinear = logic_tool_result(
        "entailment",
        {"x": "Real"},
        [],
        _binary(">", _binary("*", x, x), _const(0)),
    )
    assert nonlinear["error_kind"] == "unsupported_theory"

    symbolic_divisor = logic_tool_result(
        "entailment",
        {"x": "Real"},
        [],
        _binary(">", _binary("/", _const(1), x), _const(0)),
    )
    assert symbolic_divisor["error_kind"] == "unsupported_theory"

    undeclared = logic_tool_result("consistency", {}, [x])
    assert undeclared["error_kind"] == "parse_error"

    wrong_type = logic_tool_result(
        "consistency", {"P": "Bool"}, [_binary(">", _var("P"), _const(0))]
    )
    assert wrong_type["error_kind"] == "type_error"

    code_like = logic_tool_result(
        "consistency", {"P": "Bool"}, [{"op": "__import__", "arg": _var("P")}]
    )
    assert code_like["error_kind"] == "unsupported_operation"


def test_input_limits_are_enforced_before_solver_construction() -> None:
    with pytest.raises(FormalLogicError, match="between 1 and 30"):
        check_logic("consistency", {}, [], timeout_seconds=31)
    too_deep: dict[str, object] = _var("P")
    for _ in range(18):
        too_deep = {"op": "not", "arg": too_deep}
    assert (
        logic_tool_result("consistency", {"P": "Bool"}, [too_deep])["error_kind"]
        == "parse_error"
    )
    assert (
        logic_tool_result("consistency", {}, [{}] * 21)["error_kind"] == "parse_error"
    )


def test_boolean_xor_and_numeric_arithmetic_are_supported() -> None:
    p = _var("P")
    parity = {"op": "xor", "args": [p, _const(False), _const(False)]}
    assert check_logic("entailment", {"P": "Bool"}, [p], parity)["status"] == "valid"

    x = _var("x")
    arithmetic = _binary(
        ">=",
        _binary("/", _binary("+", {"op": "neg", "arg": x}, _const(6)), _const(2)),
        _const(1.0),
    )
    assert (
        check_logic("consistency", {"x": "Int"}, [arithmetic])["status"] == "consistent"
    )
    assert (
        check_logic(
            "entailment",
            {"x": "Int"},
            [_binary("=", x, _const(2))],
            _binary("!=", x, _const(3)),
        )["status"]
        == "valid"
    )
    assert (
        check_logic(
            "entailment",
            {"x": "Int"},
            [_binary("=", x, _const(2))],
            _binary("<=", x, _const(2.0)),
        )["status"]
        == "valid"
    )


@pytest.mark.parametrize(
    ("formula", "kind"),
    [
        ({"const": "text"}, "parse_error"),
        ({"const": 10**13}, "parse_error"),
        ({"var": 123}, "parse_error"),
        ({"op": "and", "args": [_const(True)]}, "parse_error"),
        ({"op": "not", "arg": _const(1)}, "type_error"),
        ({"op": "neg", "arg": _const(True)}, "type_error"),
        (_binary("=", _const(True), _const(1)), "type_error"),
        (_binary("/", _const(1), _const(0)), "unsupported_theory"),
        ({"op": "not", "arg": _const(True), "extra": 1}, "parse_error"),
        ([], "parse_error"),
        ({"value": True}, "parse_error"),
    ],
)
def test_formula_language_reports_invalid_nodes(formula: object, kind: str) -> None:
    result = logic_tool_result("consistency", {}, [formula])  # type: ignore[list-item]
    assert result["error_kind"] == kind


@pytest.mark.parametrize(
    ("request_fields", "message"),
    [
        ({"task": "proof"}, "Unknown logic task"),
        ({"variables": {"__qs_reserved": "Bool"}}, "Declare up to 32"),
        ({"variables": {"bad name": "Bool"}}, "Declare up to 32"),
        ({"variables": {"x": "String"}}, "Declare up to 32"),
        ({"variables": {str(i): "Bool" for i in range(33)}}, "Declare up to 32"),
        ({"premises": {}}, "premises must be a list"),
        ({"task": "entailment"}, "conclusion is required"),
        ({"timeout_seconds": True}, "timeout_seconds"),
        ({"premises": [{"const": True, "pad": "x" * 12000}]}, "12000 character"),
        ({"premises": [object()]}, "JSON serializable"),
    ],
)
def test_logic_request_validation_rejects_bad_inputs(
    request_fields: dict[str, object], message: str
) -> None:
    kwargs = {
        "task": "consistency",
        "variables": {},
        "premises": [],
        "conclusion": None,
        "timeout_seconds": 5,
    } | request_fields
    with pytest.raises(FormalLogicError, match=message):
        check_logic(**kwargs)  # type: ignore[arg-type]


def test_solver_deadlines_and_unknown_results_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(formal_logic, "_check_with_deadline", lambda *args: None)
    timeout = check_logic("consistency", {}, [])
    assert timeout["status"] == "timeout"
    assert timeout["reason"] == "request deadline exceeded"

    import z3  # noqa: PLC0415

    monkeypatch.setattr(formal_logic, "_check_with_deadline", lambda *args: z3.unknown)
    assert check_logic("consistency", {}, [])["solver_result"] == "unknown"


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        ("timeout", "timeout"),
        ("memory cap", "resource_limit"),
        ("resource limit", "resource_limit"),
        ("incomplete theory", "unknown"),
    ],
)
def test_unknown_solver_reasons_are_classified(reason: str, status: str) -> None:
    result = formal_logic._unknown({}, SimpleNamespace(reason_unknown=lambda: reason))
    assert result["status"] == status


def test_logic_tool_errors_are_stable_without_dependency_or_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise ModuleNotFoundError("No module named 'z3'", name="z3")

    monkeypatch.setattr(formal_logic, "check_logic", missing)
    assert (
        logic_tool_result("consistency", {}, [])["error_kind"] == "missing_dependency"
    )

    def exhausted(*args: object, **kwargs: object) -> None:
        raise RuntimeError("memory exhausted")

    monkeypatch.setattr(formal_logic, "check_logic", exhausted)
    assert logic_tool_result("consistency", {}, [])["status"] == "resource_limit"

    def failed(*args: object, **kwargs: object) -> None:
        raise RuntimeError("solver failed")

    monkeypatch.setattr(formal_logic, "check_logic", failed)
    assert logic_tool_result("consistency", {}, [])["error_kind"] == "internal_error"


def test_entailment_reports_second_solver_deadline_and_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import z3  # noqa: PLC0415

    outcomes = iter([z3.sat, None])
    monkeypatch.setattr(
        formal_logic, "_check_with_deadline", lambda *args: next(outcomes)
    )
    assert check_logic("entailment", {}, [], _const(True))["status"] == "timeout"

    outcomes = iter([z3.sat, z3.unknown])
    monkeypatch.setattr(
        formal_logic, "_check_with_deadline", lambda *args: next(outcomes)
    )
    assert check_logic("entailment", {}, [], _const(True))["solver_result"] == "unknown"


def test_logic_rejects_numeric_premises_and_expired_solver_deadline() -> None:
    with pytest.raises(FormalLogicError, match="must be Boolean"):
        check_logic("consistency", {}, [_const(1)])
    assert formal_logic._check_with_deadline(object(), 0.0) is None


def test_formula_builder_fails_loudly_for_an_unhandled_internal_operator() -> None:
    import z3  # noqa: PLC0415

    builder = formal_logic._FormulaBuilder(z3, z3.Context(), {})
    with pytest.raises(AssertionError, match="Unhandled validated operator"):
        builder._build_binary(
            {"op": "unsupported", "left": _const(1), "right": _const(2)},
            op="unsupported",
            depth=0,
        )
