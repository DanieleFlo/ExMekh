"""Bounded JSON formulas checked with Z3, without evaluating agent code."""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from exmekh_lib.config.reasoning import FORMAL_LOGIC
from exmekh_lib.config.reasoning_text import REASONING_TEXT
from exmekh_lib.config.tool_errors import ERROR_HINTS
from exmekh_lib.utils.tool_errors import ToolInputError, error_diagnostic

_LOGGER = logging.getLogger(__name__)


class FormalLogicError(ToolInputError):
    def __init__(
        self, message: str, *, kind: str = "parse_error", code: str | None = None
    ) -> None:
        super().__init__(message, code=code or kind)
        self.kind = kind


@dataclass(frozen=True)
class _Term:
    value: Any
    sort: str
    literal: bool = False


class _FormulaBuilder:
    def __init__(self, z3: Any, ctx: Any, variables: Mapping[str, str]) -> None:
        self._z3 = z3
        constructors = {
            "Bool": z3.Bool,
            "Int": z3.Int,
            "Real": z3.Real,
        }
        self._variables = {
            name: constructors[sort](name, ctx=ctx) for name, sort in variables.items()
        }
        self._sorts = variables
        self._context = ctx
        self._nodes = 0

    def build(self, node: object, *, depth: int = 0) -> _Term:
        self._nodes += 1
        if (
            self._nodes > FORMAL_LOGIC.max_formula_nodes
            or depth > FORMAL_LOGIC.max_formula_depth
        ):
            raise FormalLogicError(
                REASONING_TEXT.formula_exceeds_its_size_or_depth_limit
            )
        if not isinstance(node, dict):
            raise FormalLogicError(REASONING_TEXT.each_formula_node_must_be_an_object)
        if set(node) == {"var"}:
            name = node["var"]
            if not isinstance(name, str) or name not in self._variables:
                raise FormalLogicError(
                    REASONING_TEXT.undeclared_variable.format(name=name),
                    code="undeclared_logic_variable",
                )
            return _Term(self._variables[name], self._sorts[name])
        if set(node) == {"const"}:
            value = node["const"]
            if type(value) is bool:
                return _Term(self._z3.BoolVal(value, ctx=self._context), "Bool", True)
            if type(value) is int and abs(value) <= FORMAL_LOGIC.max_numeric_magnitude:
                return _Term(self._z3.IntVal(value, ctx=self._context), "Int", True)
            if (
                type(value) is float
                and math.isfinite(value)
                and abs(value) <= FORMAL_LOGIC.max_numeric_magnitude
            ):
                return _Term(
                    self._z3.RealVal(str(value), ctx=self._context), "Real", True
                )
            raise FormalLogicError(
                REASONING_TEXT.constants_must_be_bounded_booleans_or_numbers
            )

        op = node.get("op")
        if not isinstance(op, str):
            raise FormalLogicError(REASONING_TEXT.formula_node_needs_an_operator)
        if op not in {
            "not",
            "neg",
            "and",
            "or",
            "xor",
            "implies",
            "iff",
            "=",
            "!=",
            "<",
            "<=",
            ">",
            ">=",
            "+",
            "-",
            "*",
            "/",
        }:
            raise FormalLogicError(
                REASONING_TEXT.unsupported_operator.format(op=op),
                kind="unsupported_operation",
            )
        if op in {"not", "neg"}:
            return self._build_unary(node, op=op, depth=depth)
        if op in {"and", "or", "xor"}:
            return self._build_variadic(node, op=op, depth=depth)
        return self._build_binary(node, op=op, depth=depth)

    def model_values(self, model: Any) -> dict[str, bool | str]:
        """Return assignments using only declared variables and sorts."""
        return {
            name: self._z3.is_true(value) if self._sorts[name] == "Bool" else str(value)
            for name, variable in self._variables.items()
            if (value := model.eval(variable, model_completion=True)) is not None
        }

    def _build_unary(self, node: dict[str, Any], *, op: str, depth: int) -> _Term:
        self._keys(node, {"op", "arg"})
        arg = self.build(node["arg"], depth=depth + 1)
        if op == "not":
            self._require(arg, "Bool")
            return _Term(self._z3.Not(arg.value), "Bool")
        self._require_number(arg)
        return _Term(-arg.value, arg.sort, arg.literal)

    def _build_variadic(self, node: dict[str, Any], *, op: str, depth: int) -> _Term:
        self._keys(node, {"op", "args"})
        args = node["args"]
        if (
            not isinstance(args, list)
            or not 2 <= len(args) <= FORMAL_LOGIC.max_operands
        ):
            raise FormalLogicError(
                REASONING_TEXT.needs_operands.format(
                    op=op, formal_logic_max_operands=FORMAL_LOGIC.max_operands
                )
            )
        terms = [self.build(item, depth=depth + 1) for item in args]
        for term in terms:
            self._require(term, "Bool")
        values = [term.value for term in terms]
        if op == "and":
            return _Term(self._z3.And(*values), "Bool")
        if op == "or":
            return _Term(self._z3.Or(*values), "Bool")
        value = values[0]
        for other in values[1:]:
            value = self._z3.Xor(value, other)
        return _Term(value, "Bool")

    def _build_binary(self, node: dict[str, Any], *, op: str, depth: int) -> _Term:
        self._keys(node, {"op", "left", "right"})
        left = self.build(node["left"], depth=depth + 1)
        right = self.build(node["right"], depth=depth + 1)
        if op in {"implies", "iff"}:
            self._require(left, "Bool")
            self._require(right, "Bool")
            value = (
                self._z3.Implies(left.value, right.value)
                if op == "implies"
                else left.value == right.value
            )
            return _Term(value, "Bool")
        if op in {"=", "!="}:
            if left.sort == right.sort:
                first, second = left.value, right.value
            elif {left.sort, right.sort} <= {"Int", "Real"}:
                first, second, _ = self._numeric_pair(left, right)
            else:
                raise FormalLogicError(
                    REASONING_TEXT.equality_operands_have_incompatible_types,
                    kind="type_error",
                )
            return _Term(first == second if op == "=" else first != second, "Bool")
        if op in {"<", "<=", ">", ">="}:
            first, second, _ = self._numeric_pair(left, right)
            operations = {
                "<": first < second,
                "<=": first <= second,
                ">": first > second,
                ">=": first >= second,
            }
            return _Term(operations[op], "Bool")
        if op in {"+", "-", "*", "/"}:
            first, second, sort = self._numeric_pair(left, right)
            if op == "*" and not (left.literal or right.literal):
                raise FormalLogicError(
                    REASONING_TEXT.nonlinear_multiplication_is_unsupported,
                    kind="unsupported_theory",
                )
            if op == "/":
                if not right.literal or not self._z3.is_true(
                    self._z3.simplify(second != 0)
                ):
                    raise FormalLogicError(
                        REASONING_TEXT.division_requires_a_nonzero_literal_divisor,
                        kind="unsupported_theory",
                    )
                first = self._z3.ToReal(first) if sort == "Int" else first
                second = self._z3.ToReal(second) if sort == "Int" else second
                return _Term(first / second, "Real", left.literal and right.literal)
            value = {"+": first + second, "-": first - second, "*": first * second}[op]
            return _Term(value, sort, left.literal and right.literal)
        raise AssertionError(REASONING_TEXT.unhandled_validated_operator.format(op=op))

    @staticmethod
    def _keys(node: dict[str, object], expected: set[str]) -> None:
        if set(node) != expected:
            raise FormalLogicError(
                REASONING_TEXT.operator_node_must_have_keys.format(
                    expected=sorted(expected)
                )
            )

    @staticmethod
    def _require(term: _Term, sort: str) -> None:
        if term.sort != sort:
            raise FormalLogicError(
                REASONING_TEXT.expected_received.format(sort=sort, term_sort=term.sort),
                kind="type_error",
            )

    @staticmethod
    def _require_number(term: _Term) -> None:
        if term.sort not in {"Int", "Real"}:
            raise FormalLogicError(
                REASONING_TEXT.expected_a_numeric_expression, kind="type_error"
            )

    def _numeric_pair(self, left: _Term, right: _Term) -> tuple[Any, Any, str]:
        self._require_number(left)
        self._require_number(right)
        if left.sort == right.sort:
            return left.value, right.value, left.sort
        return (
            self._z3.ToReal(left.value) if left.sort == "Int" else left.value,
            self._z3.ToReal(right.value) if right.sort == "Int" else right.value,
            "Real",
        )


def check_logic(
    task: Literal["entailment", "consistency", "counterexample"],
    variables: dict[str, Literal["Bool", "Int", "Real"]],
    premises: list[dict[str, Any]],
    conclusion: dict[str, Any] | None = None,
    timeout_seconds: int = FORMAL_LOGIC.default_timeout_seconds,
) -> dict[str, object]:
    """Check a bounded formal formula; natural-language translation is external."""
    _validate_request(task, variables, premises, conclusion, timeout_seconds)
    import z3  # noqa: PLC0415

    deadline = time.monotonic() + timeout_seconds
    ctx = z3.Context()
    builder = _FormulaBuilder(z3, ctx, variables)
    formulas = [_boolean(builder.build(item)) for item in premises]
    goal = _boolean(builder.build(conclusion)) if conclusion is not None else None
    solver = z3.Solver(ctx=ctx)
    solver.set(
        unsat_core=True,
        max_memory=FORMAL_LOGIC.solver_max_memory_mb,
        rlimit=FORMAL_LOGIC.solver_resource_limit,
    )
    for index, formula in enumerate(formulas):
        solver.assert_and_track(
            formula, z3.Bool(f"__exmekh_lib_premise_{index}", ctx=ctx)
        )
    base: dict[str, object] = {"backend": "z3", "task": task}
    premise_result = _check_with_deadline(solver, deadline)
    if premise_result is None:
        return _deadline_exceeded(base)
    if premise_result == z3.unknown:
        return _unknown(base, solver)
    if premise_result == z3.unsat:
        core = {str(item) for item in solver.unsat_core()}
        return base | {
            "status": "inconsistent"
            if task == "consistency"
            else "inconsistent_premises",
            "solver_result": "unsat",
            "unsat_core": [
                premises[index]
                for index in range(len(premises))
                if f"__exmekh_lib_premise_{index}" in core
            ],
        }
    if task == "consistency":
        return base | {
            "status": "consistent",
            "solver_result": "sat",
            "model": builder.model_values(solver.model()),
        }
    assert goal is not None
    solver.push()
    solver.add(z3.Not(goal))
    result = _check_with_deadline(solver, deadline)
    if result is None:
        return _deadline_exceeded(base)
    if result == z3.unknown:
        return _unknown(base, solver)
    if result == z3.unsat:
        status = {"entailment": "valid", "counterexample": "not_found"}[task]
        return base | {
            "status": status,
            "solver_result": "unsat",
            "method": "refutation",
            "entailed": True,
        }
    status = {"entailment": "invalid", "counterexample": "found"}[task]
    return base | {
        "status": status,
        "solver_result": "sat",
        "method": "refutation",
        "entailed": False,
        "counterexample": builder.model_values(solver.model()),
    }


def _validate_request(
    task: object,
    variables: object,
    premises: object,
    conclusion: object,
    timeout_seconds: object,
) -> None:
    if not isinstance(task, str) or task not in {
        "entailment",
        "consistency",
        "counterexample",
    }:
        raise FormalLogicError(
            REASONING_TEXT.unknown_logic_task, kind="unsupported_operation"
        )
    if (
        not isinstance(variables, dict)
        or len(variables) > FORMAL_LOGIC.max_variables
        or any(
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None
            or name.startswith("__exmekh_lib_")
            or not isinstance(sort, str)
            or sort not in {"Bool", "Int", "Real"}
            for name, sort in variables.items()
        )
    ):
        raise FormalLogicError(
            REASONING_TEXT.declare_up_to_bool_int_or_real_variables_with.format(
                formal_logic_max_variables=FORMAL_LOGIC.max_variables
            )
        )
    if not isinstance(premises, list) or len(premises) > FORMAL_LOGIC.max_premises:
        raise FormalLogicError(
            REASONING_TEXT.premises_must_be_a_list_of_at_most_formulas.format(
                formal_logic_max_premises=FORMAL_LOGIC.max_premises
            )
        )
    if task != "consistency" and conclusion is None:
        raise FormalLogicError(REASONING_TEXT.conclusion_is_required_for_this_task)
    if (
        type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= FORMAL_LOGIC.max_timeout_seconds
    ):
        raise FormalLogicError(
            REASONING_TEXT.timeout_seconds_must_be_between_and.format(
                formal_logic_max_timeout_seconds=FORMAL_LOGIC.max_timeout_seconds
            )
        )
    try:
        size = len(json.dumps([variables, premises, conclusion]))
    except (TypeError, ValueError, RecursionError) as error:
        raise FormalLogicError(
            REASONING_TEXT.formulas_must_be_json_serializable
        ) from error
    if size > FORMAL_LOGIC.max_request_chars:
        raise FormalLogicError(
            REASONING_TEXT.request_exceeds_the_character_limit.format(
                formal_logic_max_request_chars=FORMAL_LOGIC.max_request_chars
            )
        )


def _boolean(term: _Term) -> Any:
    if term.sort != "Bool":
        raise FormalLogicError(
            REASONING_TEXT.premises_and_conclusions_must_be_boolean, kind="type_error"
        )
    return term.value


def _unknown(base: dict[str, object], solver: Any) -> dict[str, object]:
    reason = str(solver.reason_unknown())
    normalized = reason.lower()
    status = (
        "timeout"
        if "timeout" in normalized
        else "resource_limit"
        if "memory" in normalized or "resource" in normalized
        else "unknown"
    )
    return base | {
        "status": status,
        "solver_result": "unknown",
        "reason": reason,
        "hint": ERROR_HINTS[status],
    }


def _check_with_deadline(solver: Any, deadline: float) -> Any | None:
    remaining_ms = int((deadline - time.monotonic()) * 1000)
    if remaining_ms <= 0:
        return None
    solver.set(timeout=remaining_ms)
    return solver.check()


def _deadline_exceeded(base: dict[str, object]) -> dict[str, object]:
    return base | {
        "status": "timeout",
        "solver_result": "unknown",
        "reason": REASONING_TEXT.request_deadline_exceeded,
        "hint": ERROR_HINTS["timeout"],
    }


def logic_tool_result(
    task: Literal["entailment", "consistency", "counterexample"],
    variables: dict[str, Literal["Bool", "Int", "Real"]],
    premises: list[dict[str, Any]],
    conclusion: dict[str, Any] | None = None,
    timeout_seconds: int = FORMAL_LOGIC.default_timeout_seconds,
) -> dict[str, object]:
    """Convert validation and dependency failures to stable tool results."""
    try:
        return check_logic(task, variables, premises, conclusion, timeout_seconds)
    except ModuleNotFoundError as error:
        return {
            "status": "error",
            "backend": "z3",
            "error_kind": "missing_dependency",
            "error": REASONING_TEXT.missing_exmekh_lib_runtime_module.format(
                error_name=error.name
            ),
            "hint": REASONING_TEXT.install_the_exmekh_lib_wheel_with_its_declared_dependencies,
        }
    except FormalLogicError as error:
        return {
            "status": "error",
            "backend": "z3",
            "error_kind": error.kind,
            "error": str(error),
            **error_diagnostic(error),
        }
    except Exception as error:
        _LOGGER.exception(REASONING_TEXT.z3_logic_tool_failed)
        exhausted_memory = "memory" in str(error).lower()
        return {
            "status": "resource_limit" if exhausted_memory else "error",
            "backend": "z3",
            "error_kind": "resource_limit" if exhausted_memory else "internal_error",
            "error": REASONING_TEXT.the_z3_memory_limit_was_reached
            if exhausted_memory
            else REASONING_TEXT.the_z3_check_failed_unexpectedly,
            "hint": ERROR_HINTS[
                "resource_limit" if exhausted_memory else "internal_error"
            ],
        }
