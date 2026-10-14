"""Parse a small mathematical expression language without Python evaluation."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

import sympy as sp
from sympy.matrices.exceptions import NonInvertibleMatrixError

from exmekh_lib.config.reasoning import MATH
from exmekh_lib.config.reasoning_text import REASONING_TEXT
from exmekh_lib.utils.tool_errors import ToolInputError


class MathInputError(ToolInputError):
    pass


def calculate(
    operation: str,
    expression: str = "",
    variable: str = "x",
    lower_bound: str | None = None,
    upper_bound: str | None = None,
    matrix: list[list[float]] | None = None,
    precision: int = MATH.default_precision,
) -> dict[str, object]:
    if (
        type(precision) is not int
        or not MATH.min_precision <= precision <= MATH.max_precision
    ):
        raise MathInputError(
            REASONING_TEXT.precision_must_be_between_and.format(
                math_min_precision=MATH.min_precision,
                math_max_precision=MATH.max_precision,
            ),
            code="invalid_math",
        )
    if (
        not isinstance(variable, str)
        or not variable.isidentifier()
        or variable.startswith("_")
    ):
        raise MathInputError(
            REASONING_TEXT.variable_must_be_a_simple_identifier, code="invalid_math"
        )
    if operation in {"determinant", "inverse", "eigenvalues"}:
        result = _calculate_matrix(sp, operation, matrix)
        return {"operation": operation, "result": str(result)}

    if operation not in {
        "simplify",
        "evaluate",
        "solve",
        "differentiate",
        "integrate",
    }:
        raise MathInputError(
            REASONING_TEXT.unsupported_mathematical_operation,
            code="unsupported_operation",
        )
    if not isinstance(expression, str) or not expression.strip():
        raise MathInputError(
            REASONING_TEXT.expression_must_be_non_empty, code="invalid_math"
        )
    result = _calculate_expression(
        sp,
        operation=operation,
        expression=expression,
        variable=variable,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        precision=precision,
    )
    return {"operation": operation, "result": str(result)}


def _calculate_matrix(sp: Any, operation: str, matrix: list[list[float]] | None) -> Any:
    if (
        not isinstance(matrix, list)
        or not matrix
        or len(matrix) > MATH.max_matrix_dimension
        or any(not isinstance(row, list) or len(row) != len(matrix) for row in matrix)
    ):
        raise MathInputError(
            REASONING_TEXT.matrix_must_be_square_and_at_most_by.format(
                math_max_matrix_dimension=MATH.max_matrix_dimension,
                math_max_matrix_dimension_2=MATH.max_matrix_dimension,
            ),
            code="invalid_matrix",
        )
    if any(type(value) not in (int, float) for row in matrix for value in row):
        raise MathInputError(
            REASONING_TEXT.matrix_entries_must_be_numbers, code="invalid_matrix"
        )
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for row in matrix
        for value in row
    ):
        raise MathInputError(
            REASONING_TEXT.matrix_entries_must_be_finite, code="invalid_matrix"
        )
    value = sp.Matrix(matrix)
    if operation == "determinant":
        return value.det()
    if operation == "inverse":
        try:
            return value.inv()
        except NonInvertibleMatrixError as error:
            raise MathInputError(
                REASONING_TEXT.matrix_has_no_inverse, code="singular_matrix"
            ) from error
    return value.eigenvals()


def _calculate_expression(
    sp: Any,
    *,
    operation: str,
    expression: str,
    variable: str,
    lower_bound: str | None,
    upper_bound: str | None,
    precision: int,
) -> Any:
    symbol = sp.Symbol(variable)
    if operation == "solve" and "=" in expression:
        left, separator, right = expression.partition("=")
        if not separator or "=" in right:
            raise MathInputError(
                REASONING_TEXT.solve_accepts_at_most_one_equality_sign,
                code="invalid_math_syntax",
            )
        value = _parse(left.strip()) - _parse(right.strip())
    else:
        value = _parse(expression)
    if value.has(sp.zoo, sp.nan):
        raise MathInputError(
            REASONING_TEXT.undefined_math_result, code="undefined_math_result"
        )
    if operation == "simplify":
        result = sp.simplify(value)
    elif operation == "evaluate":
        if value.free_symbols:
            raise MathInputError(
                REASONING_TEXT.evaluate_requires_an_expression_without_variables,
                code="unbound_math_variables",
            )
        result = sp.N(value, precision)
        if result.is_finite is False:
            raise MathInputError(
                REASONING_TEXT.undefined_math_result, code="undefined_math_result"
            )
    elif operation == "solve":
        result = sp.solve(value, symbol)
    elif operation == "differentiate":
        result = sp.diff(value, symbol)
    else:
        if lower_bound is None and upper_bound is None:
            result = sp.integrate(value, symbol)
        elif lower_bound is None or upper_bound is None:
            raise MathInputError(
                REASONING_TEXT.integrate_requires_both_bounds_or_neither,
                code="incomplete_integral_bounds",
            )
        else:
            result = sp.integrate(
                value, (symbol, _parse(lower_bound), _parse(upper_bound))
            )
    return result


def _parse(expression: str) -> Any:
    import sympy as sp

    if len(expression) > MATH.max_expression_chars:
        raise MathInputError(
            REASONING_TEXT.expression_is_too_long, code="math_limit_exceeded"
        )
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise MathInputError(
            REASONING_TEXT.invalid_expression_syntax, code="invalid_math_syntax"
        ) from error
    if sum(1 for _ in ast.walk(tree)) > MATH.max_expression_nodes:
        raise MathInputError(
            REASONING_TEXT.expression_is_too_complex, code="math_limit_exceeded"
        )
    functions = {
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "log": sp.log,
        "sqrt": sp.sqrt,
        "Abs": sp.Abs,
    }
    binary: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def convert(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if len(str(node.value)) > MATH.max_number_chars:
                raise MathInputError(
                    REASONING_TEXT.number_is_too_large, code="math_limit_exceeded"
                )
            if isinstance(node.value, float) and not math.isfinite(node.value):
                raise MathInputError(
                    REASONING_TEXT.numbers_must_be_finite, code="undefined_math_result"
                )
            return sp.sympify(node.value)
        if isinstance(node, ast.Name) and node.id in {"pi", "E"}:
            return {"pi": sp.pi, "E": sp.E}[node.id]
        if (
            isinstance(node, ast.Name)
            and node.id.isidentifier()
            and not node.id.startswith("_")
        ):
            return sp.Symbol(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            operation = binary.get(type(node.op))
            if operation is None:
                raise MathInputError(
                    REASONING_TEXT.expression_contains_unsupported_syntax,
                    code="unsupported_math_syntax",
                )
            if isinstance(node.op, ast.Pow):
                if (
                    not isinstance(node.right, ast.Constant)
                    or not isinstance(node.right.value, (int, float))
                    or isinstance(node.right.value, bool)
                    or abs(node.right.value) > MATH.max_exponent
                ):
                    raise MathInputError(
                        REASONING_TEXT.exponents_must_be_numeric_and_at_most.format(
                            math_max_exponent=MATH.max_exponent
                        ),
                        code="math_limit_exceeded"
                        if isinstance(node.right, ast.Constant)
                        and type(node.right.value) in (int, float)
                        else "unsupported_math_syntax",
                    )
            return operation(convert(node.left), convert(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = functions.get(node.func.id)
            if function is not None and not node.keywords:
                accepted_counts = (1, 2) if node.func.id == "log" else (1,)
                if len(node.args) not in accepted_counts:
                    raise MathInputError(
                        REASONING_TEXT.invalid_function_arguments.format(
                            function=node.func.id,
                            count=len(node.args),
                            expected=accepted_counts,
                        ),
                        code="invalid_math_function_arguments",
                    )
                return function(*(convert(argument) for argument in node.args))
        raise MathInputError(
            REASONING_TEXT.expression_contains_unsupported_syntax,
            code="unsupported_math_syntax",
        )

    return convert(tree.body)
