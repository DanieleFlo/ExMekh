"""Parse a small mathematical expression language without Python evaluation."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from qs.config.reasoning import MATH


class MathInputError(ValueError):
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
    import sympy as sp

    if (
        type(precision) is not int
        or not MATH.min_precision <= precision <= MATH.max_precision
    ):
        raise MathInputError(
            f"precision must be between {MATH.min_precision} and {MATH.max_precision}"
        )
    if (
        not isinstance(variable, str)
        or not variable.isidentifier()
        or variable.startswith("_")
    ):
        raise MathInputError("variable must be a simple identifier")
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
        raise MathInputError("unsupported mathematical operation")
    if not isinstance(expression, str) or not expression.strip():
        raise MathInputError("expression must be non-empty")
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
            f"matrix must be square and at most {MATH.max_matrix_dimension} by {MATH.max_matrix_dimension}"
        )
    if any(type(value) not in (int, float) for row in matrix for value in row):
        raise MathInputError("matrix entries must be numbers")
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for row in matrix
        for value in row
    ):
        raise MathInputError("matrix entries must be finite")
    value = sp.Matrix(matrix)
    if operation == "determinant":
        return value.det()
    if operation == "inverse":
        return value.inv()
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
            raise MathInputError("solve accepts at most one equality sign")
        value = _parse(left.strip()) - _parse(right.strip())
    else:
        value = _parse(expression)
    if operation == "simplify":
        result = sp.simplify(value)
    elif operation == "evaluate":
        if value.free_symbols:
            raise MathInputError("evaluate requires an expression without variables")
        result = sp.N(value, precision)
    elif operation == "solve":
        result = sp.solve(value, symbol)
    elif operation == "differentiate":
        result = sp.diff(value, symbol)
    else:
        if lower_bound is None and upper_bound is None:
            result = sp.integrate(value, symbol)
        elif lower_bound is None or upper_bound is None:
            raise MathInputError("integrate requires both bounds or neither")
        else:
            result = sp.integrate(
                value, (symbol, _parse(lower_bound), _parse(upper_bound))
            )
    return result


def _parse(expression: str) -> Any:
    import sympy as sp

    if len(expression) > MATH.max_expression_chars:
        raise MathInputError("expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise MathInputError("invalid expression syntax") from error
    if sum(1 for _ in ast.walk(tree)) > MATH.max_expression_nodes:
        raise MathInputError("expression is too complex")
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
                raise MathInputError("number is too large")
            if isinstance(node.value, float) and not math.isfinite(node.value):
                raise MathInputError("numbers must be finite")
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
                raise MathInputError("expression contains unsupported syntax")
            if isinstance(node.op, ast.Pow):
                if (
                    not isinstance(node.right, ast.Constant)
                    or not isinstance(node.right.value, (int, float))
                    or isinstance(node.right.value, bool)
                    or abs(node.right.value) > MATH.max_exponent
                ):
                    raise MathInputError(
                        f"exponents must be numeric and at most {MATH.max_exponent}"
                    )
            return operation(convert(node.left), convert(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = functions.get(node.func.id)
            if function is not None and not node.keywords and 1 <= len(node.args) <= 2:
                return function(*(convert(argument) for argument in node.args))
        raise MathInputError("expression contains unsupported syntax")

    return convert(tree.body)
