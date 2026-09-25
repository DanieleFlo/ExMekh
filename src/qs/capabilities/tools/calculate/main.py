"""Expose bounded symbolic and numerical operations."""

from pathlib import Path
from typing import Literal

from qs.core.agents.tools.decorator import deep_tool
from qs.core.reasoning_tools.math import MathInputError


@deep_tool(
    name="calculate",
    description=(
        "Compute bounded symbolic or numerical expressions: simplify, evaluate, "
        "solve, differentiate, integrate, determinant, inverse, or eigenvalues. "
        "Expressions use arithmetic and common named functions only."
    ),
)
def run(
    work_root: Path,
    operation: Literal[
        "simplify",
        "evaluate",
        "solve",
        "differentiate",
        "integrate",
        "determinant",
        "inverse",
        "eigenvalues",
    ],
    expression: str = "",
    variable: str = "x",
    lower_bound: str | None = None,
    upper_bound: str | None = None,
    matrix: list[list[float]] | None = None,
    precision: int = 15,
) -> dict[str, object]:
    del work_root
    try:
        from qs.core.reasoning_tools.math import calculate

        return calculate(
            operation,
            expression,
            variable,
            lower_bound,
            upper_bound,
            matrix,
            precision,
        )
    except ModuleNotFoundError as error:
        return {
            "ok": False,
            "error_kind": "missing_dependency",
            "error": f"Missing qs runtime module: {error.name}.",
            "hint": "Install the qs wheel with its declared dependencies in this environment.",
        }
    except (MathInputError, ValueError, ZeroDivisionError) as error:
        return {"ok": False, "error_kind": "invalid_math", "error": str(error)}
