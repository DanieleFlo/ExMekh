"""Expose bounded symbolic and numerical operations."""

from pathlib import Path
from typing import Literal

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.config.reasoning import MATH
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.reasoning_tools.math import MathInputError, calculate
from exmekh_lib.utils.tool_errors import missing_dependency, tool_error_result


@deep_tool(
    name=DEFAULT_CAPABILITIES.calculate,
    description=(MATH.description),
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
    precision: int = MATH.default_precision,
) -> dict[str, object]:
    del work_root
    try:
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
        return missing_dependency(error)
    except (MathInputError, ValueError, ZeroDivisionError) as error:
        return tool_error_result(error, "invalid_math")
