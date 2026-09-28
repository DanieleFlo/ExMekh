from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MathSettings:
    """Precision defaults and input budgets for symbolic calculations."""

    default_precision: int = 15
    min_precision: int = 2
    max_precision: int = 50
    max_matrix_dimension: int = 20
    max_expression_chars: int = 2_000
    max_expression_nodes: int = 250
    max_number_chars: int = 100
    max_exponent: int = 100


@dataclass(frozen=True, slots=True)
class FormalLogicSettings:
    """Input and solver resource budgets for formal logic tools."""

    default_timeout_seconds: int = 5
    max_timeout_seconds: int = 30
    max_formula_nodes: int = 200
    max_formula_depth: int = 16
    max_numeric_magnitude: int = 10**12
    max_operands: int = 10
    max_variables: int = 32
    max_premises: int = 20
    max_request_chars: int = 12_000
    solver_max_memory_mb: int = 128
    solver_resource_limit: int = 1_000_000


MATH = MathSettings()
FORMAL_LOGIC = FormalLogicSettings()
