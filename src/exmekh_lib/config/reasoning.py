from dataclasses import dataclass
from typing import Literal


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

    description: str = (
        "Simplify/evaluate, solve, differentiate/integrate or compute square numeric matrix "
        "determinants/inverses/eigenvalues. Syntax: explicit 2*x, numeric powers x**2, = for equations; "
        "pi, E and unary sin/cos/tan/exp/log/sqrt/Abs (log also accepts a base). "
        "Supply the variable when needed and both bounds for definite integrals. No Python execution."
    )


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

    check_description: str = (
        "Check formal entailment with Z3 using the JSON formula DSL and declared Bool/Int/Real "
        "variables. Returns valid, invalid/countermodel, inconsistent_premises, unknown or timeout. "
        "Checks formulas, not factual premises or natural-language translation."
    )

    consistency_description: str = (
        "Check premise consistency with Z3: JSON formula DSL and declared Bool/Int/Real variables. "
        "Returns a model or unsat core; unknown/timeout are inconclusive. Checks formulas, not facts."
    )

    counterexample_description: str = (
        "Find a Z3 model satisfying premises and negating the conclusion: JSON formula DSL "
        "with declared Bool/Int/Real variables. found refutes the implication; not_found "
        "applies to the supported theory; unknown/timeout are inconclusive."
    )


@dataclass(frozen=True, slots=True)
class DateTimeSettings:
    """Explicit timezone and unit defaults for date/time operations."""

    default_timezone: str = "UTC"
    default_unit: Literal["seconds", "minutes", "hours", "days"] = "seconds"
    dependency_hint: str = "Install the ExMekh-Lib wheel with its declared tzdata dependency in this environment."

    @property
    def description(self) -> str:
        return (
            "now: current time; convert: ISO date/datetime to IANA timezone; add: signed "
            "amount/unit; difference: other-value in unit. "
            f"timezone defaults to {self.default_timezone}, interpreting offset-free inputs; dates mean midnight. "
            "Resolve DST ambiguity with explicit offsets. A day is 24 elapsed hours."
        )


MATH = MathSettings()
FORMAL_LOGIC = FormalLogicSettings()
DATE_TIME = DateTimeSettings()
