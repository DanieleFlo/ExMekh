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
        "Compute mathematical operations: simplify, evaluate, solve equations, differentiate, "
        "integrate (with optional bounds), and calculate matrix determinants, inverses, "
        "or eigenvalues. Use Python-style syntax: +, -, *, /, ** for powers, and parentheses. "
        "Supported functions: sin, cos, tan, exp, log, sqrt, Abs; constants: pi, E. "
        "Functions take one argument; log also accepts log(value, base). "
        "Use explicit multiplication (2*x), numeric literal exponents, and = for equations. "
        "Specify the variable when needed, both bounds for definite integrals, and square numeric "
        "matrices for matrix operations."
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
        "Use Z3 to check whether a formal Boolean or linear-arithmetic conclusion "
        "follows from formal premises. Supply the JSON formula DSL and declared Bool, "
        "Int, or Real variables. Returns valid, invalid with a model, "
        "inconsistent_premises, unknown, or timeout. This does not establish that "
        "premises are factually true or that a natural-language translation is correct."
    )

    consistency_description: str = (
        "Use Z3 to determine whether formal premises can all be true together. Supply "
        "the JSON formula DSL and declared Bool, Int, or Real variables. Returns a "
        "model when consistent or an unsat core when inconsistent; unknown and timeout "
        "remain distinct. This checks the formulas, not real-world facts."
    )

    counterexample_description: str = (
        "Use Z3 to search for a model where every formal premise is true and the "
        "formal conclusion is false. Supply the JSON formula DSL and declared Bool, "
        "Int, or Real variables. A found model refutes the formal implication; "
        "not_found means no model was found in the supported theory."
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
            "Read the current time (now), convert ISO 8601 dates/datetimes to an IANA "
            "timezone (convert), add a signed elapsed duration (add), or calculate "
            f"other minus value (difference). timezone defaults to {self.default_timezone} and also "
            "interprets inputs without an offset; date-only inputs mean midnight. "
            "Use explicit offsets for ambiguous local times. amount and unit control "
            "addition; unit controls the difference result. A day is 24 elapsed hours."
        )


MATH = MathSettings()
FORMAL_LOGIC = FormalLogicSettings()
DATE_TIME = DateTimeSettings()
