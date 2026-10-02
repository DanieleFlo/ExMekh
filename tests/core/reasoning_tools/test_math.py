import pytest

from exmekh_lib.core.reasoning_tools.math import MathInputError, calculate


def test_symbolic_and_numerical_operations() -> None:
    pytest.importorskip("sympy")
    assert calculate("differentiate", "x**2 + 3*x")["result"] == "2*x + 3"
    assert calculate("solve", "x**2=4")["result"] == "[-2, 2]"
    assert calculate("solve", "x**2 - 5*x + 6 = 0")["result"] == "[2, 3]"
    assert calculate("evaluate", "sqrt(2)")["result"].startswith("1.4142")
    assert calculate("determinant", matrix=[[1, 2], [3, 4]])["result"] == "-2"


def test_math_parser_rejects_python_execution() -> None:
    pytest.importorskip("sympy")
    with pytest.raises(MathInputError, match="unsupported syntax"):
        calculate("evaluate", "__import__('os').system('echo unsafe')")
    with pytest.raises(MathInputError, match="exponents"):
        calculate("evaluate", "2**1000")


def test_invalid_matrix_and_variable_are_reported_as_input_errors() -> None:
    with pytest.raises(MathInputError, match="variable"):
        calculate("evaluate", "1 + 1", variable=42)  # type: ignore[arg-type]
    with pytest.raises(MathInputError, match="matrix must be square"):
        calculate("determinant", matrix=[1, 2])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("operation", "expression", "expected"),
    [
        ("simplify", "x + x", "2*x"),
        ("integrate", "2*x", "x**2"),
        ("evaluate", "pi + E", "5.85987448204884"),
        ("evaluate", "+2 - 1", "1.00000000000000"),
        ("evaluate", "sin(0) + cos(0) + tan(0)", "1.00000000000000"),
    ],
)
def test_expression_language_supported_forms(
    operation: str, expression: str, expected: str
) -> None:
    assert calculate(operation, expression)["result"] == expected


def test_bounded_integral_and_matrix_operations() -> None:
    assert (
        calculate("integrate", "2*x", lower_bound="0", upper_bound="2")["result"] == "4"
    )
    assert calculate("inverse", matrix=[[2, 0], [0, 4]])["result"] == (
        "Matrix([[1/2, 0], [0, 1/4]])"
    )
    assert calculate("eigenvalues", matrix=[[2, 0], [0, 4]])["result"] == (
        "{2: 1, 4: 1}"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"precision": True}, "precision"),
        ({"precision": 51}, "precision"),
        ({"operation": "unknown"}, "unsupported mathematical operation"),
        ({"expression": ""}, "expression must be non-empty"),
        ({"operation": "evaluate", "expression": "x + 1"}, "without variables"),
        ({"operation": "integrate", "lower_bound": "0"}, "both bounds"),
        ({"operation": "solve", "expression": "x=1=2"}, "at most one equality"),
        (
            {"operation": "determinant", "matrix": [[True]]},
            "matrix entries must be numbers",
        ),
        (
            {"operation": "determinant", "matrix": [[float("nan")]]},
            "matrix entries must be finite",
        ),
    ],
)
def test_math_rejects_ambiguous_or_unbounded_requests(
    kwargs: dict[str, object], message: str
) -> None:
    request = {"operation": "evaluate", "expression": "1+1"} | kwargs
    with pytest.raises(MathInputError, match=message):
        calculate(**request)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("1+", "invalid expression syntax"),
        ("1" * 2001, "expression is too long"),
        ("+".join(["1"] * 150), "expression is too complex"),
        ("9" * 101, "number is too large"),
        ("1e309", "numbers must be finite"),
        ("x**x", "exponents must be numeric"),
        ("_private", "unsupported syntax"),
        ("sqrt(x, scale=1)", "unsupported syntax"),
        ("1//2", "unsupported syntax"),
        ("[1, 2]", "unsupported syntax"),
    ],
)
def test_math_parser_rejects_invalid_constructs(expression: str, message: str) -> None:
    with pytest.raises(MathInputError, match=message):
        calculate("simplify", expression)
