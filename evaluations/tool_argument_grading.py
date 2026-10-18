"""Deterministic grading of raw model arguments, without provider coercion."""

import inspect
import json
import math
from dataclasses import dataclass

from evaluations.tool_argument_cases import ArgumentCase
from exmekh_lib.utils.json import JsonValue, to_json_value


@dataclass(frozen=True)
class RawToolCall:
    name: str
    arguments: str


@dataclass(frozen=True)
class ArgumentCheck:
    path: str
    passed: bool
    expected: JsonValue
    actual: JsonValue
    present: bool = True


@dataclass(frozen=True)
class ArgumentGrade:
    score: float
    exact: bool
    checks: tuple[ArgumentCheck, ...]
    issue: str | None = None


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate argument key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite JSON number: {value}")
    return number


def _pointer_part(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _compare(
    expected: JsonValue, actual: JsonValue, path: str = "", *, present: bool = True
) -> list[ArgumentCheck]:
    if isinstance(expected, dict):
        members = actual if isinstance(actual, dict) else {}
        checks = [
            ArgumentCheck(
                f"{path}/@keys",
                present
                and isinstance(actual, dict)
                and members.keys() == expected.keys(),
                to_json_value(sorted(expected)),
                to_json_value(sorted(members)),
                present,
            )
        ]
        for key, value in expected.items():
            checks.extend(
                _compare(
                    value,
                    members.get(key),
                    f"{path}/{_pointer_part(key)}",
                    present=key in members,
                )
            )
        return checks
    if isinstance(expected, list):
        members = actual if isinstance(actual, list) else []
        checks = [
            ArgumentCheck(
                f"{path}/@length",
                present and isinstance(actual, list) and len(members) == len(expected),
                len(expected),
                len(members),
                present,
            )
        ]
        for index, value in enumerate(expected):
            checks.extend(
                _compare(
                    value,
                    members[index] if index < len(members) else None,
                    f"{path}/{index}",
                    present=index < len(members),
                )
            )
        return checks
    matches = type(actual) is type(expected) and actual == expected
    if type(expected) is int:
        matches = (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and actual == expected
        )
    if isinstance(expected, float):
        try:
            matches = (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9)
            )
        except OverflowError:
            matches = False
    return [ArgumentCheck(path, present and matches, expected, actual, present)]


def grade_arguments(
    case: ArgumentCase, calls: tuple[RawToolCall, ...]
) -> ArgumentGrade:
    if len(calls) != 1 or calls[0].name != case.tool.get_name():
        return ArgumentGrade(
            0.0, False, (), "Expected exactly one call to the designated tool."
        )
    try:
        arguments = json.loads(
            calls[0].arguments,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
            parse_float=_finite_float,
        )
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be a JSON object.")
    except (ValueError, TypeError) as error:
        return ArgumentGrade(0.0, False, (), f"Invalid argument JSON: {error}")
    # Only runtime-documented null/default equivalence is normalized. Missing
    # arguments remain missing, and business values never undergo coercion.
    for name, parameter in case.tool.get_signature().parameters.items():
        if (
            case.tool.is_execution_argument(name)
            and name in arguments
            and arguments[name] is None
        ):
            if parameter.default is not inspect.Parameter.empty:
                arguments[name] = parameter.default
    checks = tuple(_compare(case.expected_arguments(), arguments))
    passed = sum(check.passed for check in checks)
    return ArgumentGrade(100.0 * passed / len(checks), passed == len(checks), checks)
