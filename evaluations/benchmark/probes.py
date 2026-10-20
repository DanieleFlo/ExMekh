"""Execute independent API probes and preserve each outcome in its input position."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluations.benchmark.sandbox import Execution, execute
from evaluations.tool_argument_grading import parse_json_value


@dataclass(frozen=True)
class ProgramProbe:
    arguments: list[Any]
    execution: Execution
    value: Any = None
    parse_error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.execution.succeeded and self.parse_error is None

    def evidence(self) -> dict[str, Any]:
        return {
            "input": self.arguments,
            **asdict(self.execution),
            "parse_error": self.parse_error,
        }


@dataclass(frozen=True)
class ProbeBatch:
    results: tuple[ProgramProbe, ...]

    @property
    def infrastructure_error(self) -> str | None:
        return next(
            (
                item.execution.infrastructure_error
                for item in self.results
                if item.execution.infrastructure_error is not None
            ),
            None,
        )

    def evidence(self) -> dict[str, Any]:
        return {
            "executions": [item.evidence() for item in self.results],
            "values": [item.value if item.succeeded else None for item in self.results],
        }


def run_probes(
    workspace: Path, parameters: dict[str, Any], *, mode: str = "call"
) -> ProbeBatch:
    if not parameters["inputs"]:
        raise ValueError("Program checks require at least one input")
    results = []
    for arguments in parameters["inputs"]:
        execution = execute(
            workspace,
            {
                "mode": mode,
                "path": parameters["module"],
                "function": parameters["function"],
                "arguments": arguments,
            },
        )
        value, error = None, None
        if execution.succeeded:
            try:
                value = parse_json_value(execution.stdout)
            except (ValueError, TypeError) as failure:
                error = str(failure)
        results.append(ProgramProbe(arguments, execution, value, error))
        if execution.infrastructure_error is not None:
            break
    return ProbeBatch(tuple(results))
