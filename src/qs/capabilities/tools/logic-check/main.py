"""Check formal entailment with Z3."""

from pathlib import Path
from typing import Any, Literal

from qs.config.reasoning import FORMAL_LOGIC
from qs.core.agents.tools.decorator import deep_tool
from qs.core.reasoning_tools.formal_logic import logic_tool_result


@deep_tool(
    name="logic-check",
    description=(
        "Use Z3 to check whether a formal Boolean or linear-arithmetic conclusion "
        "follows from formal premises. Supply the JSON formula DSL and declared "
        "Bool, Int, or Real variables. Returns valid, invalid with a model, "
        "inconsistent_premises, unknown, or timeout. This does not establish "
        "that premises are factually true or that a natural-language translation is correct."
    ),
)
def run(
    work_root: Path,
    variables: dict[str, Literal["Bool", "Int", "Real"]],
    premises: list[dict[str, Any]],
    conclusion: dict[str, Any],
    timeout_seconds: int = FORMAL_LOGIC.default_timeout_seconds,
) -> dict[str, object]:
    del work_root
    return logic_tool_result(
        "entailment", variables, premises, conclusion, timeout_seconds
    )
