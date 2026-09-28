"""Find a formal model that refutes a proposed conclusion."""

from pathlib import Path
from typing import Any, Literal

from qs.config.reasoning import FORMAL_LOGIC
from qs.core.agents.tools.decorator import deep_tool
from qs.core.reasoning_tools.formal_logic import logic_tool_result


@deep_tool(
    name="logic-counterexample",
    description=(
        "Use Z3 to search for a model where every formal premise is true and "
        "the formal conclusion is false. Supply the JSON formula DSL and "
        "declared Bool, Int, or Real variables. A found model refutes the "
        "formal implication; not_found means no model was found in the supported theory."
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
        "counterexample", variables, premises, conclusion, timeout_seconds
    )
