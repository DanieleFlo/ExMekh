"""Check whether formal constraints have a model."""

from pathlib import Path
from typing import Any, Literal

from exmekh_lib.config.reasoning import FORMAL_LOGIC
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.reasoning_tools.formal_logic import logic_tool_result


@deep_tool(
    name="logic-consistency",
    description=(
        "Use Z3 to determine whether formal premises can all be true together. "
        "Supply the JSON formula DSL and declared Bool, Int, or Real variables. "
        "Returns a model when consistent or an unsat core when inconsistent; "
        "unknown and timeout remain distinct. This checks the formulas, not real-world facts."
    ),
)
def run(
    work_root: Path,
    variables: dict[str, Literal["Bool", "Int", "Real"]],
    premises: list[dict[str, Any]],
    timeout_seconds: int = FORMAL_LOGIC.default_timeout_seconds,
) -> dict[str, object]:
    del work_root
    return logic_tool_result("consistency", variables, premises, None, timeout_seconds)
