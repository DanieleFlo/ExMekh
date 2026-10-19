"""Read the clock and perform timezone-aware date/time calculations."""

from pathlib import Path
from typing import Literal

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.config.reasoning import DATE_TIME
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.reasoning_tools.date_time import (
    DateTimeDependencyError,
    DateTimeInputError,
    calculate_date_time,
)
from exmekh_lib.utils.tool_errors import tool_error_result


@deep_tool(
    name=DEFAULT_CAPABILITIES.date_time,
    description=(DATE_TIME.description),
)
def run(
    work_root: Path,
    operation: Literal["now", "convert", "add", "difference"],
    value: str = "",
    other: str = "",
    timezone: str = DATE_TIME.default_timezone,
    amount: float = 0,
    unit: Literal["seconds", "minutes", "hours", "days"] = DATE_TIME.default_unit,
) -> dict[str, object]:
    del work_root
    try:
        return calculate_date_time(
            operation, value, other, timezone=timezone, amount=amount, unit=unit
        )
    except DateTimeDependencyError as error:
        return {
            "ok": False,
            "error_kind": "missing_dependency",
            "error": str(error),
            "hint": DATE_TIME.dependency_hint,
        }
    except DateTimeInputError as error:
        return tool_error_result(error, "invalid_date_time")
