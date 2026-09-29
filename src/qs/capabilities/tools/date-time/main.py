"""Read the clock and perform timezone-aware date/time calculations."""

from pathlib import Path
from typing import Literal

from qs.config.reasoning import DATE_TIME
from qs.core.agents.tools.decorator import deep_tool
from qs.core.reasoning_tools.date_time import (
    DateTimeDependencyError,
    DateTimeInputError,
    calculate_date_time,
)


@deep_tool(
    name="date-time",
    description=(
        "Read the current time (now), convert ISO 8601 dates/datetimes to an IANA "
        "timezone (convert), add a signed elapsed duration (add), or calculate "
        "other minus value (difference). timezone defaults to UTC and also "
        "interprets inputs without an offset; date-only inputs mean midnight. "
        "Use explicit offsets for ambiguous local times. amount and unit control "
        "addition; unit controls the difference result. A day is 24 elapsed hours."
    ),
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
            "hint": "Install the qs wheel with its declared tzdata dependency in this environment.",
        }
    except DateTimeInputError as error:
        return {"ok": False, "error_kind": "invalid_date_time", "error": str(error)}
