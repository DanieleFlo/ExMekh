"""ISO date/time operations with explicit zones and elapsed-time arithmetic."""

import math
from datetime import UTC, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from exmekh_lib.config.reasoning import DATE_TIME
from exmekh_lib.config.reasoning_text import REASONING_TEXT


class DateTimeInputError(ValueError):
    pass


class DateTimeDependencyError(RuntimeError):
    """No IANA timezone database is available in this environment."""


def calculate_date_time(
    operation: str,
    value: str = "",
    other: str = "",
    *,
    timezone: str = DATE_TIME.default_timezone,
    amount: float = 0,
    unit: str = DATE_TIME.default_unit,
) -> dict[str, object]:
    if not isinstance(operation, str) or operation not in {
        "now",
        "convert",
        "add",
        "difference",
    }:
        raise DateTimeInputError(REASONING_TEXT.unsupported_date_time_operation)
    unit_duration = _parse_unit_duration(unit)
    zone = _resolve_timezone(timezone)
    metadata: dict[str, object] = {"operation": operation, "timezone": timezone}
    try:
        if operation == "now":
            return metadata | _datetime_result(datetime.now(zone))
        start = _parse_datetime(value, zone, "value")
        if operation == "difference":
            end = _parse_datetime(other, zone, "other")
            return (
                metadata
                | {"unit": unit}
                | _difference_result(start, end, unit_duration)
            )
        if operation == "add":
            start = _add_elapsed_duration(start, amount, unit_duration)
        return metadata | _datetime_result(start)
    except OverflowError as error:
        raise DateTimeInputError(
            REASONING_TEXT.date_time_or_duration_is_outside_the_supported_range
        ) from error


def _parse_unit_duration(unit: str) -> timedelta:
    if not isinstance(unit, str) or unit not in {"seconds", "minutes", "hours", "days"}:
        raise DateTimeInputError(
            REASONING_TEXT.unit_must_be_seconds_minutes_hours_or_days
        )
    return timedelta(**{unit: 1})


def _add_elapsed_duration(value: datetime, amount: float, unit: timedelta) -> datetime:
    if type(amount) not in (int, float) or not math.isfinite(amount):
        raise DateTimeInputError(REASONING_TEXT.amount_must_be_a_finite_number)
    return (value.astimezone(UTC) + amount * unit).astimezone(value.tzinfo)


def _difference_result(
    start: datetime, end: datetime, unit: timedelta
) -> dict[str, object]:
    elapsed = end.astimezone(UTC) - start.astimezone(UTC)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "total_seconds": elapsed.total_seconds(),
        "result": elapsed / unit,
    }


def _resolve_timezone(name: str) -> tzinfo:
    if not isinstance(name, str) or not name:
        raise DateTimeInputError(
            REASONING_TEXT.timezone_must_be_a_non_empty_iana_name_such
        )
    if name == "UTC":
        return UTC
    try:
        return ZoneInfo(name)
    except ValueError as error:
        raise DateTimeInputError(
            REASONING_TEXT.timezone_must_be_a_valid_iana_name_such_as
        ) from error
    except ZoneInfoNotFoundError as error:
        if not available_timezones():
            raise DateTimeDependencyError(
                REASONING_TEXT.the_iana_timezone_database_is_unavailable_install_the_declared
            ) from error
        raise DateTimeInputError(
            REASONING_TEXT.unknown_timezone_use_an_iana_name_such_as_europe.format(
                name=name
            )
        ) from error


def _parse_datetime(text: str, zone: tzinfo, parameter: str) -> datetime:
    if not isinstance(text, str) or not text.strip():
        raise DateTimeInputError(
            REASONING_TEXT.must_be_a_non_empty_iso_date_or_datetime.format(
                parameter=parameter
            )
        )
    try:
        value = datetime.fromisoformat(text.strip())
    except ValueError as error:
        raise DateTimeInputError(
            REASONING_TEXT.must_be_an_iso_date_or_datetime.format(parameter=parameter)
        ) from error
    if value.tzinfo is not None:
        return value.astimezone(zone)
    return _localize_datetime(value, zone, parameter)


def _localize_datetime(value: datetime, zone: tzinfo, parameter: str) -> datetime:
    local = value.replace(tzinfo=zone)
    # A UTC round trip detects local times skipped by a daylight-saving transition.
    if local.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != value:
        raise DateTimeInputError(
            REASONING_TEXT.local_time_does_not_exist_in_this_timezone_supply.format(
                parameter=parameter
            )
        )
    if local.utcoffset() != value.replace(tzinfo=zone, fold=1).utcoffset():
        raise DateTimeInputError(
            REASONING_TEXT.local_time_is_ambiguous_in_this_timezone_supply_an.format(
                parameter=parameter
            )
        )
    return local


def _datetime_result(value: datetime) -> dict[str, object]:
    return {
        "datetime": value.isoformat(),
        "utc": value.astimezone(UTC).isoformat(),
        "unix_timestamp": value.timestamp(),
    }
