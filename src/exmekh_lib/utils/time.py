"""Compact runtime timing metadata without changing stored measurements."""

from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from math import isfinite
from typing import Any

from exmekh_lib.config.runtime_text import RUNTIME_TEXT


def format_timestamp(value: datetime) -> str:
    """Render an aware timestamp in UTC, without fractional seconds."""
    if value.utcoffset() is None:
        raise ValueError(RUNTIME_TEXT.timestamp_timezone_required)
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def format_duration(seconds: float) -> str:
    """Round before splitting so tenths carry into minutes and hours."""
    try:
        valid = type(seconds) in (int, float) and isfinite(seconds) and seconds >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(RUNTIME_TEXT.invalid_duration)
    numerator, denominator = Decimal(str(seconds)).as_integer_ratio()
    tenths, remainder = divmod(numerator * 10, denominator)
    tenths += int(remainder * 2 >= denominator)
    hours, remaining = divmod(tenths, 36_000)
    minutes, remaining = divmod(remaining, 600)
    whole_seconds, fraction = divmod(remaining, 10)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if remaining or not parts:
        value = f"{whole_seconds}.{fraction}" if fraction else str(whole_seconds)
        parts.append(f"{value}s")
    return " ".join(parts)


def format_timing_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Format known XML timing fields, preserving unknown legacy metadata."""
    formatted = dict(fields)
    for name in ("started-at", "finished-at"):
        value = fields.get(name)
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value)
            except ValueError:
                continue
        if isinstance(value, datetime) and value.utcoffset() is not None:
            try:
                formatted[name] = format_timestamp(value)
            except (ValueError, OverflowError):
                continue
    elapsed = fields.get("elapsed-seconds")
    if isinstance(elapsed, (int, float, str)) and not isinstance(elapsed, bool):
        try:
            seconds = float(elapsed)
        except (ValueError, OverflowError):
            return formatted
        if isfinite(seconds) and seconds >= 0:
            del formatted["elapsed-seconds"]
            formatted["elapsed"] = format_duration(seconds)
    return formatted
