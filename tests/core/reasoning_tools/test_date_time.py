from datetime import UTC, datetime
from pathlib import Path

import pytest

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.capabilities.defaults import DefaultCapabilities
from qs.core.models.tool import Tool
from qs.core.providers.openai_chat_completion_annotation_schema_builder import (
    OpenAIChatCompletionAnnotationSchemaBuilder,
)
from qs.core.providers.openai_function_tool_definition_mapper import (
    OpenAIFunctionToolDefinitionMapper,
)
from qs.core.reasoning_tools import date_time
from qs.core.reasoning_tools.date_time import (
    DateTimeDependencyError,
    DateTimeInputError,
    calculate_date_time,
)


def test_now_uses_the_requested_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 1, 12, 30, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr(date_time, "datetime", FixedDateTime)

    result = calculate_date_time("now", timezone="Europe/Rome")

    assert result["datetime"] == "2026-07-01T14:30:00+02:00"
    assert result["utc"] == "2026-07-01T12:30:00+00:00"
    assert result["timezone"] == "Europe/Rome"
    assert (
        result["unix_timestamp"] == datetime(2026, 7, 1, 12, 30, tzinfo=UTC).timestamp()
    )


@pytest.mark.parametrize(
    ("value", "timezone", "expected"),
    [
        ("2026-01-01T10:15:00Z", "Europe/Rome", "2026-01-01T11:15:00+01:00"),
        ("2026-07-01T10:15:00+02:00", "UTC", "2026-07-01T08:15:00+00:00"),
        ("2024-02-29", "UTC", "2024-02-29T00:00:00+00:00"),
        ("2026-01-01T10:15:00", "Asia/Kolkata", "2026-01-01T10:15:00+05:30"),
    ],
)
def test_convert_accepts_iso_dates_offsets_and_local_times(
    value: str, timezone: str, expected: str
) -> None:
    assert (
        calculate_date_time("convert", value, timezone=timezone)["datetime"] == expected
    )


@pytest.mark.parametrize(
    ("value", "amount", "unit", "timezone", "expected"),
    [
        ("2024-02-28", 1, "days", "UTC", "2024-02-29T00:00:00+00:00"),
        ("2026-01-01", -1, "seconds", "UTC", "2025-12-31T23:59:59+00:00"),
        ("2026-01-01", 0.5, "minutes", "UTC", "2026-01-01T00:00:30+00:00"),
        ("2026-03-29T01:30:00", 1, "hours", "Europe/Rome", "2026-03-29T03:30:00+02:00"),
        (
            "2026-10-25T02:30:00+02:00",
            1,
            "hours",
            "Europe/Rome",
            "2026-10-25T02:30:00+01:00",
        ),
    ],
)
def test_add_uses_elapsed_duration_across_calendar_and_dst_boundaries(
    value: str, amount: float, unit: str, timezone: str, expected: str
) -> None:
    result = calculate_date_time(
        "add", value, timezone=timezone, amount=amount, unit=unit
    )
    assert result["datetime"] == expected


@pytest.mark.parametrize(
    ("value", "other", "unit", "expected_seconds", "expected_result"),
    [
        ("2026-01-01T10:00:00+02:00", "2026-01-01T08:00:00Z", "seconds", 0, 0),
        ("2026-03-29", "2026-03-30", "hours", 23 * 3600, 23),
        ("2026-01-02", "2026-01-01", "days", -86400, -1),
    ],
)
def test_difference_measures_other_minus_value_as_elapsed_time(
    value: str, other: str, unit: str, expected_seconds: float, expected_result: float
) -> None:
    result = calculate_date_time(
        "difference", value, other, timezone="Europe/Rome", unit=unit
    )
    assert result["total_seconds"] == expected_seconds
    assert result["result"] == expected_result
    assert result["unit"] == unit


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"operation": "unknown"}, "unsupported"),
        ({"operation": []}, "unsupported"),
        ({"value": ""}, "value must"),
        ({"value": "2025-02-29"}, "ISO 8601"),
        ({"value": "12:00"}, "ISO 8601"),
        ({"value": None}, "value must"),
        ({"timezone": "Unknown/Zone"}, "timezone"),
        ({"timezone": "../Europe/Rome"}, "timezone"),
        ({"timezone": None}, "timezone"),
        ({"unit": "months"}, "unit"),
        ({"unit": None}, "unit"),
        ({"operation": "difference", "other": ""}, "other must"),
        ({"operation": "add", "amount": True}, "finite number"),
        ({"operation": "add", "amount": "1"}, "finite number"),
        ({"operation": "add", "amount": float("nan")}, "finite number"),
        ({"operation": "add", "amount": float("inf")}, "finite number"),
        ({"operation": "add", "amount": 1e300}, "range"),
        (
            {"operation": "add", "value": "9999-12-31", "amount": 1, "unit": "days"},
            "range",
        ),
        ({"value": "2026-03-29T02:30:00", "timezone": "Europe/Rome"}, "does not exist"),
        ({"value": "2026-10-25T02:30:00", "timezone": "Europe/Rome"}, "ambiguous"),
    ],
)
def test_invalid_requests_report_actionable_errors(
    kwargs: dict[str, object], message: str
) -> None:
    request = {"operation": "convert", "value": "2026-01-01"} | kwargs
    with pytest.raises(DateTimeInputError, match=message):
        calculate_date_time(**request)  # type: ignore[arg-type]


def test_utc_does_not_require_a_timezone_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_zone(key: str):
        raise AssertionError(f"Unexpected timezone database access: {key}")

    monkeypatch.setattr(date_time, "ZoneInfo", unavailable_zone)
    assert (
        calculate_date_time("convert", "2026-01-01")["utc"]
        == "2026-01-01T00:00:00+00:00"
    )


@pytest.fixture
def bundled_tool(tmp_path: Path) -> Tool:
    loaded = Capabilities(None, default_capabilities=DefaultCapabilities()).load(
        work_root=tmp_path
    )
    return next(tool for tool in loaded.tools if tool.name == "date-time")


def test_bundled_tool_uses_default_timezone_and_unit(bundled_tool: Tool) -> None:
    result = bundled_tool.execute(
        operation="difference", value="2026-01-01", other="2026-01-02"
    )
    assert result["total_seconds"] == 86400
    assert result["timezone"] == "UTC"
    assert result["unit"] == "seconds"


def test_bundled_tool_returns_structured_input_errors(bundled_tool: Tool) -> None:
    assert bundled_tool.execute(operation="convert", value="invalid") == {
        "ok": False,
        "error_kind": "invalid_date_time",
        "error": "value must be an ISO 8601 date or datetime.",
    }


def test_missing_timezone_database_is_a_dependency_error(
    monkeypatch: pytest.MonkeyPatch, bundled_tool: Tool
) -> None:
    def unavailable_zone(key: str):
        raise date_time.ZoneInfoNotFoundError(key)

    monkeypatch.setattr(date_time, "ZoneInfo", unavailable_zone)
    monkeypatch.setattr(date_time, "available_timezones", lambda: set())

    with pytest.raises(DateTimeDependencyError, match="tzdata"):
        calculate_date_time("now", timezone="Europe/Rome")
    result = bundled_tool.execute(operation="now", timezone="Europe/Rome")
    assert result["ok"] is False
    assert result["error_kind"] == "missing_dependency"
    assert "tzdata" in result["hint"]
    assert (
        bundled_tool.execute(operation="convert", value="2026-01-01")["timezone"]
        == "UTC"
    )


def test_unknown_timezone_remains_an_input_error(
    monkeypatch: pytest.MonkeyPatch, bundled_tool: Tool
) -> None:
    monkeypatch.setattr(date_time, "available_timezones", lambda: {"UTC"})
    result = bundled_tool.execute(operation="now", timezone="Unknown/Zone")
    assert result["ok"] is False
    assert result["error_kind"] == "invalid_date_time"
    assert "Unknown timezone" in result["error"]


def test_bundled_tool_exposes_a_strict_provider_schema(bundled_tool: Tool) -> None:
    mapper = OpenAIFunctionToolDefinitionMapper(
        OpenAIChatCompletionAnnotationSchemaBuilder()
    )
    schema = mapper.map(bundled_tool, strict=True).get_parameters()
    assert schema["properties"]["operation"]["enum"] == [
        "now",
        "convert",
        "add",
        "difference",
    ]
    assert schema["properties"]["unit"]["enum"] == [
        "seconds",
        "minutes",
        "hours",
        "days",
    ]
    assert set(schema["properties"]) == {
        "operation",
        "value",
        "other",
        "timezone",
        "amount",
        "unit",
        "execution_mode",
        "wait_seconds",
    }
    assert set(schema["required"]) == set(schema["properties"])
