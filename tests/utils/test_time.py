from datetime import UTC, datetime, timedelta, timezone
from sys import float_info

import pytest

from exmekh_lib.utils.time import (
    format_duration,
    format_timestamp,
    format_timing_fields,
)


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "0s"),
        (0.000001, "0s"),
        (0.04, "0s"),
        (0.05, "0.1s"),
        (1.05, "1.1s"),
        (3.14159, "3.1s"),
        (59.94, "59.9s"),
        (59.95, "1m"),
        (60, "1m"),
        (68, "1m 8s"),
        (3599.95, "1h"),
        (3600, "1h"),
        (3600.1, "1h 0.1s"),
        (3723.456789, "1h 2m 3.5s"),
        (90000, "25h"),
    ],
)
def test_duration_uses_hours_minutes_and_at_most_tenths(seconds, expected):
    assert format_duration(seconds) == expected


def test_timestamp_uses_utc_with_second_precision():
    value = datetime(
        2026, 10, 17, 16, 32, 8, 987654, tzinfo=timezone(timedelta(hours=2))
    )
    assert format_timestamp(value) == "2026-10-17T14:32:08Z"


def test_timestamp_rejects_an_unknown_timezone_instead_of_using_host_local_time():
    with pytest.raises(ValueError, match="timezone-aware"):
        format_timestamp(datetime(2026, 10, 17, 14, 32, 8))


@pytest.mark.parametrize(
    "value",
    [
        -1,
        float("inf"),
        float("nan"),
        True,
        None,
        "1",
        pytest.param(10**400, id="oversized-int"),
    ],
)
def test_duration_rejects_invalid_runtime_measurements(value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        format_duration(value)


def test_finite_large_duration_does_not_overflow_during_rounding():
    formatted = format_duration(float_info.max)
    assert "h" in formatted
    assert "inf" not in formatted


def test_huge_legacy_measurements_do_not_prevent_rendering_other_metadata():
    value = 10**400
    assert format_timing_fields({"status": "completed", "elapsed-seconds": value}) == {
        "status": "completed",
        "elapsed-seconds": value,
    }


def test_timing_fields_are_formatted_without_changing_the_source():
    fields = {
        "started-at": "2026-10-17T16:32:08.987654+02:00",
        "finished-at": datetime(2026, 10, 17, 14, 33, 9, 123456, tzinfo=UTC),
        "elapsed-seconds": 60.13579,
        "status": "completed",
    }
    assert format_timing_fields(fields) == {
        "started-at": "2026-10-17T14:32:08Z",
        "finished-at": "2026-10-17T14:33:09Z",
        "elapsed": "1m 0.1s",
        "status": "completed",
    }
    assert fields["elapsed-seconds"] == 60.13579
    assert fields["started-at"] == "2026-10-17T16:32:08.987654+02:00"


@pytest.mark.parametrize("value", [None, "then", "2026-10-17T14:32:08", 123])
def test_legacy_timestamps_without_a_known_timezone_are_preserved(value):
    assert format_timing_fields({"started-at": value}) == {"started-at": value}


@pytest.mark.parametrize("value", [None, "unknown", -1, float("inf")])
def test_unknown_legacy_durations_are_preserved(value):
    assert format_timing_fields({"elapsed-seconds": value}) == {
        "elapsed-seconds": value
    }


def test_legacy_numeric_duration_strings_are_formatted():
    assert format_timing_fields({"elapsed-seconds": "7.123456"}) == {"elapsed": "7.1s"}


def test_legacy_timestamp_outside_the_utc_range_is_preserved():
    value = "0001-01-01T00:00:00+14:00"
    assert format_timing_fields({"started-at": value}) == {"started-at": value}
