"""Browser action contracts and observable execution outcomes."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from exmekh_lib.core.web_navigation import actions, security
from exmekh_lib.core.web_navigation.errors import WebNavigationError


class _Locator:
    def __init__(self, page: "_Page") -> None:
        self.page = page
        self.first = self

    async def click(self, *, timeout: int) -> None:
        self.page.html = "<p>Loaded</p>"

    async def fill(self, value: str, *, timeout: int) -> None:
        self.page.value = value

    async def select_option(self, value: str, *, timeout: int) -> None:
        self.page.value = value

    async def input_value(self, *, timeout: int) -> str:
        return self.page.value

    async def scroll_into_view_if_needed(self, *, timeout: int) -> None:
        self.page.scrolled = True


class _Page:
    def __init__(self) -> None:
        self.url = "https://example.org/start"
        self.html = "<p>Before</p>"
        self.value = ""
        self.scrolled = False
        self.waits: list[int] = []
        self.mouse = SimpleNamespace(wheel=self.wheel)

    def locator(self, selector: str) -> _Locator:
        return _Locator(self)

    async def content(self) -> str:
        return self.html

    async def wheel(self, horizontal: int, vertical: int) -> None:
        self.scrolled = vertical == 5000

    async def wait_for_selector(self, selector: str, *, timeout: int) -> None:
        self.waits.append(timeout)

    async def wait_for_timeout(self, milliseconds: int) -> None:
        self.waits.append(milliseconds)

    async def goto(self, destination: str, **kwargs: Any) -> None:
        self.url = destination

    async def go_back(self, **kwargs: Any) -> None:
        self.url = "https://example.org/start"

    async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
        self.waits.append(timeout)


def test_browser_actions_execute_in_order_with_observable_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    monkeypatch.setattr(actions, "validate_public_url", lambda url: url)
    page = _Page()
    policy = security.NavigationPolicy(initial_url=page.url)
    sequence = actions.validate_actions(
        [
            {"kind": "CLICK", "selector": "#open"},
            {"kind": "fill", "selector": "input", "value": ""},
            {"kind": "select", "selector": "select", "value": "first"},
            {"kind": "scroll", "selector": "#target", "delta_y": 9000},
            {"kind": "wait", "selector": "#ready"},
            {"kind": "wait", "milliseconds": 6000},
            {"kind": "navigate", "url": "https://example.org/next"},
            {"kind": "back"},
        ]
    )
    result = asyncio.run(
        actions.execute_actions(
            page, sequence, timeout_ms=300, navigation_policy=policy
        )
    )
    assert len(result) == len(sequence)
    assert all(item["success"] and item["effect_observed"] for item in result)
    assert page.scrolled
    assert page.url == "https://example.org/start"
    assert 5000 in page.waits


@pytest.mark.parametrize(
    ("input_actions", "message"),
    [
        ("bad", "JSON array"),
        ([4], "must be an object"),
        ([{"kind": "unknown"}], "unsupported kind"),
        ([{"kind": "back", "extra": 1}], "unexpected fields"),
        ([{"kind": "click"}], "requires string field 'selector'"),
        (
            [{"kind": "select", "selector": "#a", "value": ""}],
            "requires string field 'value'",
        ),
        ([{"kind": "scroll", "delta_y": True}], "delta_y must be numeric"),
        ([{"kind": "wait", "milliseconds": True}], "milliseconds must be an integer"),
        ([{"kind": "navigate", "url": ""}], "requires string field 'url'"),
    ],
)
def test_browser_actions_reject_invalid_sequences(
    input_actions: object, message: str
) -> None:
    with pytest.raises(WebNavigationError, match=message):
        actions.validate_actions(input_actions)  # type: ignore[arg-type]


def test_action_failure_stops_sequence_and_reports_error() -> None:
    class BrokenPage(_Page):
        async def wait_for_selector(self, selector: str, *, timeout: int) -> None:
            raise TimeoutError("not found")

    page = BrokenPage()
    outcomes = asyncio.run(
        actions.execute_actions(
            page,
            [{"kind": "wait", "selector": "#missing"}, {"kind": "back"}],
            timeout_ms=100,
        )
    )
    assert outcomes == [
        {
            "index": 0,
            "kind": "wait",
            "success": False,
            "effect_observed": False,
            "error": "TimeoutError",
            "error_code": "action_timeout",
            "message": "Action 0 (wait) failed: TimeoutError: not found.",
            "hint": "Read the page again and check the selector, visibility and current state. Earlier actions may already have completed; do not repeat an entire sequence blindly.",
        }
    ]


def test_cross_origin_action_failure_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    page = _Page()
    policy = security.NavigationPolicy(initial_url=page.url)
    with pytest.raises(WebNavigationError, match="left the initial origin"):
        asyncio.run(
            actions.execute_actions(
                page,
                [{"kind": "navigate", "url": "https://other.example/"}],
                timeout_ms=100,
                navigation_policy=policy,
            )
        )


def test_wait_for_page_effect_detects_no_change() -> None:
    page = _Page()
    assert (
        asyncio.run(
            actions._wait_for_page_effect(
                page,
                before=(page.url, page.html),
                timeout_ms=100,
            )
        )
        is False
    )


def test_recoverable_browser_error_is_returned_as_action_outcome() -> None:
    class FailingPage(_Page):
        async def wait_for_selector(self, selector: str, *, timeout: int) -> None:
            raise WebNavigationError("Invalid selector", kind="invalid_actions")

    outcomes = asyncio.run(
        actions.execute_actions(
            FailingPage(), [{"kind": "wait", "selector": "bad"}], timeout_ms=100
        )
    )
    assert outcomes[0]["error"] == "invalid_actions"
    assert outcomes[0]["error_code"] == "invalid_actions"
    assert outcomes[0]["hint"]


def test_failed_action_preserves_completed_actions_and_stops_before_later_effects():
    class BrokenPage(_Page):
        async def wait_for_selector(self, selector, *, timeout):
            raise TimeoutError("missing")

    page = BrokenPage()
    outcomes = asyncio.run(
        actions.execute_actions(
            page,
            [
                {"kind": "fill", "selector": "input", "value": "saved"},
                {"kind": "wait", "selector": "#missing"},
                {"kind": "fill", "selector": "input", "value": "must not run"},
            ],
            timeout_ms=100,
        )
    )
    assert len(outcomes) == 2
    assert outcomes[0]["success"] is True
    assert outcomes[1]["error_code"] == "action_timeout"
    assert "Earlier actions" in outcomes[1]["hint"]
    assert page.value == "saved"


def test_action_without_observed_effect_warns_before_repetition():
    page = _Page()
    outcomes = asyncio.run(
        actions.execute_actions(page, [{"kind": "back"}], timeout_ms=100)
    )
    assert outcomes[0]["success"] is True
    assert outcomes[0]["effect_observed"] is False
    assert outcomes[0]["error_code"] == "action_effect_unobserved"
    assert "before repeating" in outcomes[0]["hint"]


def test_navigation_rejection_carries_completed_actions(monkeypatch):
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    page = _Page()
    policy = security.NavigationPolicy(initial_url=page.url)
    with pytest.raises(WebNavigationError) as caught:
        asyncio.run(
            actions.execute_actions(
                page,
                [
                    {"kind": "fill", "selector": "input", "value": "saved"},
                    {"kind": "navigate", "url": "https://other.example/"},
                ],
                timeout_ms=100,
                navigation_policy=policy,
            )
        )
    assert caught.value.details["failed_action_index"] == 1
    assert caught.value.details["actions"][0]["success"] is True
    assert page.value == "saved"


def test_scroll_without_target_and_unknown_internal_action() -> None:
    page = _Page()
    assert asyncio.run(
        actions._execute_action(
            page,
            {"kind": "scroll", "delta_y": 5000},
            timeout_ms=100,
            navigation_policy=None,
        )
    )
    assert (
        asyncio.run(
            actions._execute_action(
                page,
                {"kind": "unsupported"},
                timeout_ms=100,
                navigation_policy=None,
            )
        )
        is False
    )


def test_actions_can_run_without_navigation_policy() -> None:
    page = _Page()
    outcomes = asyncio.run(
        actions.execute_actions(
            page,
            [{"kind": "navigate", "url": "https://example.org/next"}],
            timeout_ms=100,
        )
    )
    assert outcomes[0]["success"] is True
    assert page.url == "https://example.org/next"
