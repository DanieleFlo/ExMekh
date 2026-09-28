"""Validation and execution of the bounded browser action language."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from qs.config.web import WEB_NAVIGATION

from .errors import WebNavigationError
from .security import NavigationPolicy, validate_public_url

MAX_ACTIONS = WEB_NAVIGATION.max_actions
ACTION_KINDS = {"click", "fill", "select", "scroll", "wait", "navigate", "back"}


def validate_actions(actions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return normalized actions after strict shape and budget validation."""
    if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
        raise WebNavigationError(
            "Actions must be a JSON array.", kind="invalid_actions"
        )
    if not actions or len(actions) > MAX_ACTIONS:
        raise WebNavigationError(
            f"Actions must contain between 1 and {MAX_ACTIONS} items.",
            kind="invalid_actions",
        )
    normalized: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if not isinstance(action, Mapping):
            raise WebNavigationError(
                f"Action {index} must be an object.", kind="invalid_actions"
            )
        item = dict(action)
        kind = str(item.get("kind", "")).strip().lower()
        if kind not in ACTION_KINDS:
            raise WebNavigationError(
                f"Action {index} has unsupported kind: {kind or '<empty>'}.",
                kind="invalid_actions",
            )
        item["kind"] = kind
        _validate_action(index=index, action=item)
        normalized.append(item)
    return normalized


def _validate_action(*, index: int, action: dict[str, Any]) -> None:
    """Validate the fields required by one normalized action."""
    kind = action["kind"]
    allowed_fields = {
        "click": {"kind", "selector"},
        "fill": {"kind", "selector", "value"},
        "select": {"kind", "selector", "value"},
        "scroll": {"kind", "selector", "delta_y"},
        "wait": {"kind", "selector", "milliseconds"},
        "navigate": {"kind", "url"},
        "back": {"kind"},
    }
    unexpected = action.keys() - allowed_fields[kind]
    if unexpected:
        raise WebNavigationError(
            f"Action {index} has unexpected fields: {', '.join(sorted(unexpected))}.",
            kind="invalid_actions",
        )
    if kind in {"click", "fill", "select"}:
        _require_text(action, "selector", index=index)
    if kind in {"fill", "select"}:
        _require_text(action, "value", index=index, allow_empty=kind == "fill")
    if kind == "scroll":
        delta = action.get("delta_y", WEB_NAVIGATION.default_scroll_pixels)
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            raise WebNavigationError(
                f"Action {index} delta_y must be numeric.", kind="invalid_actions"
            )
        action["delta_y"] = max(
            -WEB_NAVIGATION.max_scroll_pixels,
            min(WEB_NAVIGATION.max_scroll_pixels, int(delta)),
        )
    if kind == "wait":
        selector = action.get("selector")
        milliseconds = action.get("milliseconds")
        if bool(selector) == (milliseconds is not None):
            raise WebNavigationError(
                f"Action {index} wait requires exactly one of selector or milliseconds.",
                kind="invalid_actions",
            )
        if milliseconds is not None:
            if isinstance(milliseconds, bool) or not isinstance(milliseconds, int):
                raise WebNavigationError(
                    f"Action {index} milliseconds must be an integer.",
                    kind="invalid_actions",
                )
            action["milliseconds"] = max(
                0, min(WEB_NAVIGATION.max_action_wait_ms, milliseconds)
            )
    if kind == "navigate":
        action["url"] = validate_public_url(_require_text(action, "url", index=index))


def _require_text(
    action: Mapping[str, Any],
    field: str,
    *,
    index: int,
    allow_empty: bool = False,
) -> str:
    """Return one string action field or raise a precise validation error."""
    value = action.get(field)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise WebNavigationError(
            f"Action {index} requires string field '{field}'.", kind="invalid_actions"
        )
    return value


async def execute_actions(
    page: Any,
    actions: Sequence[Mapping[str, Any]],
    *,
    timeout_ms: int,
    navigation_policy: NavigationPolicy | None = None,
) -> list[dict[str, Any]]:
    """Execute actions in order and stop at the first observable failure."""
    outcomes: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        kind = str(action["kind"])
        try:
            effect_observed = await _execute_action(
                page,
                action,
                timeout_ms=timeout_ms,
                navigation_policy=navigation_policy,
            )
            if navigation_policy is not None:
                navigation_policy.raise_if_violated()
        except WebNavigationError as error:
            if error.kind in {
                "cross_origin_navigation",
                "redirect_limit_exceeded",
            }:
                raise
            outcomes.append(
                {
                    "index": index,
                    "kind": kind,
                    "success": False,
                    "effect_observed": False,
                    "error": error.kind,
                }
            )
            break
        except Exception as error:
            outcomes.append(
                {
                    "index": index,
                    "kind": kind,
                    "success": False,
                    "effect_observed": False,
                    "error": type(error).__name__,
                }
            )
            break
        outcomes.append(
            {
                "index": index,
                "kind": kind,
                "success": True,
                "effect_observed": effect_observed,
            }
        )
    return outcomes


async def _execute_action(
    page: Any,
    action: Mapping[str, Any],
    *,
    timeout_ms: int,
    navigation_policy: NavigationPolicy | None,
) -> bool:
    """Execute one action and report whether its intended effect was observed."""
    kind = str(action["kind"])
    if kind == "click":
        before = await _page_state(page)
        await page.locator(action["selector"]).first.click(timeout=timeout_ms)
        return await _wait_for_page_effect(page, before=before, timeout_ms=timeout_ms)
    elif kind == "fill":
        locator = page.locator(action["selector"]).first
        await locator.fill(action["value"], timeout=timeout_ms)
        return await locator.input_value(timeout=timeout_ms) == action["value"]
    elif kind == "select":
        locator = page.locator(action["selector"]).first
        await locator.select_option(action["value"], timeout=timeout_ms)
        return await locator.input_value(timeout=timeout_ms) == action["value"]
    elif kind == "scroll":
        selector = action.get("selector")
        if selector:
            await page.locator(selector).first.scroll_into_view_if_needed(
                timeout=timeout_ms
            )
        await page.mouse.wheel(0, int(action["delta_y"]))
        return True
    elif kind == "wait":
        if action.get("selector"):
            await page.wait_for_selector(action["selector"], timeout=timeout_ms)
        else:
            await page.wait_for_timeout(int(action["milliseconds"]))
        return True
    elif kind == "navigate":
        destination = str(action["url"])
        if navigation_policy is not None:
            destination = navigation_policy.validate_action_url(destination)
        previous_url = str(page.url)
        await page.goto(destination, wait_until="domcontentloaded", timeout=timeout_ms)
        await page.wait_for_load_state("load", timeout=timeout_ms)
        return str(page.url) != previous_url or str(page.url) == destination
    elif kind == "back":
        previous_url = str(page.url)
        await page.go_back(wait_until="domcontentloaded", timeout=timeout_ms)
        await page.wait_for_load_state("load", timeout=timeout_ms)
        return str(page.url) != previous_url
    return False


async def _page_state(page: Any) -> tuple[str, str]:
    """Return a compact URL and rendered-HTML observation for effect detection."""
    return str(page.url), str(await page.content())


async def _wait_for_page_effect(
    page: Any, *, before: tuple[str, str], timeout_ms: int
) -> bool:
    """Wait until a click changes URL or rendered DOM and the state stabilizes."""
    deadline_ms = min(timeout_ms, WEB_NAVIGATION.action_effect_timeout_ms)
    elapsed_ms = 0
    previous = before
    changed = False
    stable_observations = 0
    while elapsed_ms <= deadline_ms:
        current = await _page_state(page)
        changed = changed or current != before
        if changed and current == previous:
            stable_observations += 1
            if stable_observations >= 2:
                return True
        else:
            stable_observations = 0
        previous = current
        await page.wait_for_timeout(WEB_NAVIGATION.action_poll_interval_ms)
        elapsed_ms += WEB_NAVIGATION.action_poll_interval_ms
    return changed
