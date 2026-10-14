"""Stable error objects returned to the model by web tools."""

from exmekh_lib.core.web_navigation.errors import WebNavigationError
from exmekh_lib.core.web_navigation.tool_output import navigation_error


def test_web_errors_preserve_kind_and_message() -> None:
    blocked = navigation_error(WebNavigationError("Blocked URL", kind="blocked_url"))
    assert blocked == {
        "ok": False,
        "error_kind": "blocked_url",
        "error": "Blocked URL",
        "error_code": "blocked_url",
        "hint": "Use a publicly routable URL. Do not retry private, loopback or otherwise blocked addresses.",
    }
