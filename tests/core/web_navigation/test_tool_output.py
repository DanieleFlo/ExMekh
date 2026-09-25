"""Stable error objects returned to the model by web tools."""

from qs.core.web_navigation.errors import WebNavigationError
from qs.core.web_navigation.tool_output import missing_dependency, navigation_error


def test_web_errors_preserve_kind_and_install_hint() -> None:
    blocked = navigation_error(WebNavigationError("Blocked URL", kind="blocked_url"))
    assert blocked == {"ok": False, "error_kind": "blocked_url", "error": "Blocked URL"}

    dependency = missing_dependency(
        ModuleNotFoundError("No module named 'crawl4ai'", name="crawl4ai")
    )
    assert dependency["error_kind"] == "missing_dependency"
    assert "crawl4ai" in dependency["error"]
    assert "qs wheel" in dependency["hint"]
