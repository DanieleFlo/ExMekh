"""Dependency failures share one model-facing result across tools."""

from exmekh_lib.utils.tool_errors import missing_dependency


def test_missing_dependency_preserves_module_and_install_hint() -> None:
    result = missing_dependency(
        ModuleNotFoundError("No module named 'crawl4ai'", name="crawl4ai")
    )
    assert result == {
        "ok": False,
        "error_kind": "missing_dependency",
        "error": "Missing exmekh_lib runtime module: crawl4ai.",
        "hint": "Install the ExMekh-Lib wheel with its declared dependencies in this environment.",
    }
