"""Convert web-navigation failures into actionable tool results."""

from exmekh_lib.core.web_navigation.errors import WebNavigationError


def navigation_error(error: WebNavigationError) -> dict[str, object]:
    return {"ok": False, "error_kind": error.kind, "error": str(error)}


def missing_dependency(error: ModuleNotFoundError) -> dict[str, object]:
    return {
        "ok": False,
        "error_kind": "missing_dependency",
        "error": f"Missing exmekh_lib runtime module: {error.name}.",
        "hint": "Install the ExMekh-Lib wheel with its declared dependencies in this environment.",
    }
