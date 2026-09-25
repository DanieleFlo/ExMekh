"""Convert web-navigation failures into actionable tool results."""

from qs.core.web_navigation.errors import WebNavigationError


def navigation_error(error: WebNavigationError) -> dict[str, object]:
    return {"ok": False, "error_kind": error.kind, "error": str(error)}


def missing_dependency(error: ModuleNotFoundError) -> dict[str, object]:
    return {
        "ok": False,
        "error_kind": "missing_dependency",
        "error": f"Missing qs runtime module: {error.name}.",
        "hint": "Install the qs wheel with its declared dependencies in this environment.",
    }
