"""Convert web-navigation failures into actionable tool results."""

from exmekh_lib.core.web_navigation.errors import WebNavigationError


def navigation_error(error: WebNavigationError) -> dict[str, object]:
    return {"ok": False, "error_kind": error.kind, "error": str(error)}
