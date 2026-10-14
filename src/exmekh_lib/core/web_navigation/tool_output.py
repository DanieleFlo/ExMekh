"""Convert web-navigation failures into actionable tool results."""

from exmekh_lib.core.web_navigation.errors import WebNavigationError
from exmekh_lib.utils.tool_errors import tool_error_result


def navigation_error(error: WebNavigationError) -> dict[str, object]:
    return tool_error_result(error, error.kind)
