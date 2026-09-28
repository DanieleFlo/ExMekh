"""Discover public pages through bounded text and optional image search."""

from pathlib import Path

from qs.config.web import WEB_NAVIGATION
from qs.core.agents.tools.decorator import deep_tool
from qs.core.web_navigation.errors import WebNavigationError
from qs.core.web_navigation.tool_output import missing_dependency, navigation_error


@deep_tool(
    name="search-web",
    description=(
        "Search public web pages. Results are unverified discovery leads; read a "
        "page before citing it. image_path optionally reranks results against a "
        "local image inside the work root."
    ),
)
def run(
    work_root: Path,
    query: str,
    max_results: int = WEB_NAVIGATION.default_search_results,
    timeout_seconds: int = WEB_NAVIGATION.default_timeout_seconds,
    image_path: str | None = None,
) -> dict[str, object]:
    try:
        from qs.core.web_navigation.search import search_web, search_web_multimodal

        if image_path is None:
            return search_web(
                query, max_results=max_results, timeout_seconds=timeout_seconds
            )
        from qs.core.web_navigation.image_discovery import load_local_query_image

        image = load_local_query_image(image_path=image_path, workroot=work_root)
        return search_web_multimodal(
            query,
            local_image=image,
            max_results=max_results,
            timeout_seconds=timeout_seconds,
        )
    except ModuleNotFoundError as error:
        return missing_dependency(error)
    except WebNavigationError as error:
        return navigation_error(error)
