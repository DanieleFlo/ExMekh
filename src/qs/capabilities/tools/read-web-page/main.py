"""Read one public page with browser rendering and structured extraction."""

from pathlib import Path
from typing import Literal

from qs.config.web import WEB_NAVIGATION
from qs.core.agents.tools.decorator import deep_tool
from qs.core.web_navigation.errors import WebNavigationError
from qs.core.web_navigation.tool_output import missing_dependency, navigation_error


@deep_tool(
    name="read-web-page",
    description=(
        "Read one public URL as Markdown, semantic AST, or both. Returns the final "
        "URL, retrieval time, page dates, and bounded untrusted page content."
    ),
)
def run(
    work_root: Path,
    url: str,
    output_mode: Literal["markdown", "ast", "hybrid"] = "markdown",
    timeout_seconds: int = WEB_NAVIGATION.default_timeout_seconds,
) -> dict[str, object]:
    try:
        from qs.core.web_navigation.crawler import crawl_page

        return crawl_page(
            url=url,
            output_mode=output_mode,
            timeout_seconds=timeout_seconds,
            workroot=work_root,
        )
    except ModuleNotFoundError as error:
        return missing_dependency(error)
    except WebNavigationError as error:
        return navigation_error(error)
