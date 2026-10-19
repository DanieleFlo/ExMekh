"""Apply a bounded sequence of browser actions to one public page."""

from pathlib import Path
from typing import Annotated, Literal

from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.config.web import WEB_NAVIGATION
from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.web_navigation.actions import BrowserAction
from exmekh_lib.core.web_navigation.errors import WebNavigationError
from exmekh_lib.core.web_navigation.tool_output import navigation_error
from exmekh_lib.utils.tool_errors import missing_dependency


@deep_tool(
    name=DEFAULT_CAPABILITIES.interact_web_page,
    description=(WEB_NAVIGATION.interact_web_page_description),
)
def run(
    work_root: Path,
    url: str,
    actions: Annotated[
        list[BrowserAction],
        {"minItems": 1, "maxItems": WEB_NAVIGATION.max_actions},
        WEB_NAVIGATION.action_guidance,
    ],
    output_mode: Literal["markdown", "ast", "hybrid"] = "ast",
    timeout_seconds: int = WEB_NAVIGATION.default_timeout_seconds,
) -> dict[str, object]:
    try:
        from exmekh_lib.core.web_navigation.actions import validate_actions
        from exmekh_lib.core.web_navigation.crawler import crawl_page

        validated_actions = validate_actions(actions)
        return crawl_page(
            url=url,
            actions=validated_actions,
            output_mode=output_mode,
            timeout_seconds=timeout_seconds,
            workroot=work_root,
        )
    except ModuleNotFoundError as error:
        return missing_dependency(error)
    except WebNavigationError as error:
        return navigation_error(error)
