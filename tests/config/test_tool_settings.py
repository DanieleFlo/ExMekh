from dataclasses import replace
from pathlib import Path

import pytest

from qs.config import web
from qs.config.reasoning import FORMAL_LOGIC, MATH
from qs.config.runtime import RUNTIME
from qs.core.agents.tools.deep_tools import DeepTools
from qs.core.reasoning_tools import formal_logic
from qs.core.reasoning_tools import math as math_tools
from qs.core.tool_calls import memory_tools
from qs.core.web_navigation import crawler, search
from qs.core.web_navigation.errors import WebNavigationError


@pytest.mark.parametrize(
    "tool_name", ["search-web", "read-web-page", "interact-web-page"]
)
def test_web_tool_uses_shared_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tool_name: str
) -> None:
    settings = replace(
        web.WEB_NAVIGATION, default_timeout_seconds=7, default_search_results=2
    )
    monkeypatch.setattr(web, "WEB_NAVIGATION", settings)
    received = {}

    def capture(**arguments):
        received.update(arguments)
        return {"ok": True}

    def capture_search(query, **arguments):
        return capture(query=query, **arguments)

    monkeypatch.setattr(search, "search_web", capture_search)
    monkeypatch.setattr(crawler, "crawl_page", capture)
    tools_root = Path(__file__).parents[2] / "src/qs/capabilities/tools"
    tool = DeepTools([tools_root / tool_name], work_root=tmp_path).load()[0]
    arguments = (
        {"query": "topic"}
        if tool_name == "search-web"
        else {"url": "https://example.org"}
    )
    if tool_name == "interact-web-page":
        arguments["actions"] = [{"kind": "back"}]

    assert tool.execute(**arguments) == {"ok": True}
    assert received["timeout_seconds"] == 7
    if tool_name == "search-web":
        assert received["max_results"] == 2


def test_search_and_crawler_use_shared_timeout_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = replace(web.WEB_NAVIGATION, max_timeout_seconds=7)
    monkeypatch.setattr(search, "WEB_NAVIGATION", settings)
    monkeypatch.setattr(crawler, "WEB_NAVIGATION", settings)
    monkeypatch.setattr(crawler, "validate_public_url", lambda url: url)
    monkeypatch.setattr(
        search, "_retrieve_candidates", lambda query, *, timeout_seconds: []
    )

    assert (
        search.search_web("topic", max_results=1, timeout_seconds=7)["query"] == "topic"
    )
    with pytest.raises(WebNavigationError, match="between 1 and 7"):
        search.search_web("topic", max_results=1, timeout_seconds=8)
    with pytest.raises(WebNavigationError, match="between 1 and 7"):
        crawler.crawl_page(
            url="https://example.org",
            output_mode="markdown",
            timeout_seconds=8,
            workroot=None,
        )


def test_math_precision_validation_uses_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(math_tools, "MATH", replace(MATH, max_precision=4))

    assert math_tools.calculate("evaluate", "1/3", precision=4)["result"] == "0.3333"
    with pytest.raises(math_tools.MathInputError, match="between 2 and 4"):
        math_tools.calculate("evaluate", "1/3", precision=5)


def test_logic_validation_uses_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        formal_logic, "FORMAL_LOGIC", replace(FORMAL_LOGIC, max_premises=1)
    )

    assert (
        formal_logic.check_logic("consistency", {}, [{"const": True}])["status"]
        == "consistent"
    )
    with pytest.raises(formal_logic.FormalLogicError, match="at most 1 formulas"):
        formal_logic.check_logic("consistency", {}, [{"const": True}] * 2)


def test_history_pagination_uses_configured_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        memory_tools, "RUNTIME", replace(RUNTIME, memory_max_page_limit=3)
    )
    history = memory_tools.ExecutionHistory()
    for index in range(5):
        history.record("tool", f"call-{index}")

    assert len(history.select(start_position=1, limit=3).entries) == 3
    with pytest.raises(ValueError, match="between 2 and 3"):
        history.select(limit=4)
