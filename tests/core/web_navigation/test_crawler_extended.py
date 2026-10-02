"""Crawler lifecycle, page capture, cache, and browser recovery."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from exmekh_lib.core.web_navigation import crawler
from exmekh_lib.core.web_navigation.errors import WebNavigationError

URL = "https://example.org/page"


class _Page:
    def __init__(self) -> None:
        self.url = URL
        self.context = SimpleNamespace(route=self.route)
        self.routes: list[tuple[str, object]] = []

    async def route(self, pattern: str, guard: object) -> None:
        self.routes.append((pattern, guard))

    async def wait_for_load_state(self, state: str, *, timeout: int) -> None:
        assert state == "load" and timeout > 0

    async def content(self) -> str:
        return (
            "<html><body><main><h1>Article</h1><p>"
            + "Long text. " * 30
            + "</p></main></body></html>"
        )

    async def title(self) -> str:
        return "Captured title"


class _FakeCrawler:
    result: object
    error: Exception | None = None
    instances: list["_FakeCrawler"] = []

    def __init__(self, *, config: object, base_directory: str) -> None:
        self.hooks: dict[str, object] = {}
        self.crawler_strategy = SimpleNamespace(set_hook=self.set_hook)
        self.closed = False
        self.page = _Page()
        self.instances.append(self)

    def set_hook(self, name: str, callback: object) -> None:
        self.hooks[name] = callback

    async def start(self) -> None:
        if self.error is not None:
            raise self.error

    async def arun(self, *, url: str, config: object) -> object:
        await self.hooks["on_page_context_created"](self.page, None)  # type: ignore[operator]
        if "before_retrieve_html" in self.hooks:
            await self.hooks["before_retrieve_html"](self.page)  # type: ignore[operator]
        return self.result

    async def close(self) -> None:
        self.closed = True


def _install_fake_crawl4ai(monkeypatch: pytest.MonkeyPatch) -> None:
    class Generator:
        def __init__(self, **kwargs: object) -> None:
            pass

        def generate_markdown(self, html: str, *, base_url: str) -> object:
            assert "Long text." in html
            return SimpleNamespace(
                markdown=SimpleNamespace(
                    fit_markdown="Short", raw_markdown="Full article. " * 20
                )
            )

    fake_module = SimpleNamespace(
        AsyncWebCrawler=_FakeCrawler,
        BrowserConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        CacheMode=SimpleNamespace(DISABLED="disabled"),
        CrawlerRunConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        DefaultMarkdownGenerator=Generator,
        content_filter_strategy=SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "crawl4ai", fake_module)
    monkeypatch.setitem(
        sys.modules,
        "crawl4ai.content_filter_strategy",
        SimpleNamespace(PruningContentFilter=lambda **kwargs: object()),
    )
    monkeypatch.setattr(crawler, "validate_public_url", lambda url: url)
    monkeypatch.setattr(
        crawler, "normalized_origin", lambda url: ("https", "example.org", 443)
    )
    monkeypatch.setattr(
        crawler.NavigationPolicy, "__init__", lambda self, **kwargs: None
    )
    monkeypatch.setattr(
        crawler.NavigationPolicy, "raise_if_violated", lambda self: None
    )
    monkeypatch.setattr(
        crawler.NavigationPolicy,
        "initial_origin",
        property(lambda self: ("https", "example.org", 443)),
    )
    _FakeCrawler.instances = []
    _FakeCrawler.error = None
    _FakeCrawler.result = SimpleNamespace(
        success=True,
        error_message=None,
        redirected_url=None,
        url=URL,
        html="<html><body><main><h1>Article</h1><p>"
        + "Long text. " * 30
        + "</p></main></body></html>",
        metadata={"title": "Metadata title"},
        status_code=200,
    )


def test_crawler_builds_all_output_modes_and_installs_route_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_crawl4ai(monkeypatch)
    for mode in ("markdown", "ast", "hybrid"):
        payload = asyncio.run(
            crawler._crawl_page(
                url=URL,
                output_mode=mode,
                timeout_seconds=2,
                cache_root=tmp_path,
                actions=None,
            )
        )
        assert payload["title"] == "Metadata title"
        assert payload["content_scope"] == "main"
        assert ("markdown" in payload) == (mode in {"markdown", "hybrid"})
        assert ("ast" in payload) == (mode in {"ast", "hybrid"})
        assert _FakeCrawler.instances[-1].closed
        assert _FakeCrawler.instances[-1].page.routes[0][0] == "**/*"


def test_crawler_action_capture_reports_completed_outcomes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_crawl4ai(monkeypatch)

    async def fake_execute(*args: object, **kwargs: object) -> list[dict[str, object]]:
        return [{"index": 0, "kind": "wait", "success": True, "effect_observed": True}]

    monkeypatch.setattr(crawler, "execute_actions", fake_execute)
    payload = asyncio.run(
        crawler._crawl_page(
            url=URL,
            output_mode="markdown",
            timeout_seconds=2,
            cache_root=tmp_path,
            actions=[{"kind": "wait", "milliseconds": 1}],
        )
    )
    assert payload["actions_completed"] is True
    assert payload["actions"][0]["kind"] == "wait"
    assert payload["title"] == "Captured title"


def test_crawler_propagates_action_navigation_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_crawl4ai(monkeypatch)

    async def blocked(*args: object, **kwargs: object) -> None:
        raise WebNavigationError(
            "Moved to another origin", kind="cross_origin_navigation"
        )

    monkeypatch.setattr(crawler, "execute_actions", blocked)
    with pytest.raises(WebNavigationError) as failure:
        asyncio.run(
            crawler._crawl_page(
                url=URL,
                output_mode="markdown",
                timeout_seconds=2,
                cache_root=tmp_path,
                actions=[{"kind": "navigate", "url": "https://other.example/"}],
            )
        )
    assert failure.value.kind == "cross_origin_navigation"


def test_capture_hook_without_actions_still_records_rendered_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        crawler.NavigationPolicy, "__init__", lambda self, **kwargs: None
    )
    capture = crawler._BrowserCapture(url=URL, actions=None, timeout_ms=100)
    page = _Page()
    asyncio.run(capture.apply_actions(page))
    assert capture._final_url == URL
    assert "Article" in capture._final_html


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (RuntimeError("Executable doesn't exist"), "browser_unavailable"),
        (RuntimeError("network failed"), "crawl_failed"),
        (
            WebNavigationError("blocked", kind="cross_origin_navigation"),
            "cross_origin_navigation",
        ),
    ],
)
def test_crawler_classifies_browser_failures_and_closes_resources(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, error: Exception, kind: str
) -> None:
    _install_fake_crawl4ai(monkeypatch)
    _FakeCrawler.error = error
    with pytest.raises(WebNavigationError) as failure:
        asyncio.run(
            crawler._crawl_page(
                url=URL,
                output_mode="markdown",
                timeout_seconds=2,
                cache_root=tmp_path,
                actions=None,
            )
        )
    assert failure.value.kind == kind
    assert _FakeCrawler.instances[-1].closed


def test_crawler_rejects_unsuccessful_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_crawl4ai(monkeypatch)
    _FakeCrawler.result.success = False
    _FakeCrawler.result.error_message = "Page unavailable"
    with pytest.raises(WebNavigationError, match="Page unavailable"):
        asyncio.run(
            crawler._crawl_page(
                url=URL,
                output_mode="markdown",
                timeout_seconds=2,
                cache_root=tmp_path,
                actions=None,
            )
        )


def test_page_capture_rejects_failed_actions_and_cross_origin_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(crawler, "validate_public_url", lambda url: url)
    monkeypatch.setattr(
        crawler, "normalized_origin", lambda url: ("https", url.split("/")[2], 443)
    )
    capture = crawler._BrowserCapture(url=URL, actions=None, timeout_ms=100)
    result = SimpleNamespace(
        success=True,
        error_message=None,
        redirected_url="https://other.example/",
        url=URL,
        html="",
        metadata={},
    )
    with pytest.raises(WebNavigationError) as failure:
        capture.captured_page(result, requested_url=URL)
    assert failure.value.kind == "cross_origin_navigation"


def test_crawler_input_validation_and_cache_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(crawler, "validate_public_url", lambda url: url)
    monkeypatch.setattr(
        crawler, "normalized_origin", lambda url: ("https", "example.org", 443)
    )
    with pytest.raises(WebNavigationError) as failure:
        crawler.crawl_page(
            url=URL, output_mode="invalid", timeout_seconds=2, workroot=tmp_path
        )
    assert failure.value.kind == "invalid_output_mode"
    with pytest.raises(WebNavigationError) as failure:
        crawler.crawl_page(
            url=URL, output_mode="markdown", timeout_seconds=0, workroot=tmp_path
        )
    assert failure.value.kind == "invalid_timeout"

    calls = 0

    async def fake_crawl(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"final_url": URL, "markdown": "content"}

    monkeypatch.setattr(crawler, "_crawl_page", fake_crawl)
    monkeypatch.delenv("CRAWL4_AI_BASE_DIRECTORY", raising=False)
    first = crawler.crawl_page(
        url=URL, output_mode="markdown", timeout_seconds=2, workroot=tmp_path
    )
    second = crawler.crawl_page(
        url=URL, output_mode="markdown", timeout_seconds=2, workroot=tmp_path
    )
    assert first == second
    assert calls == 1
    assert "CRAWL4_AI_BASE_DIRECTORY" not in os.environ

    path = crawler._cache_path(
        cache_root=tmp_path / ".exmekh_lib" / "web-cache", url=URL, mode="markdown"
    )
    os.utime(path, (time.time() - 2000, time.time() - 2000))
    crawler.crawl_page(
        url=URL, output_mode="markdown", timeout_seconds=2, workroot=tmp_path
    )
    assert calls == 2


def test_cache_rejects_invalid_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(crawler, "validate_public_url", lambda url: url)
    path = crawler._cache_path(cache_root=tmp_path, url=URL, mode="ast")
    path.parent.mkdir(parents=True)
    for value in ["broken JSON", json.dumps([]), json.dumps({"final_url": 1})]:
        path.write_text(value, encoding="utf-8")
        assert (
            crawler._read_page_cache(cache_root=tmp_path, url=URL, mode="ast") is None
        )

    def blocked(url: str) -> None:
        raise WebNavigationError("Blocked URL", kind="blocked_url")

    path.write_text(json.dumps({"final_url": URL}), encoding="utf-8")
    monkeypatch.setattr(crawler, "validate_public_url", blocked)
    assert crawler._read_page_cache(cache_root=tmp_path, url=URL, mode="ast") is None


def test_markdown_selection_and_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    assert crawler._extract_markdown(" plain ") == ("plain", False)
    result = SimpleNamespace(
        markdown=SimpleNamespace(fit_markdown="Brief", raw_markdown="A" * 1000)
    )
    assert crawler._extract_markdown(result)[0] == "A" * 1000
    assert crawler._extract_markdown(result, prefer_raw=True)[0] == "A" * 1000
    result.markdown.fit_markdown = "B" * 600
    assert crawler._extract_markdown(result)[0] == "B" * 600
    monkeypatch.setattr(crawler, "MAX_MARKDOWN_CHARS", 5)
    text, truncated = crawler._extract_markdown("abcdefgh")
    assert text == "abcde\n\n[Content truncated]"
    assert truncated


def test_browser_install_process_errors_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unable(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="playwright", timeout=1)

    monkeypatch.setattr(crawler.subprocess, "run", unable)
    with pytest.raises(WebNavigationError) as failure:
        crawler._install_headless_chromium()
    assert failure.value.kind == "browser_unavailable"

    monkeypatch.setattr(
        crawler.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    crawler._install_headless_chromium()
