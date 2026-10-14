"""Crawl4AI lifecycle and output normalization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from playwright.async_api import TimeoutError as BrowserTimeoutError

from exmekh_lib.config.web import WEB_FORMAT, WEB_NAVIGATION
from exmekh_lib.config.web_text import WEB_TEXT

from .actions import execute_actions
from .ast import build_semantic_ast
from .content import extract_page_dates, select_primary_html
from .errors import WebNavigationError
from .security import (
    NavigationPolicy,
    build_route_guard,
    normalized_origin,
    validate_public_url,
)

if TYPE_CHECKING:
    from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy

OutputMode = Literal["markdown", "ast", "hybrid"]
MAX_MARKDOWN_CHARS = WEB_NAVIGATION.max_markdown_chars
CACHE_TTL_SECONDS = WEB_NAVIGATION.cache_ttl_seconds
CACHE_FORMAT_VERSION = WEB_FORMAT.cache_format_version
CHROMIUM_INSTALL_TIMEOUT_SECONDS = WEB_NAVIGATION.chromium_install_timeout_seconds
_CRAWL_ENV_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)


def crawl_page(
    *,
    url: str,
    output_mode: str,
    timeout_seconds: int,
    workroot: Path | None,
    actions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one isolated Crawl4AI browser lifecycle and return normalized output."""
    normalized_url = validate_public_url(url)
    mode = _validate_output_mode(output_mode)
    if (
        type(timeout_seconds) is not int
        or not 1 <= timeout_seconds <= WEB_NAVIGATION.max_timeout_seconds
    ):
        raise WebNavigationError(
            WEB_TEXT.timeout_seconds_must_be_between_and.format(
                web_navigation_max_timeout_seconds=WEB_NAVIGATION.max_timeout_seconds
            ),
            kind="invalid_timeout",
        )
    root = Path(workroot).resolve() if workroot is not None else Path.cwd().resolve()
    cache_root = root / WEB_FORMAT.cache_directory
    if actions is None:
        cached = _read_page_cache(cache_root=cache_root, url=normalized_url, mode=mode)
        if cached is not None:
            return cached
    with _CRAWL_ENV_LOCK:
        previous_directory = os.environ.get(WEB_FORMAT.crawler_directory_env_var)
        os.environ[WEB_FORMAT.crawler_directory_env_var] = str(cache_root)
        try:
            payload = _crawl_with_browser_recovery(
                url=normalized_url,
                output_mode=mode,
                timeout_seconds=timeout_seconds,
                cache_root=cache_root,
                actions=actions,
            )
        finally:
            if previous_directory is None:
                os.environ.pop(WEB_FORMAT.crawler_directory_env_var, None)
            else:
                os.environ[WEB_FORMAT.crawler_directory_env_var] = previous_directory
    if actions is None:
        _write_page_cache(
            cache_root=cache_root,
            url=normalized_url,
            mode=mode,
            payload=payload,
        )
    return payload


def _crawl_with_browser_recovery(
    *,
    url: str,
    output_mode: OutputMode,
    timeout_seconds: int,
    cache_root: Path,
    actions: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Install a missing headless browser once and retry the same crawl."""

    async def crawl() -> dict[str, Any]:
        return await _crawl_page(
            url=url,
            output_mode=output_mode,
            timeout_seconds=timeout_seconds,
            cache_root=cache_root,
            actions=actions,
        )

    try:
        return asyncio.run(crawl())
    except WebNavigationError as error:
        if error.kind != "browser_unavailable":
            raise
    _install_headless_chromium()
    return asyncio.run(crawl())


def _install_headless_chromium() -> None:
    """Provision Playwright's headless Chromium in the current Python environment."""
    command = [
        sys.executable,
        "-m",
        "playwright",
        "install",
        "chromium",
        "--only-shell",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=CHROMIUM_INSTALL_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise WebNavigationError(
            WEB_TEXT.could_not_install_chromium_automatically_run_python_m_playwright,
            kind="browser_unavailable",
        ) from error
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        suffix = (
            WEB_TEXT.playwright_reported.format(
                detail=detail[-1][: WEB_NAVIGATION.max_install_diagnostic_chars]
            )
            if detail
            else ""
        )
        raise WebNavigationError(
            WEB_TEXT.could_not_install_chromium_automatically_run_python_m_playwright
            + suffix,
            kind="browser_unavailable",
        )


@dataclass(frozen=True, slots=True)
class _CapturedPage:
    final_url: str
    html: str
    title: str
    actions: list[dict[str, Any]]
    actions_completed: bool


class _BrowserCapture:
    """Own browser hooks, navigation policy, and the observed final page."""

    def __init__(
        self,
        *,
        url: str,
        actions: Sequence[Mapping[str, Any]] | None,
        timeout_ms: int,
    ) -> None:
        self._policy = NavigationPolicy(initial_url=url)
        self._actions = actions
        self._timeout_ms = timeout_ms
        self._outcomes: list[dict[str, Any]] = []
        self._final_url: str | None = None
        self._final_html: str | None = None
        self._final_title: str | None = None
        self._action_error: WebNavigationError | None = None

    async def configure_context(self, page: Any, context: Any, **_: Any) -> Any:
        """Install the network boundary before the first navigation."""
        del context
        await page.context.route(
            "**/*", build_route_guard(page=page, policy=self._policy)
        )
        return page

    async def apply_actions(self, page: Any, **_: Any) -> Any:
        """Run the action sequence and capture the rendered destination."""
        await page.wait_for_load_state("load", timeout=self._timeout_ms)
        if self._actions:
            try:
                self._outcomes.extend(
                    await execute_actions(
                        page,
                        self._actions,
                        timeout_ms=self._timeout_ms,
                        navigation_policy=self._policy,
                    )
                )
            except WebNavigationError as error:
                self._action_error = error
                return page
        await page.wait_for_load_state("load", timeout=self._timeout_ms)
        self._final_url = str(page.url)
        self._final_html = str(await page.content())
        self._final_title = str(await page.title())
        return page

    def action_details(self) -> dict[str, object]:
        """Expose an independent snapshot of observed actions for failure recovery."""
        return (
            {"actions": deepcopy(self._outcomes)} if self._actions is not None else {}
        )

    def raise_if_failed(self) -> None:
        if self._action_error is not None:
            raise self._action_error
        try:
            self._policy.raise_if_violated()
        except WebNavigationError as error:
            error.add_details(self.action_details())
            raise

    def captured_page(self, result: Any, *, requested_url: str) -> _CapturedPage:
        """Validate the crawl and return the observed page data."""
        self.raise_if_failed()
        status = getattr(result, "status_code", None)
        http_failed = type(status) is int and 400 <= status < 600
        if not result.success or http_failed:
            fallback = (
                WEB_TEXT.remote_resource_returned_http.format(response_status=status)
                if http_failed
                else WEB_TEXT.unknown_crawl4ai_failure
            )
            error = WebNavigationError.from_crawl_failure(
                str(result.error_message or fallback),
                status=status,
                requested_url=requested_url,
            )
            error.add_details(self.action_details())
            raise error
        final_url = str(
            self._final_url or result.redirected_url or result.url or requested_url
        )
        validate_public_url(final_url)
        if normalized_origin(final_url) != self._policy.initial_origin:
            raise WebNavigationError(
                WEB_TEXT.main_frame_navigation_left_the_initial_origin.format(
                    final_url=final_url
                ),
                kind="cross_origin_navigation",
                details=self.action_details(),
            )
        return _CapturedPage(
            final_url=final_url,
            html=str(self._final_html or result.html or ""),
            title=str(self._final_title or (result.metadata or {}).get("title") or ""),
            actions=deepcopy(self._outcomes),
            actions_completed=len(self._outcomes) == len(self._actions or ())
            and all(
                item["success"] and item["effect_observed"] for item in self._outcomes
            ),
        )


async def _crawl_page(
    *,
    url: str,
    output_mode: OutputMode,
    timeout_seconds: int,
    cache_root: Path,
    actions: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Execute one asynchronous crawl and optional ordered action sequence."""
    cache_root.mkdir(parents=True, exist_ok=True)
    from crawl4ai import (
        AsyncWebCrawler,
        BrowserConfig,
        CacheMode,
        CrawlerRunConfig,
        DefaultMarkdownGenerator,
        content_filter_strategy,  # noqa: PLC0415
    )
    from crawl4ai.content_filter_strategy import PruningContentFilter  # noqa: PLC0415

    browser_config = BrowserConfig(headless=True, verbose=False, text_mode=False)
    filter_class = getattr(
        content_filter_strategy, "PruningContentFilterLXML", PruningContentFilter
    )
    markdown_generator = DefaultMarkdownGenerator(
        content_filter=filter_class(
            threshold=WEB_NAVIGATION.pruning_threshold,
            threshold_type="fixed",
            min_word_threshold=WEB_NAVIGATION.pruning_min_words,
        )
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.DISABLED,
        check_robots_txt=True,
        page_timeout=timeout_seconds * 1000,
        wait_until="domcontentloaded",
        remove_overlay_elements=True,
        remove_consent_popups=True,
        process_iframes=False,
        markdown_generator=markdown_generator,
    )
    capture = _BrowserCapture(
        url=url, actions=actions, timeout_ms=timeout_seconds * 1000
    )
    crawler = AsyncWebCrawler(config=browser_config, base_directory=str(cache_root))
    strategy = cast("AsyncPlaywrightCrawlerStrategy", crawler.crawler_strategy)
    strategy.set_hook("on_page_context_created", capture.configure_context)
    if actions:
        strategy.set_hook("before_retrieve_html", capture.apply_actions)
    try:
        await crawler.start()
        result = await crawler.arun(url=url, config=run_config)
        captured = capture.captured_page(result, requested_url=url)
    except WebNavigationError as error:
        capture.raise_if_failed()
        error.add_details(capture.action_details())
        raise
    except Exception as error:
        capture.raise_if_failed()
        if isinstance(error, (TimeoutError, BrowserTimeoutError)):
            raise WebNavigationError(
                WEB_TEXT.browser_crawl_failed.format(
                    error_type_name=type(error).__name__
                ),
                kind="crawl_failed",
                code="crawl_timeout",
                details=capture.action_details(),
            ) from error
        if WEB_TEXT.executable_doesn_t_exist in str(error):
            raise WebNavigationError(
                WEB_TEXT.chromium_is_unavailable_run_python_m_playwright_install_chromium,
                kind="browser_unavailable",
                details=capture.action_details(),
            ) from error
        raise WebNavigationError(
            WEB_TEXT.browser_crawl_failed.format(error_type_name=type(error).__name__),
            kind="crawl_failed",
            details=capture.action_details(),
        ) from error
    finally:
        await _close_crawler(crawler, sys.exception(), capture.action_details())
    primary_html, content_scope = select_primary_html(captured.html)
    regenerated_markdown = markdown_generator.generate_markdown(
        primary_html,
        base_url=captured.final_url,
    )
    markdown, markdown_truncated = _extract_markdown(
        regenerated_markdown,
        prefer_raw=content_scope in {"main", "article"},
    )
    payload: dict[str, Any] = {
        "requested_url": url,
        "final_url": captured.final_url,
        "title": captured.title,
        "status_code": result.status_code,
        "output_mode": output_mode,
        "untrusted_content": True,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "page_dates": extract_page_dates(result.metadata, captured.html),
        "content_scope": content_scope,
        "markdown_truncated": markdown_truncated,
    }
    if actions is not None:
        payload["actions"] = captured.actions
        payload["actions_completed"] = captured.actions_completed
    if output_mode in {"markdown", "hybrid"}:
        payload["markdown"] = markdown
        payload["markdown_chars"] = len(markdown)
    if output_mode in {"ast", "hybrid"}:
        payload["ast"] = build_semantic_ast(captured.html, base_url=captured.final_url)
    return payload


async def _close_crawler(
    crawler: Any, primary_error: BaseException | None, details: dict[str, object]
) -> None:
    """Keep the original failure when cleanup also fails, and report both causes."""
    try:
        await crawler.close()
    except Exception as error:
        message = WEB_TEXT.browser_cleanup_failed.format(
            error_type=type(error).__name__, error=error
        )
        if primary_error is None:
            raise WebNavigationError(
                message,
                kind="crawl_failed",
                code="browser_cleanup_failed",
                details=details,
            ) from error
        _LOGGER.exception(WEB_TEXT.browser_cleanup_failed_log)
        if isinstance(primary_error, WebNavigationError):
            primary_error.add_details({"cleanup_error": message})
        else:
            primary_error.add_note(message)


def _validate_output_mode(value: str) -> OutputMode:
    """Normalize and validate one public output-mode value."""
    mode = str(value).strip().lower()
    if mode not in {"markdown", "ast", "hybrid"}:
        raise WebNavigationError(
            WEB_TEXT.output_mode_must_be_markdown_ast_or_hybrid,
            kind="invalid_output_mode",
        )
    return mode  # type: ignore[return-value]


def _extract_markdown(result: Any, *, prefer_raw: bool = False) -> tuple[str, bool]:
    """Select fit Markdown with a raw fallback and apply the context budget."""
    markdown_result = getattr(result, "markdown", result)
    if isinstance(markdown_result, str):
        value = markdown_result
    else:
        fit_markdown = str(getattr(markdown_result, "fit_markdown", "") or "")
        raw_markdown = str(getattr(markdown_result, "raw_markdown", "") or "")
        minimum_fit_chars = min(
            WEB_NAVIGATION.maximum_fit_chars,
            max(
                WEB_NAVIGATION.minimum_fit_chars,
                len(raw_markdown) // WEB_NAVIGATION.fit_length_divisor,
            ),
        )
        value = (
            raw_markdown
            if prefer_raw
            else fit_markdown
            if len(fit_markdown.strip()) >= minimum_fit_chars
            else raw_markdown
        )
    stripped = value.strip()
    truncated = len(stripped) > MAX_MARKDOWN_CHARS
    if truncated:
        stripped = stripped[:MAX_MARKDOWN_CHARS].rstrip() + WEB_TEXT.content_truncated
    return stripped, truncated


def _cache_path(*, cache_root: Path, url: str, mode: str) -> Path:
    """Return the workroot-local cache path for one normalized page payload."""
    digest = hashlib.sha256(
        f"{CACHE_FORMAT_VERSION}\0{mode}\0{url}".encode("utf-8")
    ).hexdigest()
    return cache_root / WEB_FORMAT.page_cache_folder / f"{digest}.json"


def _read_page_cache(*, cache_root: Path, url: str, mode: str) -> dict[str, Any] | None:
    """Read one fresh normalized payload from the isolated filesystem cache."""
    path = _cache_path(cache_root=cache_root, url=url, mode=mode)
    try:
        if time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("final_url"), str):
        return None
    try:
        validate_public_url(payload["final_url"])
        if normalized_origin(payload["final_url"]) != normalized_origin(url):
            return None
    except WebNavigationError:
        return None
    return payload


def _write_page_cache(
    *, cache_root: Path, url: str, mode: str, payload: Mapping[str, Any]
) -> None:
    """Persist one normalized page payload atomically beneath the workroot."""
    path = _cache_path(cache_root=cache_root, url=url, mode=mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
