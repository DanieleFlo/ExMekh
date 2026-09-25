import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.capabilities.defaults import BUILTIN_CAPABILITIES_PATH
from qs.core.web_navigation import search, security
from qs.core.web_navigation.actions import validate_actions
from qs.core.web_navigation.errors import WebNavigationError


def test_search_canonicalization_keeps_ipv6_valid_and_removes_tracking() -> None:
    assert search._canonicalize_url(
        "https://[2606:4700:4700::1111]:443/a/?utm_source=x&q=1"
    ) == ("https://[2606:4700:4700::1111]:443/a?q=1")


def test_search_with_stubbed_candidates_and_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search,
        "_retrieve_candidates",
        lambda *_args, **_kwargs: [
            {"title": "A", "url": "https://example.org/a", "snippet": "alpha"},
            {"title": "B", "url": "https://example.org/b", "snippet": "beta"},
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "qs.core.web_navigation.ranking",
        SimpleNamespace(
            rank_results=lambda *_args, **kwargs: (
                [(0, 0.9), (1, 0.8)]
                if kwargs.get("image_ranking")
                else [(1, 0.9), (0, 0.8)]
            ),
            compute_siglip_ranking=lambda *_args, **_kwargs: (
                [0, 1],
                {0: "https://example.org/img"},
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "qs.core.web_navigation.image_discovery",
        SimpleNamespace(
            IMAGE_CANDIDATE_LIMIT=15,
            discover_result_images=lambda *_args, **_kwargs: [],
        ),
    )

    text_result = search.search_web("topic", max_results=1)
    image_result = search.search_web_multimodal("topic", local_image=object())
    assert text_result["results"][0]["title"] == "B"
    assert image_result["visual_ranking_applied"] is True
    assert image_result["results"][0]["image_url"] == "https://example.org/img"


def test_rejects_private_url_and_invalid_port() -> None:
    with pytest.raises(WebNavigationError, match="not publicly routable"):
        security.validate_public_url("http://127.0.0.1/path")
    with pytest.raises(WebNavigationError, match="invalid port"):
        security.validate_public_url("https://example.org:invalid/")


def test_navigation_policy_preserves_the_first_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    policy = security.NavigationPolicy(initial_url="https://source.example/start")

    assert policy.observe_main_frame_navigation("https://other.example/page") is False
    assert policy.observe_main_frame_navigation("https://source.example/start") is True
    policy.record_violation(WebNavigationError("Later failure", kind="later"))

    assert policy.violation is not None
    assert policy.violation.kind == "cross_origin_navigation"
    with pytest.raises(WebNavigationError, match="left the initial origin"):
        policy.raise_if_violated()


def test_route_guard_rejects_cross_origin_http_redirect_before_fulfillment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    frame = object()
    request = SimpleNamespace(
        url="https://source.example/start",
        frame=frame,
        is_navigation_request=lambda: True,
    )

    class Route:
        def __init__(self) -> None:
            self.request = request
            self.aborted = False
            self.fulfilled = False

        async def fetch(self, *, max_redirects: int) -> SimpleNamespace:
            assert max_redirects == 0
            return SimpleNamespace(
                status=302, headers={"location": "https://other.example/page"}
            )

        async def abort(self) -> None:
            self.aborted = True

        async def fulfill(self, **_kwargs: object) -> None:
            self.fulfilled = True

    route = Route()
    policy = security.NavigationPolicy(initial_url=request.url)
    guard = security.build_route_guard(
        page=SimpleNamespace(main_frame=frame), policy=policy
    )
    asyncio.run(guard(route))

    assert route.aborted is True
    assert route.fulfilled is False
    assert policy.violation is not None
    assert policy.violation.kind == "cross_origin_navigation"


def test_route_guard_checks_later_redirects_before_fulfillment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(security, "validate_public_url", lambda url: url)
    frame = object()
    request = SimpleNamespace(
        url="https://source.example/start",
        method="GET",
        frame=frame,
        is_navigation_request=lambda: True,
    )

    class Route:
        def __init__(self) -> None:
            self.request = request
            self.fetched: list[str] = []
            self.aborted = False
            self.fulfilled = False

        async def fetch(
            self, *, url: str | None = None, max_redirects: int
        ) -> SimpleNamespace:
            assert max_redirects == 0
            self.fetched.append(url or request.url)
            location = "/middle" if url is None else "https://other.example/private"
            return SimpleNamespace(status=302, headers={"location": location})

        async def abort(self) -> None:
            self.aborted = True

        async def fulfill(self, **_kwargs: object) -> None:
            self.fulfilled = True

    route = Route()
    policy = security.NavigationPolicy(initial_url=request.url)
    guard = security.build_route_guard(
        page=SimpleNamespace(main_frame=frame), policy=policy
    )
    asyncio.run(guard(route))

    assert route.fetched == [request.url, "https://source.example/middle"]
    assert route.aborted is True
    assert route.fulfilled is False
    assert policy.violation is not None
    assert policy.violation.kind == "cross_origin_navigation"


def test_actions_require_bounded_typed_sequence() -> None:
    assert validate_actions([{"kind": "wait", "milliseconds": 1}]) == [
        {"kind": "wait", "milliseconds": 1}
    ]
    with pytest.raises(WebNavigationError, match="exactly one"):
        validate_actions([{"kind": "wait", "selector": "#ready", "milliseconds": 1}])
    with pytest.raises(WebNavigationError, match="between 1 and 10"):
        validate_actions([])


def test_web_tool_entrypoints_forward_and_report_missing_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = {
        tool.name: tool
        for tool in Capabilities([BUILTIN_CAPABILITIES_PATH])
        .load(work_root=tmp_path)
        .tools
    }
    monkeypatch.setitem(
        sys.modules,
        "qs.core.web_navigation.crawler",
        SimpleNamespace(
            crawl_page=lambda **kwargs: kwargs,
        ),
    )
    monkeypatch.setattr(
        search,
        "search_web",
        lambda *args, **kwargs: {
            "query": args[0],
            "max_results": kwargs["max_results"],
        },
    )
    assert tools["search-web"].execute(query="topic")["query"] == "topic"
    assert (
        tools["read-web-page"].execute(url="https://example.org")["workroot"]
        == tmp_path
    )
    interaction = tools["interact-web-page"].execute(
        url="https://example.org", actions=[{"kind": "wait", "milliseconds": 1}]
    )
    assert interaction["actions"] == [{"kind": "wait", "milliseconds": 1}]

    monkeypatch.setitem(sys.modules, "qs.core.web_navigation.crawler", None)
    unavailable = tools["read-web-page"].execute(url="https://example.org")
    assert unavailable["error_kind"] == "missing_dependency"


def test_crawl_sets_and_restores_its_work_root_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("lxml")
    from qs.core.web_navigation import crawler  # noqa: PLC0415

    monkeypatch.setattr(crawler, "validate_public_url", lambda url: url)
    observed: list[str | None] = []

    async def fake_crawl(**_kwargs: object) -> dict[str, object]:
        observed.append(os.environ.get("CRAWL4_AI_BASE_DIRECTORY"))
        return {"ok": True}

    monkeypatch.setattr(crawler, "_crawl_page", fake_crawl)
    monkeypatch.setenv("CRAWL4_AI_BASE_DIRECTORY", "original")
    result = crawler.crawl_page(
        url="https://example.org",
        output_mode="ast",
        timeout_seconds=2,
        workroot=tmp_path,
        actions=[{"kind": "wait", "milliseconds": 1}],
    )
    assert result == {"ok": True}
    assert observed == [str(tmp_path / ".qs" / "web-cache")]
    assert os.environ["CRAWL4_AI_BASE_DIRECTORY"] == "original"


def test_crawl_provisions_missing_browser_and_retries_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qs.core.web_navigation import crawler  # noqa: PLC0415

    monkeypatch.setattr(crawler, "validate_public_url", lambda url: url)
    attempts = 0
    installs = 0

    async def fake_crawl(**_kwargs: object) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise WebNavigationError("Missing Chromium.", kind="browser_unavailable")
        return {"ok": True}

    def fake_install() -> None:
        nonlocal installs
        installs += 1

    monkeypatch.setattr(crawler, "_crawl_page", fake_crawl)
    monkeypatch.setattr(crawler, "_install_headless_chromium", fake_install)
    result = crawler.crawl_page(
        url="https://example.org",
        output_mode="markdown",
        timeout_seconds=2,
        workroot=tmp_path,
        actions=[{"kind": "wait", "milliseconds": 1}],
    )
    assert result == {"ok": True}
    assert attempts == 2
    assert installs == 1


def test_crawl_does_not_install_browser_for_other_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qs.core.web_navigation import crawler  # noqa: PLC0415

    monkeypatch.setattr(crawler, "validate_public_url", lambda url: url)

    async def fake_crawl(**_kwargs: object) -> dict[str, object]:
        raise WebNavigationError("Page failed.", kind="crawl_failed")

    def unexpected_install() -> None:
        pytest.fail("Browser installation must be limited to missing executables")

    monkeypatch.setattr(crawler, "_crawl_page", fake_crawl)
    monkeypatch.setattr(crawler, "_install_headless_chromium", unexpected_install)
    with pytest.raises(WebNavigationError, match="Page failed"):
        crawler.crawl_page(
            url="https://example.org",
            output_mode="markdown",
            timeout_seconds=2,
            workroot=tmp_path,
            actions=[{"kind": "wait", "milliseconds": 1}],
        )


def test_browser_install_failure_reports_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qs.core.web_navigation import crawler  # noqa: PLC0415

    def failed_install(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 1, "", "Download unavailable")

    monkeypatch.setattr(crawler.subprocess, "run", failed_install)
    with pytest.raises(WebNavigationError, match="Download unavailable") as failure:
        crawler._install_headless_chromium()
    assert failure.value.kind == "browser_unavailable"
    assert "python -m playwright install chromium --only-shell" in str(failure.value)


def test_page_cache_discards_a_previous_cross_origin_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qs.core.web_navigation import crawler  # noqa: PLC0415

    monkeypatch.setattr(crawler, "validate_public_url", lambda url: url)
    original_url = "https://source.example/start"
    path = crawler._cache_path(cache_root=tmp_path, url=original_url, mode="markdown")
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"final_url": "https://other.example/page", "markdown": "old"}),
        encoding="utf-8",
    )
    assert (
        crawler._read_page_cache(cache_root=tmp_path, url=original_url, mode="markdown")
        is None
    )


def test_local_image_must_stay_inside_work_root(tmp_path: Path) -> None:
    pytest.importorskip("aiohttp")
    pytest.importorskip("lxml")
    from PIL import Image  # noqa: PLC0415

    from qs.core.web_navigation.image_discovery import (  # noqa: PLC0415
        load_local_query_image,
    )

    root = tmp_path / "work"
    root.mkdir()
    inside = root / "query.png"
    Image.new("RGB", (2, 2)).save(inside)
    assert load_local_query_image(image_path="query.png", workroot=root).size == (2, 2)
    with pytest.raises(WebNavigationError, match="inside workroot"):
        load_local_query_image(image_path=str(inside), workroot=tmp_path / "other")
