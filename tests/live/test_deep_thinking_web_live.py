"""Real sites exercising discovery, extraction, browser actions, and URL policy."""

import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote, urlparse

import pytest

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.capabilities.defaults import BUILTIN_CAPABILITIES_PATH
from qs.core.web_navigation import search as web_search

pytestmark = pytest.mark.live

STATISTICS_URL = "https://docs.python.org/3/library/statistics.html"
MATH_URL = "https://docs.python.org/3/library/math.html"
LOGIN_URL = "https://the-internet.herokuapp.com/login"


def _web_tools(work_root: Path):
    loaded = Capabilities([BUILTIN_CAPABILITIES_PATH]).load(work_root=work_root)
    return {tool.name: tool for tool in loaded.tools}


def _search_with_one_retry(tool, **arguments: object) -> dict[str, object]:
    result = tool.execute(**arguments)
    if result.get("error_kind") == "search_failed":
        time.sleep(2)
        result = tool.execute(**arguments)
    return result


def test_live_search_discovers_official_sources_for_two_queries(tmp_path: Path) -> None:
    tools = _web_tools(tmp_path)

    for query, expected_domain in (
        ("IANA reserved example domains", "iana.org"),
        ("Python statistics mean standard library documentation", "docs.python.org"),
    ):
        result = _search_with_one_retry(tools["search-web"], query=query, max_results=5)
        assert "result_count" in result, result
        assert result["query"] == query
        assert result["result_count"] > 0
        urls = [item["url"] for item in result["results"]]
        assert len(urls) == len(set(urls))
        assert all(urlparse(url).scheme in {"http", "https"} for url in urls)
        assert any(
            (host := urlparse(url).hostname) == expected_domain
            or (host is not None and host.endswith("." + expected_domain))
            for url in urls
        ), urls


def test_live_news_search_reads_a_source_instead_of_only_snippets(
    tmp_path: Path,
) -> None:
    tools = _web_tools(tmp_path)
    search = _search_with_one_retry(
        tools["search-web"], query="notizie principali di oggi", max_results=5
    )
    assert search.get("result_count", 0) > 0, search

    attempts = []
    for candidate in search["results"]:
        page = tools["read-web-page"].execute(
            url=candidate["url"], output_mode="markdown"
        )
        attempts.append((candidate["url"], page.get("error_kind")))
        if (
            page.get("markdown_chars", 0) >= 400
            and urlparse(page["final_url"]).hostname
            == urlparse(candidate["url"]).hostname
        ):
            assert page["title"]
            assert page["retrieved_at"]
            assert page["markdown"].strip()
            break
    else:
        pytest.fail(f"No discovered news source was readable: {attempts}")


def test_live_browser_bootstraps_from_an_empty_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.getenv("QS_LIVE_BROWSER_BOOTSTRAP") != "1":
        pytest.skip("set QS_LIVE_BROWSER_BOOTSTRAP=1 to download a fresh browser")

    with TemporaryDirectory(dir=tmp_path) as browser_directory:
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", browser_directory)
        tools = _web_tools(tmp_path)
        page = tools["read-web-page"].execute(url=STATISTICS_URL)
        assert page.get("status_code") == 200, page
        assert "statistics" in page["markdown"].lower()
        assert list(Path(browser_directory).iterdir())


def test_live_read_complex_documentation_and_reuse_cache(tmp_path: Path) -> None:
    tools = _web_tools(tmp_path)

    first = tools["read-web-page"].execute(url=STATISTICS_URL, output_mode="hybrid")
    assert first["status_code"] == 200
    assert first["final_url"] == STATISTICS_URL
    assert "statistics" in first["markdown"].lower()
    assert "mean" in first["markdown"].lower()
    assert first["ast"]["stats"]["original_nodes"] > 100

    cached = tools["read-web-page"].execute(url=STATISTICS_URL, output_mode="hybrid")
    assert cached == first


def test_live_browser_form_submission_captures_destination_page(tmp_path: Path) -> None:
    tools = _web_tools(tmp_path)

    inspected = tools["read-web-page"].execute(url=LOGIN_URL, output_mode="ast")
    assert "#username" in str(inspected["ast"])
    assert "#password" in str(inspected["ast"])

    result = tools["interact-web-page"].execute(
        url=LOGIN_URL,
        actions=[
            {"kind": "fill", "selector": "#username", "value": "tomsmith"},
            {"kind": "fill", "selector": "#password", "value": "SuperSecretPassword!"},
            {"kind": "click", "selector": "button[type=submit]"},
        ],
        output_mode="hybrid",
    )
    assert result["actions_completed"] is True
    assert all(
        item["success"] and item["effect_observed"] for item in result["actions"]
    )
    assert result["final_url"] == "https://the-internet.herokuapp.com/secure"
    assert "You logged into a secure area!" in result["markdown"]
    assert "Secure Area" in str(result["ast"])


def test_live_same_origin_navigation_reaches_a_second_document(tmp_path: Path) -> None:
    tools = _web_tools(tmp_path)
    result = tools["interact-web-page"].execute(
        url=STATISTICS_URL,
        actions=[{"kind": "navigate", "url": MATH_URL}],
        output_mode="markdown",
    )
    assert result["actions_completed"] is True
    assert result["final_url"] == MATH_URL
    assert "math" in result["markdown"].lower()


def test_live_cross_origin_redirect_and_private_url_are_blocked(tmp_path: Path) -> None:
    tools = _web_tools(tmp_path)
    redirect = tools["read-web-page"].execute(
        url="https://httpbin.org/redirect-to?url=https%3A%2F%2Fexample.com%2F"
    )
    assert redirect["error_kind"] == "cross_origin_navigation"

    second_hop = "/redirect-to?url=" + quote("https://example.com/", safe="")
    chained = tools["read-web-page"].execute(
        url="https://httpbin.org/redirect-to?url=" + quote(second_hop, safe="")
    )
    assert chained["error_kind"] == "cross_origin_navigation"

    private = tools["read-web-page"].execute(url="http://127.0.0.1/admin")
    assert private["error_kind"] == "blocked_url"


def test_live_image_assisted_search_reports_whether_visual_ranking_ran(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image, ImageDraw  # noqa: PLC0415

    # Provider throttling is covered by the live search tests above. Keep this
    # test focused on the real image-loading and multimodal ranking path.
    candidates = [
        {
            "title": "Python statistics reference",
            "url": STATISTICS_URL,
            "snippet": "Statistics and numerical data",
        },
        {
            "title": "Python math reference",
            "url": MATH_URL,
            "snippet": "Mathematical functions",
        },
    ]
    monkeypatch.setattr(
        web_search, "_retrieve_candidates", lambda *args, **kwargs: candidates
    )

    image = Image.new("RGB", (256, 256), "skyblue")
    ImageDraw.Draw(image).ellipse((96, 24, 176, 104), fill="yellow")
    image.save(tmp_path / "sunny-sky.png")

    result = _web_tools(tmp_path)["search-web"].execute(
        query="sunny blue sky illustration",
        image_path="sunny-sky.png",
        max_results=3,
    )
    assert "result_count" in result, result
    assert result["result_count"] > 0
    assert {item["url"] for item in result["results"]} == {  # type: ignore[index]
        STATISTICS_URL,
        MATH_URL,
    }
    assert isinstance(result["visual_ranking_applied"], bool)
    assert isinstance(result["visual_images_used"], int)
    if os.getenv("QS_REQUIRE_LIVE_VISUAL_RANKING") == "1":
        assert result["visual_ranking_applied"] is True
