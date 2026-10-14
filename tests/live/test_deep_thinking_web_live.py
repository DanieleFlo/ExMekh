"""Real sites exercising discovery, extraction, browser actions, and URL policy."""

import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote, urlparse

import pytest

from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.capabilities.defaults import BUILTIN_CAPABILITIES_PATH
from exmekh_lib.core.web_navigation import search as web_search

pytestmark = pytest.mark.live

STATISTICS_URL = "https://docs.python.org/3/library/statistics.html"
MATH_URL = "https://docs.python.org/3/library/math.html"
FORM_URL = "https://httpbin.org/forms/post"


def _web_tools(work_root: Path):
    loaded = Capabilities([BUILTIN_CAPABILITIES_PATH]).load(work_root=work_root)
    return {tool.name: tool for tool in loaded.tools}


def _search_with_retry(tool, **arguments: object) -> dict[str, object]:
    for delay in (0, 5, 10, 20):
        if delay:
            time.sleep(delay)
        result = tool.execute(**arguments)
        if result.get("error_kind") != "search_failed":
            return result
    return result


def test_live_search_reads_a_discovered_source_instead_of_only_snippets(
    tmp_path: Path,
) -> None:
    tools = _web_tools(tmp_path)
    query = "notizie principali di oggi"
    search = _search_with_retry(tools["search-web"], query=query, max_results=5)
    assert search.get("result_count", 0) > 0, search
    assert search["query"] == query
    urls = [item["url"] for item in search["results"]]
    assert search["result_count"] == len(urls) <= 5
    assert len(urls) == len(set(urls))
    assert all(urlparse(url).scheme in {"http", "https"} for url in urls)

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
        pytest.fail(f"No discovered source was readable: {attempts}")


def test_live_browser_bootstraps_from_an_empty_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.getenv("EXMEKH_LIB_LIVE_BROWSER_BOOTSTRAP") != "1":
        pytest.skip(
            "set EXMEKH_LIB_LIVE_BROWSER_BOOTSTRAP=1 to download a fresh browser"
        )

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


@pytest.mark.parametrize(
    ("status", "error_code"),
    [
        (401, "http_access_denied"),
        (403, "http_access_denied"),
        (404, "http_not_found"),
        (429, "http_rate_limited"),
        (503, "http_server_error"),
    ],
)
def test_live_http_failure_preserves_remote_status_and_recovery(
    tmp_path: Path, status: int, error_code: str
) -> None:
    url = f"https://httpbin.org/status/{status}"
    result = _web_tools(tmp_path)["read-web-page"].execute(url=url)

    assert result.get("ok") is False, result
    assert result["error_kind"] == "crawl_failed", result
    assert result["error_code"] == error_code, result
    assert result["details"]["status_code"] == status, result
    assert result["details"]["requested_url"] == url, result
    assert result["error"] and result["hint"], result


def test_live_browser_form_submission_captures_destination_page(tmp_path: Path) -> None:
    tools = _web_tools(tmp_path)

    inspected = tools["read-web-page"].execute(url=FORM_URL, output_mode="ast")
    assert inspected.get("status_code") == 200, inspected
    assert 'input[name="custname"]' in str(inspected["ast"])
    assert 'input[name="custemail"]' in str(inspected["ast"])

    result = tools["interact-web-page"].execute(
        url=FORM_URL,
        actions=[
            {
                "kind": "fill",
                "selector": 'input[name="custname"]',
                "value": "ExMekh Live",
            },
            {
                "kind": "fill",
                "selector": 'input[name="custemail"]',
                "value": "live@example.test",
            },
            {"kind": "click", "selector": "button"},
        ],
        output_mode="hybrid",
    )
    assert result.get("actions_completed") is True, result
    assert all(
        item["success"] and item["effect_observed"] for item in result["actions"]
    )
    assert result["final_url"] == "https://httpbin.org/post"
    assert "ExMekh Live" in result["markdown"]
    assert "live@example.test" in result["markdown"]
    assert "ExMekh Live" in str(result["ast"])


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
    if os.getenv("EXMEKH_LIB_REQUIRE_LIVE_VISUAL_RANKING") == "1":
        assert result["visual_ranking_applied"] is True
