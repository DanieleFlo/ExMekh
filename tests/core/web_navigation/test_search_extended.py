"""Search normalization, provider failures, and optional visual ranking."""

import sys
from types import SimpleNamespace

import pytest

from exmekh_lib.core.web_navigation import search
from exmekh_lib.core.web_navigation.errors import WebNavigationError


@pytest.mark.parametrize(
    ("query", "max_results", "timeout", "kind"),
    [
        (" ", 5, 30, "invalid_query"),
        ("topic", 0, 30, "invalid_max_results"),
        ("topic", True, 30, "invalid_max_results"),
        ("topic", 5, 0, "invalid_timeout"),
        ("topic", 5, True, "invalid_timeout"),
    ],
)
def test_search_validates_public_arguments(
    query: str, max_results: int, timeout: int, kind: str
) -> None:
    with pytest.raises(WebNavigationError) as failure:
        search._validate_search_arguments(
            query=query, max_results=max_results, timeout_seconds=timeout
        )
    assert failure.value.kind == kind


def test_search_normalizes_tracking_links_and_deduplicates_provider_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDDGS:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 4

        def text(
            self, query: str, *, max_results: int, backend: str
        ) -> list[dict[str, str]]:
            assert query == "research topic"
            assert max_results == search.SEARCH_CANDIDATE_LIMIT
            assert "wikipedia" in backend
            return [
                {
                    "title": "First",
                    "href": "https://WWW.EXAMPLE.org//paper/?utm_source=feed&id=1",
                    "body": "abstract",
                },
                {
                    "title": "First",
                    "href": "https://www.example.org/paper?id=1&fbclid=x",
                    "body": "duplicate",
                },
                {
                    "title": "Another",
                    "href": "https://example.org/videos",
                    "body": "video",
                },
                {
                    "title": "Another",
                    "href": "https://example.org/",
                    "body": "same canonical content",
                },
                {"title": "Bad", "href": "javascript:alert(1)", "body": "ignored"},
            ]

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=FakeDDGS))
    monkeypatch.setitem(
        sys.modules, "ddgs.exceptions", SimpleNamespace(DDGSException=RuntimeError)
    )
    results = search._retrieve_candidates("research topic", timeout_seconds=4)
    assert len(results) == 2
    assert results[0] == {
        "title": "First",
        "url": "https://www.example.org/paper?id=1",
        "snippet": "abstract",
    }


def test_provider_failure_becomes_stable_search_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenDDGS:
        def __init__(self, *, timeout: int) -> None:
            pass

        def text(self, *args: object, **kwargs: object) -> None:
            raise TimeoutError("provider timeout")

    monkeypatch.setitem(sys.modules, "ddgs", SimpleNamespace(DDGS=BrokenDDGS))
    monkeypatch.setitem(
        sys.modules, "ddgs.exceptions", SimpleNamespace(DDGSException=RuntimeError)
    )
    with pytest.raises(WebNavigationError) as failure:
        search._retrieve_candidates("topic", timeout_seconds=1)
    assert failure.value.kind == "search_failed"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org:bad/path",
        "https://user@example.org/",
        "ftp://example.org/",
        "/relative",
    ],
)
def test_search_discards_unsafe_or_unusable_links(url: str) -> None:
    assert search._canonicalize_url(url) is None
    assert search._normalize_result({"href": url}) is None


def test_search_ignores_nonobject_candidates_and_stops_at_limit() -> None:
    assert search._normalize_result("not a provider result") is None
    candidates = [
        {"title": "A", "url": "https://example.org/a"},
        {"title": "B", "url": "https://example.org/b"},
    ]
    assert search._deduplicate_results(candidates, limit=1) == candidates[:1]


def test_search_payload_includes_visual_candidate_provenance() -> None:
    candidates = [{"title": "A", "url": "https://example.org/a", "snippet": "a"}]
    payload = search._build_search_payload(
        query="topic",
        candidates=candidates,
        ranked=[(0, 0.75)],
        max_results=1,
        image_urls={0: "https://example.org/image.png"},
        visual_ranking_applied=True,
    )
    assert payload["results"][0]["image_url"] == "https://example.org/image.png"  # type: ignore[index]
    assert payload["visual_images_used"] == 1
    assert payload["untrusted_content"] is True


def test_multimodal_search_falls_back_when_visual_model_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        search,
        "_retrieve_candidates",
        lambda *args, **kwargs: [
            {"title": "A", "url": "https://example.org/a", "snippet": "a"}
        ],
    )

    def failed_visual(*args: object, **kwargs: object) -> None:
        raise RuntimeError("model unavailable")

    monkeypatch.setitem(
        sys.modules,
        "exmekh_lib.core.web_navigation.ranking",
        SimpleNamespace(
            rank_results=lambda *args, **kwargs: [(0, 0.5)],
            compute_siglip_ranking=failed_visual,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "exmekh_lib.core.web_navigation.image_discovery",
        SimpleNamespace(
            IMAGE_CANDIDATE_LIMIT=15,
            discover_result_images=lambda *args, **kwargs: [],
        ),
    )
    result = search.search_web_multimodal("topic", local_image=object())  # type: ignore[arg-type]
    assert result["result_count"] == 1
    assert result["visual_ranking_applied"] is False
    assert result["visual_images_used"] == 0
