"""Deterministic public-web search used to discover page URLs."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from .errors import WebNavigationError

if TYPE_CHECKING:
    from PIL import Image

MAX_SEARCH_RESULTS = 10
SEARCH_CANDIDATE_LIMIT = 50
SEARCH_BACKENDS = "duckduckgo,brave,mojeek,startpage,wikipedia,yahoo"
_TRACKING_PARAMETERS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
_LOGGER = logging.getLogger(__name__)


def search_web(
    query: str,
    *,
    max_results: int = 5,
    timeout_seconds: int = 30,
) -> dict[str, object]:
    """Search several non-Google providers and return bounded candidate pages."""
    from .ranking import rank_results  # noqa: PLC0415 - optional search extra

    normalized_query = _validate_search_arguments(
        query=query,
        max_results=max_results,
        timeout_seconds=timeout_seconds,
    )
    candidates = _retrieve_candidates(normalized_query, timeout_seconds=timeout_seconds)
    ranked = rank_results(normalized_query, candidates)
    return _build_search_payload(
        query=normalized_query,
        candidates=candidates,
        ranked=ranked,
        max_results=max_results,
    )


def search_web_multimodal(
    query: str,
    *,
    local_image: Image.Image,
    max_results: int = 5,
    timeout_seconds: int = 30,
) -> dict[str, object]:
    """Search with text discovery and an optional SigLIP2 visual reranking stage."""
    from .image_discovery import (  # noqa: PLC0415
        IMAGE_CANDIDATE_LIMIT,
        discover_result_images,
    )
    from .ranking import compute_siglip_ranking, rank_results  # noqa: PLC0415

    normalized_query = _validate_search_arguments(
        query=query,
        max_results=max_results,
        timeout_seconds=timeout_seconds,
    )
    candidates = _retrieve_candidates(normalized_query, timeout_seconds=timeout_seconds)
    preliminary = rank_results(normalized_query, candidates)
    selected_indices = [index for index, _ in preliminary[:IMAGE_CANDIDATE_LIMIT]]
    discovered_images = discover_result_images(
        candidates,
        selected_indices,
        timeout_seconds=timeout_seconds,
    )
    image_urls: dict[int, str] = {}
    visual_ranking_applied = False
    try:
        image_ranking, image_urls = compute_siglip_ranking(
            normalized_query,
            candidates,
            local_image=local_image,
            discovered_images=discovered_images,
        )
    except Exception as error:  # noqa: BLE001 - visual ranking is an optional signal.
        _LOGGER.warning("SigLIP2 visual ranking unavailable: %s", error)
        ranked = preliminary
    else:
        visual_ranking_applied = True
        ranked = rank_results(normalized_query, candidates, image_ranking=image_ranking)
    return _build_search_payload(
        query=normalized_query,
        candidates=candidates,
        ranked=ranked,
        max_results=max_results,
        image_urls=image_urls,
        visual_ranking_applied=visual_ranking_applied,
    )


def _validate_search_arguments(
    *, query: str, max_results: int, timeout_seconds: int
) -> str:
    """Validate public search arguments and return the normalized query."""
    normalized_query = query.strip() if isinstance(query, str) else ""
    if not normalized_query:
        raise WebNavigationError(
            "Search query must be a non-empty string.", kind="invalid_query"
        )
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= MAX_SEARCH_RESULTS
    ):
        raise WebNavigationError(
            f"max_results must be between 1 and {MAX_SEARCH_RESULTS}.",
            kind="invalid_max_results",
        )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or not 1 <= timeout_seconds <= 120
    ):
        raise WebNavigationError(
            "timeout_seconds must be between 1 and 120.", kind="invalid_timeout"
        )
    return normalized_query


def _retrieve_candidates(
    normalized_query: str, *, timeout_seconds: int
) -> list[dict[str, object]]:
    """Retrieve, normalize, and deduplicate the complete DDGS candidate pool."""
    from ddgs import DDGS  # noqa: PLC0415 - optional search extra
    from ddgs.exceptions import DDGSException  # noqa: PLC0415

    try:
        raw_results = DDGS(timeout=timeout_seconds).text(
            normalized_query,
            max_results=SEARCH_CANDIDATE_LIMIT,
            backend=SEARCH_BACKENDS,
        )
    except (DDGSException, OSError, TimeoutError) as error:
        raise WebNavigationError(
            "Public web search failed across the available providers.",
            kind="search_failed",
        ) from error

    normalized_results = [
        normalized
        for item in raw_results or []
        if (normalized := _normalize_result(item)) is not None
    ]
    return _deduplicate_results(
        normalized_results,
        limit=SEARCH_CANDIDATE_LIMIT,
    )


def _build_search_payload(
    *,
    query: str,
    candidates: list[dict[str, object]],
    ranked: list[tuple[int, float]],
    max_results: int,
    image_urls: dict[int, str] | None = None,
    visual_ranking_applied: bool | None = None,
) -> dict[str, object]:
    """Build the stable public response from one final candidate ranking."""
    results: list[dict[str, object]] = []
    for index, score in ranked[:max_results]:
        candidate = dict(candidates[index])
        candidate["relevance_score"] = score
        if image_urls and index in image_urls:
            candidate["image_url"] = image_urls[index]
        results.append(candidate)
    payload: dict[str, object] = {
        "operation": "search",
        "query": query,
        "results": results,
        "result_count": len(results),
        "max_results": max_results,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "discovery_only": True,
        "untrusted_content": True,
    }
    if visual_ranking_applied is not None:
        payload["visual_ranking_applied"] = visual_ranking_applied
        payload["visual_images_used"] = (
            len(image_urls or {}) if visual_ranking_applied else 0
        )
    return payload


def _normalize_result(item: object) -> dict[str, object] | None:
    """Normalize one provider result and omit unusable destination URLs."""
    if not isinstance(item, dict):
        return None
    url = _canonicalize_url(str(item.get("href", "")).strip())
    if url is None:
        return None
    return {
        "title": str(item.get("title", "")).strip(),
        "url": url,
        "snippet": str(item.get("body", "")).strip(),
    }


def _canonicalize_url(url: str) -> str | None:
    """Normalize one result URL and remove tracking-only distinctions."""
    try:
        parsed = urlparse(url)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    hostname = parsed.hostname.rstrip(".").lower()
    netloc_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = netloc_host if port is None else f"{netloc_host}:{port}"
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(
        [
            (key, value)
            for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if key.lower() not in _TRACKING_PARAMETERS
            and not key.lower().startswith("utm_")
        ],
        doseq=True,
    )
    return urlunparse((parsed.scheme.lower(), netloc, path, "", query, ""))


def _deduplicate_results(
    candidates: list[dict[str, object]], *, limit: int
) -> list[dict[str, object]]:
    """Return ranked candidates after canonical URL and title/host deduplication."""
    results: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    seen_titles: set[tuple[str, str]] = set()
    for candidate in candidates:
        url = str(candidate["url"])
        parsed = urlparse(url)
        path = re.sub(r"/videos$", "", parsed.path.rstrip("/")) or "/"
        url_key = urlunparse(parsed._replace(path=path, query="", fragment=""))
        title_key = re.sub(r"\W+", " ", str(candidate["title"]).lower()).strip()
        host_title = (str(parsed.hostname or "").removeprefix("www."), title_key)
        if url_key in seen_urls or (title_key and host_title in seen_titles):
            continue
        seen_urls.add(url_key)
        seen_titles.add(host_title)
        results.append(candidate)
        if len(results) >= limit:
            break
    return results
