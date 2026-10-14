"""Heuristics for primary page content and temporal provenance."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

from lxml import html as lxml_html

from exmekh_lib.config.web import WEB_CONTENT


def select_primary_html(html: str) -> tuple[str, str]:
    """Return cleaned primary HTML and the semantic scope used to select it."""
    try:
        root = lxml_html.fromstring(html or "<html></html>")
    except (TypeError, ValueError):
        return str(html or ""), "document"

    main_candidates = root.xpath("//main | //*[@role='main']")
    selected = _largest_candidate(
        main_candidates, minimum_chars=WEB_CONTENT.minimum_primary_chars
    )
    scope = "main"
    if selected is not None:
        nested_article = _largest_candidate(
            selected.xpath(".//article"),
            minimum_chars=WEB_CONTENT.minimum_primary_chars,
        )
        if nested_article is not None and _text_length(nested_article) >= max(
            WEB_CONTENT.minimum_primary_chars,
            int(_text_length(selected) * WEB_CONTENT.minimum_article_fraction),
        ):
            selected = nested_article
            scope = "article"
    if selected is None:
        selected = _largest_candidate(
            root.xpath("//article"), minimum_chars=WEB_CONTENT.minimum_primary_chars
        )
        scope = "article"
    if selected is None:
        body = root.find("body")
        selected = body if body is not None else root
        scope = "document"

    _remove_boilerplate(selected)
    if scope == "main":
        _remove_small_recommendation_articles(selected)
    return cast(str, lxml_html.tostring(selected, encoding="unicode")), scope


def extract_page_dates(
    metadata: Mapping[str, Any] | None, html: str
) -> dict[str, dict[str, str] | None]:
    """Extract published and modified dates with their observable source."""
    candidates: list[tuple[str, str, str]] = []
    for key, value in (metadata or {}).items():
        if isinstance(value, (str, int, float)) and str(value).strip():
            candidates.append((str(key).lower(), str(value).strip(), f"metadata:{key}"))

    try:
        root = lxml_html.fromstring(html or "<html></html>")
    except (TypeError, ValueError):
        root = None
    if root is not None:
        for element in root.xpath("//meta[@content]"):
            key = str(
                element.get("property")
                or element.get("name")
                or element.get("itemprop")
                or ""
            ).lower()
            value = str(element.get("content") or "").strip()
            if key and value:
                candidates.append((key, value, f"meta:{key}"))
        for element in root.xpath("//time[@datetime]"):
            value = str(element.get("datetime") or "").strip()
            if value:
                candidates.append(("datepublished", value, "time:datetime"))
        candidates.extend(_json_ld_dates(root))

    return {
        "published": _first_date(candidates, WEB_CONTENT.published_keys),
        "modified": _first_date(candidates, WEB_CONTENT.modified_keys),
    }


def _largest_candidate(elements: list[Any], *, minimum_chars: int) -> Any | None:
    """Return the candidate with the most normalized visible text."""
    scored = [(_text_length(element), element) for element in elements]
    if not scored:
        return None
    length, selected = max(scored, key=lambda item: item[0])
    return selected if length >= minimum_chars else None


def _text_length(element: Any) -> int:
    """Return the normalized text length of one HTML element."""
    return len(" ".join(str(element.text_content()).split()))


def _remove_boilerplate(root: Any) -> None:
    """Remove navigation and common advertising/recommendation containers."""
    for element in list(root.xpath(".//nav | .//footer | .//aside")):
        element.getparent().remove(element)
    for element in list(root.iterdescendants()):
        identity = " ".join(
            str(element.get(name) or "") for name in ("id", "class")
        ).lower()
        if any(token in identity for token in WEB_CONTENT.boilerplate_tokens):
            element.getparent().remove(element)


def _remove_small_recommendation_articles(root: Any) -> None:
    """Remove small article cards when substantial direct main content remains."""
    articles = list(root.xpath(".//article"))
    if not articles:
        return
    total_length = _text_length(root)
    nested_length = sum(_text_length(article) for article in articles)
    if total_length - nested_length < WEB_CONTENT.minimum_primary_chars:
        return
    for article in articles:
        if _text_length(article) < WEB_CONTENT.recommendation_article_chars:
            article.getparent().remove(article)


def _json_ld_dates(root: Any) -> list[tuple[str, str, str]]:
    """Return temporal values found in bounded JSON-LD blocks."""
    dates: list[tuple[str, str, str]] = []
    for index, element in enumerate(
        root.xpath("//script[@type='application/ld+json']")[
            : WEB_CONTENT.max_json_ld_blocks
        ]
    ):
        try:
            payload = json.loads(element.text or "")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        _collect_json_dates(payload, dates=dates, source=f"json-ld:{index}")
    return dates


def _collect_json_dates(
    value: object, *, dates: list[tuple[str, str, str]], source: str
) -> None:
    """Collect temporal scalar fields recursively from one JSON-LD value."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).lower()
            if normalized_key in {"datepublished", "datemodified"} and isinstance(
                item, (str, int, float)
            ):
                dates.append((normalized_key, str(item).strip(), source))
            elif isinstance(item, (Mapping, list)):
                _collect_json_dates(item, dates=dates, source=source)
    elif isinstance(value, list):
        for item in value:
            _collect_json_dates(item, dates=dates, source=source)


def _first_date(
    candidates: list[tuple[str, str, str]], keys: tuple[str, ...]
) -> dict[str, str] | None:
    """Return the first temporal value matching the ordered key aliases."""
    for expected in keys:
        for key, value, source in candidates:
            if key == expected:
                return {"value": value, "source": source}
    return None
