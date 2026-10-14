"""Deterministic HTML-to-semantic-AST conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

from lxml import html as lxml_html

from exmekh_lib.config.web import WEB_NAVIGATION
from exmekh_lib.config.web_text import WEB_TEXT

MAX_AST_NODES = WEB_NAVIGATION.max_ast_nodes
MAX_AST_TEXT_CHARS = WEB_NAVIGATION.max_ast_text_chars
_SKIPPED_TAGS = {"script", "style", "noscript", "template", "svg", "canvas"}
_KINDS = {
    "header": "region",
    "main": "region",
    "article": "section",
    "section": "section",
    "nav": "navigation",
    "aside": "aside",
    "footer": "footer",
    "h1": "heading",
    "h2": "heading",
    "h3": "heading",
    "h4": "heading",
    "h5": "heading",
    "h6": "heading",
    "p": "paragraph",
    "ul": "list",
    "ol": "list",
    "li": "list_item",
    "a": "link",
    "img": "image",
    "pre": "code",
    "code": "code",
    "table": "table",
    "blockquote": "quote",
    "form": "form",
    "button": "control",
    "input": "control",
    "select": "control",
    "textarea": "control",
    "details": "control",
    "summary": "control",
    "iframe": "embed",
}


@dataclass(slots=True)
class _Budget:
    """Track AST emission budgets and truncation state."""

    emitted_nodes: int = 0
    original_nodes: int = 0
    text_chars: int = 0
    truncated: bool = False


def build_semantic_ast(html: str, *, base_url: str) -> dict[str, Any]:
    """Build a bounded hierarchical semantic AST from rendered HTML."""
    try:
        root = lxml_html.fromstring(html or "<html></html>")
    except (TypeError, ValueError) as error:
        raise ValueError(WEB_TEXT.rendered_page_html_could_not_be_parsed) from error
    budget = _Budget()
    children = _build_children(root, base_url=base_url, budget=budget)
    return {
        "id": "document",
        "kind": "document",
        "tag": "html",
        "children": children,
        "stats": {
            "original_nodes": budget.original_nodes,
            "emitted_nodes": budget.emitted_nodes,
            "text_chars": budget.text_chars,
            "truncated": budget.truncated,
            "max_nodes": MAX_AST_NODES,
            "max_text_chars": MAX_AST_TEXT_CHARS,
        },
    }


def _build_children(
    element: Any, *, base_url: str, budget: _Budget
) -> list[dict[str, Any]]:
    """Recursively emit semantic descendants while preserving useful hierarchy."""
    nodes: list[dict[str, Any]] = []
    preserve_direct_text = _KINDS.get(_tag_name(element)) is None
    if preserve_direct_text:
        text_node = _build_text_node(element.text, budget=budget)
        if text_node is not None:
            nodes.append(text_node)
    for child in element:
        tag = _tag_name(child)
        if not tag:
            continue
        if tag in _SKIPPED_TAGS:
            continue
        stat = _stat_pair(child)
        kind = "stat" if stat is not None else _KINDS.get(tag)
        if kind is None:
            nodes.extend(_build_children(child, base_url=base_url, budget=budget))
            if preserve_direct_text:
                tail_node = _build_text_node(child.tail, budget=budget)
                if tail_node is not None:
                    nodes.append(tail_node)
            continue
        budget.original_nodes += 1
        if budget.emitted_nodes >= MAX_AST_NODES:
            budget.truncated = True
            continue
        budget.emitted_nodes += 1
        node: dict[str, Any] = {
            "id": f"node-{budget.emitted_nodes}",
            "kind": kind,
            "tag": tag,
        }
        if stat is not None:
            node["label"], node["value"] = stat
        role = child.get("role")
        if role:
            node["role"] = role
        attributes = _safe_attributes(child, tag=tag, base_url=base_url)
        if attributes:
            node["attributes"] = attributes
        selector, stability = _selector(child)
        if kind in {"link", "form", "control", "navigation"}:
            node["selector"] = selector
            node["selector_stability"] = stability
        text = _node_text(child, kind=kind)
        if text and budget.text_chars < MAX_AST_TEXT_CHARS:
            remaining = MAX_AST_TEXT_CHARS - budget.text_chars
            clipped = text[:remaining]
            node["text"] = clipped
            budget.text_chars += len(clipped)
            if len(clipped) < len(text):
                budget.truncated = True
        elif text:
            budget.truncated = True
        descendants = (
            []
            if stat is not None
            else _build_children(child, base_url=base_url, budget=budget)
        )
        if descendants:
            node["children"] = descendants
        nodes.append(node)
        if preserve_direct_text:
            tail_node = _build_text_node(child.tail, budget=budget)
            if tail_node is not None:
                nodes.append(tail_node)
    return nodes


def _build_text_node(value: object, *, budget: _Budget) -> dict[str, Any] | None:
    """Build one bounded text node for content inside unclassified containers."""
    text = _normalize_text(str(value or ""))
    if not text:
        return None
    budget.original_nodes += 1
    if budget.emitted_nodes >= MAX_AST_NODES:
        budget.truncated = True
        return None
    remaining = MAX_AST_TEXT_CHARS - budget.text_chars
    if remaining <= 0:
        budget.truncated = True
        return None
    clipped = text[:remaining]
    budget.emitted_nodes += 1
    budget.text_chars += len(clipped)
    if len(clipped) < len(text):
        budget.truncated = True
    return {
        "id": f"node-{budget.emitted_nodes}",
        "kind": "text",
        "tag": "#text",
        "text": clipped,
    }


def _stat_pair(element: Any) -> tuple[str, str] | None:
    """Recognize one compact label/value card without page-specific selectors."""
    if _tag_name(element) not in {"div", "li", "section"}:
        return None
    children = [
        child
        for child in element
        if _tag_name(child) and _tag_name(child) not in _SKIPPED_TAGS
    ]
    if not 2 <= len(children) <= 4:
        return None
    texts = [_normalize_text(" ".join(child.itertext())) for child in children]
    texts = [text for text in texts if text]
    if len(texts) < 2:
        return None
    label, value = texts[0], texts[1]
    identity = " ".join(
        str(element.get(name) or "") for name in ("id", "class")
    ).lower()
    value_identity = " ".join(
        str(children[1].get(name) or "") for name in ("id", "class")
    ).lower()
    has_stat_identity = any(
        token in f"{identity} {value_identity}"
        for token in ("box", "count", "metric", "stat", "value", "bottom-part")
    )
    has_numeric_value = bool(re.search(r"\d", value))
    if (
        not has_stat_identity
        or not has_numeric_value
        or len(label) > WEB_NAVIGATION.max_stat_label_chars
        or len(value) > WEB_NAVIGATION.max_stat_value_chars
    ):
        return None
    return label, value


def _tag_name(element: Any) -> str:
    """Return a normalized element tag name."""
    tag = getattr(element, "tag", "")
    return str(tag).split("}")[-1].lower() if isinstance(tag, str) else ""


def _node_text(element: Any, *, kind: str) -> str:
    """Return normalized direct semantic text without leaking input values."""
    if kind in {
        "section",
        "region",
        "navigation",
        "aside",
        "footer",
        "form",
        "list",
        "table",
    }:
        return ""
    if _tag_name(element) == "input":
        return _normalize_text(
            element.get("aria-label") or element.get("placeholder") or ""
        )
    return _normalize_text(" ".join(element.itertext()))


def _normalize_text(value: str) -> str:
    """Collapse whitespace in extracted visible text."""
    return re.sub(r"\s+", " ", value).strip()


def _safe_attributes(element: Any, *, tag: str, base_url: str) -> dict[str, str]:
    """Return a small attribute allowlist that excludes form values and secrets."""
    allowed = (
        "id",
        "name",
        "type",
        "role",
        "aria-label",
        "placeholder",
        "alt",
        "title",
    )
    result = {name: element.get(name) for name in allowed if element.get(name)}
    if tag == "a" and element.get("href"):
        result["href"] = urljoin(base_url, element.get("href"))
    if tag in {"img", "iframe"} and element.get("src"):
        result["src"] = urljoin(base_url, element.get("src"))
    return result


def _selector(element: Any) -> tuple[str, str]:
    """Build a deterministic selector and label whether it is stable or positional."""
    tag = _tag_name(element)
    if element.get("id"):
        return f"#{_css_identifier(element.get('id'))}", "stable"
    for attribute in ("data-testid", "data-test", "data-qa", "name", "aria-label"):
        value = element.get(attribute)
        if value:
            return f'{tag}[{attribute}="{_css_string(value)}"]', "stable"
    parts: list[str] = []
    current = element
    while current is not None and _tag_name(current) not in {"", "html"}:
        current_tag = _tag_name(current)
        parent = current.getparent()
        siblings = (
            [item for item in parent if _tag_name(item) == current_tag]
            if parent is not None
            else []
        )
        position = siblings.index(current) + 1 if current in siblings else 1
        parts.append(f"{current_tag}:nth-of-type({position})")
        current = parent
    return " > ".join(reversed(parts)), "positional"


def _css_identifier(value: str) -> str:
    """Escape a conservative CSS identifier."""
    return re.sub(r"([^a-zA-Z0-9_-])", lambda match: "\\" + match.group(1), value)


def _css_string(value: str) -> str:
    """Escape a CSS quoted attribute value."""
    return value.replace("\\", "\\\\").replace('"', '\\"')
