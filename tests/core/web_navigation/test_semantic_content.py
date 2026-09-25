"""Rendered-page extraction with varied, realistic document structures."""

import pytest
from lxml import html as lxml_html

from qs.core.web_navigation import ast, content


def _nodes(document: dict[str, object]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []

    def visit(items: list[dict[str, object]]) -> None:
        for item in items:
            result.append(item)
            visit(item.get("children", []))  # type: ignore[arg-type]

    visit(document["children"])  # type: ignore[arg-type]
    return result


def test_semantic_ast_keeps_controls_and_links_without_form_secrets() -> None:
    html = """<html><body>Intro
      <nav><a id="toc:1" href="/chapter" aria-label="Go">Read</a></nav>
      <main><h1>Document title</h1><p>Visible <em>text</em></p>
        <form><input name="token" value="secret" placeholder="Search here">
          <button data-testid="submit" role="button">Search</button></form>
        <img src="/cover.png" alt="Cover"><iframe src="/preview"></iframe>
        <ul><li>One</li></ul><pre>print(1)</pre>
        <script>hidden()</script><style>.hidden {}</style>
      </main>Outro</body></html>"""
    document = ast.build_semantic_ast(html, base_url="https://example.org/book")
    nodes = _nodes(document)

    assert document["stats"]["truncated"] is False  # type: ignore[index]
    assert any(
        node.get("kind") == "text" and node.get("text") == "Intro" for node in nodes
    )
    assert any(
        node.get("kind") == "text" and node.get("text") == "Outro" for node in nodes
    )
    link = next(node for node in nodes if node.get("kind") == "link")
    assert link["attributes"]["href"] == "https://example.org/chapter"  # type: ignore[index]
    assert link["selector"] == "#toc\\:1"
    image = next(node for node in nodes if node.get("kind") == "image")
    assert image["attributes"]["src"] == "https://example.org/cover.png"  # type: ignore[index]
    input_node = next(node for node in nodes if node.get("tag") == "input")
    assert input_node["text"] == "Search here"
    assert "secret" not in str(document)
    assert all(node.get("tag") not in {"script", "style"} for node in nodes)


@pytest.mark.parametrize(
    ("attribute", "value", "expected"),
    [
        ("data-testid", "a", 'button[data-testid="a"]'),
        ("data-test", "b", 'button[data-test="b"]'),
        ("data-qa", "c", 'button[data-qa="c"]'),
        ("name", "d", 'button[name="d"]'),
        ("aria-label", 'Say "hello"', 'button[aria-label="Say \\"hello\\""]'),
    ],
)
def test_semantic_control_selectors_are_stable(
    attribute: str, value: str, expected: str
) -> None:
    document = ast.build_semantic_ast(
        f'<html><body><button {attribute}="{value.replace(chr(34), "&quot;")}">Open</button></body></html>',
        base_url="https://example.org",
    )
    button = next(node for node in _nodes(document) if node.get("tag") == "button")
    assert button["selector"] == expected
    assert button["selector_stability"] == "stable"


def test_semantic_ast_recognizes_stat_cards_and_positional_selectors() -> None:
    document = ast.build_semantic_ast(
        "<html><body><div class='metric'><span>Stars</span><b>125</b></div>"
        "<section><button>First</button><button>Second</button></section></body></html>",
        base_url="https://example.org",
    )
    nodes = _nodes(document)
    stat = next(node for node in nodes if node.get("kind") == "stat")
    assert (stat["label"], stat["value"]) == ("Stars", "125")
    buttons = [node for node in nodes if node.get("tag") == "button"]
    assert buttons[1]["selector"].endswith("button:nth-of-type(2)")  # type: ignore[union-attr]
    assert buttons[1]["selector_stability"] == "positional"


def test_semantic_ast_enforces_node_and_text_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ast, "MAX_AST_NODES", 2)
    limited_nodes = ast.build_semantic_ast(
        "<html><body><p>One</p><p>Two</p><p>Three</p></body></html>",
        base_url="https://example.org",
    )
    assert limited_nodes["stats"]["emitted_nodes"] == 2  # type: ignore[index]
    assert limited_nodes["stats"]["truncated"] is True  # type: ignore[index]

    monkeypatch.setattr(ast, "MAX_AST_NODES", 400)
    monkeypatch.setattr(ast, "MAX_AST_TEXT_CHARS", 3)
    limited_text = ast.build_semantic_ast(
        "<html><body><p>Lengthy text</p><p>More text</p></body></html>",
        base_url="https://example.org",
    )
    assert limited_text["stats"]["text_chars"] == 3  # type: ignore[index]
    assert limited_text["stats"]["truncated"] is True  # type: ignore[index]


def test_semantic_ast_reports_invalid_html_input() -> None:
    with pytest.raises(ValueError, match="could not be parsed"):
        ast.build_semantic_ast(42, base_url="https://example.org")  # type: ignore[arg-type]


def test_semantic_ast_preserves_wrapped_tail_text_and_skips_comments() -> None:
    document = ast.build_semantic_ast(
        "<html><body><div>Before <span>inside</span> after</div><!-- comment -->"
        "<section><div><span></span><span>Only value</span></div></section></body></html>",
        base_url="https://example.org",
    )
    nodes = _nodes(document)
    assert any(
        node.get("kind") == "text" and node.get("text") == "after" for node in nodes
    )
    assert all("comment" not in str(node) for node in nodes)


def test_semantic_text_nodes_stop_at_both_independent_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ast, "MAX_AST_NODES", 1)
    budget = ast._Budget(emitted_nodes=1)
    assert ast._build_text_node("next", budget=budget) is None
    assert budget.truncated

    monkeypatch.setattr(ast, "MAX_AST_NODES", 400)
    monkeypatch.setattr(ast, "MAX_AST_TEXT_CHARS", 2)
    budget = ast._Budget(text_chars=2)
    assert ast._build_text_node("next", budget=budget) is None
    assert budget.truncated

    budget = ast._Budget()
    assert ast._build_text_node("long text", budget=budget)["text"] == "lo"  # type: ignore[index]
    assert budget.truncated


def test_primary_content_prefers_substantial_article_inside_main() -> None:
    article = "A detailed article. " * 30
    html = (
        "<html><body><nav>Navigation</nav><main>"
        f"<article><h1>Title</h1><p>{article}</p></article>"
        "<aside>Advertisement</aside></main></body></html>"
    )
    selected, scope = content.select_primary_html(html)
    assert scope == "article"
    assert "A detailed article." in selected
    assert "Advertisement" not in selected


def test_primary_content_uses_main_and_removes_small_recommendations() -> None:
    html = (
        "<html><body><main><p>" + "Long main body. " * 30 + "</p>"
        "<article class='recommended-content'><p>Other story.</p></article>"
        "<footer>Footer</footer></main></body></html>"
    )
    selected, scope = content.select_primary_html(html)
    assert scope == "main"
    assert "Long main body." in selected
    assert "Other story." not in selected
    assert "Footer" not in selected


def test_primary_content_falls_back_to_article_or_document() -> None:
    article, article_scope = content.select_primary_html(
        "<html><body><article>" + "Story. " * 40 + "</article></body></html>"
    )
    assert article_scope == "article"
    assert "Story." in article
    document, scope = content.select_primary_html(
        "<html><body><p>Brief.</p></body></html>"
    )
    assert scope == "document"
    assert "Brief." in document
    invalid, invalid_scope = content.select_primary_html(42)  # type: ignore[arg-type]
    assert invalid_scope == "document"
    assert invalid == "42"


def test_page_dates_preserve_source_and_precedence() -> None:
    html = """<html><head>
      <meta property="article:published_time" content="2026-01-02">
      <meta name="dateModified" content="2026-02-03">
      <script type="application/ld+json">{"@graph":[{"datePublished":"2025-01-01"}]}</script>
      </head><body><time datetime="2026-04-05">Today</time></body></html>"""
    dates = content.extract_page_dates({"article:published_time": "2026-03-04"}, html)
    assert dates["published"] == {
        "value": "2026-03-04",
        "source": "metadata:article:published_time",
    }
    assert dates["modified"] == {
        "value": "2026-02-03",
        "source": "meta:datemodified",
    }


def test_page_dates_read_nested_json_ld_and_ignore_malformed_values() -> None:
    html = """<html><head>
      <script type="application/ld+json">{invalid</script>
      <script type="application/ld+json">{"@graph":[{"author":{"dateModified":"2026-06-01"},"datePublished":"2026-05-01"}]}</script>
      </head></html>"""
    dates = content.extract_page_dates({"empty": "", "ignored": None}, html)
    assert dates["published"] == {
        "value": "2026-05-01",
        "source": "json-ld:1",
    }
    assert dates["modified"] == {
        "value": "2026-06-01",
        "source": "json-ld:1",
    }
    assert content.extract_page_dates(None, 42) == {  # type: ignore[arg-type]
        "published": None,
        "modified": None,
    }


def test_small_article_cards_are_removed_only_when_main_text_is_substantial() -> None:
    root = lxml_html.fromstring(
        "<main><p>"
        + "Direct story. " * 25
        + "</p><article>Related link</article></main>"
    )
    content._remove_small_recommendation_articles(root)
    assert not root.xpath(".//article")

    root = lxml_html.fromstring(
        "<main><p>Short intro</p><article>Related link</article></main>"
    )
    content._remove_small_recommendation_articles(root)
    assert root.xpath(".//article")

    root = lxml_html.fromstring(
        "<main><p>"
        + "Direct story. " * 25
        + "</p><article>"
        + "Long article. " * 100
        + "</article></main>"
    )
    content._remove_small_recommendation_articles(root)
    assert root.xpath(".//article")


def test_json_ld_date_walker_skips_nonscalar_dates_and_visits_nested_lists() -> None:
    dates: list[tuple[str, str, str]] = []
    content._collect_json_dates(
        [{"datePublished": ["ignored"], "nested": [{"dateModified": "2026-07-01"}]}, 4],
        dates=dates,
        source="test",
    )
    assert dates == [("datemodified", "2026-07-01", "test")]
    dates.clear()
    content._collect_json_dates(
        {"nested": {"datePublished": "2026-08-01"}},
        dates=dates,
        source="nested",
    )
    assert dates == [("datepublished", "2026-08-01", "nested")]
    content._collect_json_dates({"author": "Ada"}, dates=dates, source="ignored")
    assert dates == [("datepublished", "2026-08-01", "nested")]


def test_page_dates_ignore_empty_metadata_and_time_values() -> None:
    dates = content.extract_page_dates(
        None,
        '<html><head><meta name="datePublished" content=""></head>'
        '<body><time datetime="">Unknown</time></body></html>',
    )
    assert dates == {"published": None, "modified": None}
