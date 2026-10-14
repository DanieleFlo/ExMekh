"""Keep repository diagnostics and presentation prose in their domain configs."""

import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest

from exmekh_lib.core.cache import tool_call_record


def _is_docstring(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    expression = parents.get(node)
    container = parents.get(expression)
    return (
        isinstance(expression, ast.Expr)
        and isinstance(
            container, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and container.body[0] is expression
    )


def test_diagnostics_and_human_phrases_are_not_embedded_in_production_code():
    source = Path(__file__).resolve().parents[2] / "src" / "exmekh_lib"
    embedded: list[str] = []
    log_methods = {"debug", "info", "warning", "error", "exception", "critical"}
    for path in sorted(source.rglob("*.py")):
        if "config" in path.relative_to(source).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {
            child: node
            for node in ast.walk(tree)
            for child in ast.iter_child_nodes(node)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args:
                error = isinstance(node.func, ast.Name) and node.func.id.endswith(
                    ("Error", "Exception")
                )
                log = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in log_methods
                )
                if (error or log) and isinstance(
                    node.args[0], (ast.Constant, ast.JoinedStr)
                ):
                    embedded.append(
                        f"{path.relative_to(source)}:{node.lineno} diagnostic"
                    )
            if not isinstance(node, (ast.Constant, ast.JoinedStr)) or _is_docstring(
                node, parents
            ):
                continue
            if isinstance(parents.get(node), (ast.JoinedStr, ast.FormattedValue)):
                continue
            text = (
                node.value
                if isinstance(node, ast.Constant)
                else "".join(
                    part.value for part in node.values if isinstance(part, ast.Constant)
                )
            )
            # Protocol keys, selectors and identifiers are vocabulary, not prose.
            if isinstance(text, str) and re.search(
                r"\b[A-Za-z]{2,}(?: [A-Za-z]{2,})+", text
            ):
                embedded.append(f"{path.relative_to(source)}:{node.lineno} prose")
    assert not embedded, "Move presentation text into config:\n" + "\n".join(embedded)


def test_record_validation_uses_its_configured_diagnostic(monkeypatch):
    message = "configured empty record name"
    monkeypatch.setattr(
        tool_call_record,
        "CACHE_TEXT",
        replace(
            tool_call_record.CACHE_TEXT, toolcallrecord_name_cannot_be_empty=message
        ),
    )
    with pytest.raises(ValueError) as failure:
        tool_call_record.ToolCallRecord(" ", "id", {}, None)
    assert str(failure.value) == message
