"""Accurate line navigation for Python and Markdown."""

import ast
import re
from dataclasses import dataclass

from .lines import physical_lines
from .models import Region, RegionIndex


def python_navigation(text: str, index: RegionIndex) -> dict[str, object]:
    tree = ast.parse(text)

    def visit(node: ast.AST, parents: tuple[str, ...]) -> None:
        scope = parents
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = ".".join((*parents, node.name))
            start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            index.add(kind, name, start, node.end_lineno)
            scope = (*parents, node.name)
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, ())
    return {"language": "python", "syntax_valid": True}


@dataclass(frozen=True, slots=True)
class _Heading:
    level: int
    title: str
    start_line: int


@dataclass(frozen=True, slots=True)
class _Fence:
    marker: str
    length: int
    language: str
    start_line: int

    def closes(self, match: re.Match[str]) -> bool:
        return (
            match[1][0] == self.marker
            and len(match[1]) >= self.length
            and not match[2].strip()
        )

    def region(self, end_line: int) -> Region:
        return Region(
            "code_fence", self.language or "untyped", self.start_line, end_line
        )


def markdown_navigation(text: str, index: RegionIndex) -> None:
    lines = physical_lines(text)
    headings: list[_Heading] = []
    regions: list[Region] = []
    fence: _Fence | None = None
    previous_content = ""
    for number, line in enumerate(lines, 1):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            previous_content = ""
            if marker and fence.closes(marker):
                regions.append(fence.region(number))
                fence = None
            continue
        if marker:
            previous_content = ""
            fence = _Fence(marker[1][0], len(marker[1]), marker[2].strip(), number)
            continue
        heading = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+|$)(.*)$", line)
        if heading:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2]).strip()
            headings.append(_Heading(len(heading[1]), title, number))
            previous_content = ""
        elif previous_content and re.fullmatch(r" {0,3}(?:=+|-+)[ \t]*", line):
            headings.append(
                _Heading(1 if "=" in line else 2, lines[number - 2].strip(), number - 1)
            )
            previous_content = ""
        else:
            previous_content = line.strip()
    if fence is not None:
        regions.append(fence.region(len(lines)))

    regions.extend(_section_regions(headings, len(lines)))
    # Apply the budget after combining section and fence navigation.
    for region in sorted(regions, key=lambda item: item.start_line or 0):
        index.add(region.kind, region.name, region.start_line, region.end_line)


def _section_regions(headings: list[_Heading], total_lines: int) -> list[Region]:
    active: list[_Heading] = []
    regions: list[Region] = []
    for heading in headings:
        while active and active[-1].level >= heading.level:
            previous = active.pop()
            regions.append(
                Region(
                    "section",
                    previous.title,
                    previous.start_line,
                    heading.start_line - 1,
                )
            )
        active.append(heading)
    regions.extend(
        Region("section", heading.title, heading.start_line, total_lines)
        for heading in active
    )
    return regions
