"""Render reserved XML conditions without reserializing the surrounding Markdown."""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from xml.etree import ElementTree as ET

from exmekh_lib.config.capability_prompts import CAPABILITY_PROMPTS
from exmekh_lib.core.agents.capability.markdown_literals import literal_spans

_TOKENS = re.compile(CAPABILITY_PROMPTS.tokens)


@dataclass(frozen=True, slots=True)
class CapabilityPromptContext:
    known_tools: frozenset[str]
    known_skills: frozenset[str]
    selected_tools: frozenset[str]
    selected_skills: frozenset[str]
    direct_tools: frozenset[str]
    direct_skills: frozenset[str]
    reachable_tools: frozenset[str]
    reachable_skills: frozenset[str]

    def validate_references(
        self, names: Iterable[str], additional: Iterable[str] = ()
    ) -> None:
        missing = (
            set(names) - self.selected_tools - self.selected_skills - set(additional)
        )
        if missing:
            raise ValueError(
                CAPABILITY_PROMPTS.unavailable_reference.format(
                    name=", ".join(sorted(missing))
                )
            )

    def permits(self, kind: str, name: str, scope: str) -> bool:
        known = self.known_tools if kind == "tool" else self.known_skills
        if name not in known:
            raise ValueError(
                CAPABILITY_PROMPTS.unknown_condition.format(kind=kind, name=name)
            )
        selected, direct, reachable = (
            (self.selected_tools, self.direct_tools, self.reachable_tools)
            if kind == "tool"
            else (self.selected_skills, self.direct_skills, self.reachable_skills)
        )
        return (
            name
            in {"direct": direct, "reachable": reachable, "selected": selected}[scope]
        )


def has_conditions(text: str) -> bool:
    return bool(_TOKENS.search(text))


def render_conditions(text: str, context: CapabilityPromptContext) -> str:
    """Only reserved tags outside code spans/fences and comments are interpreted."""
    protected = literal_spans(text)
    stack: list[tuple[str, bool]] = []
    output = []
    position = 0
    for match in _TOKENS.finditer(text):
        if any(start <= match.start() < end for start, end in protected):
            continue
        if all(enabled for _, enabled in stack):
            output.append(text[position : match.start()])
        tag = match.group()
        if tag.startswith("</"):
            kind = tag[5:-1]
            if not stack or stack[-1][0] != kind:
                raise _invalid(tag)
            stack.pop()
        else:
            kind, names, scope = _parse_condition(tag)
            # Evaluate every name so a match cannot hide an unknown capability.
            matches = [context.permits(kind, name, scope) for name in names]
            stack.append((kind, any(matches)))
        position = match.end()
    if stack:
        raise _invalid(CAPABILITY_PROMPTS.unclosed_condition.format(kind=stack[-1][0]))
    output.append(text[position:])
    return "".join(output)


def _parse_condition(tag: str) -> tuple[str, tuple[str, ...], str]:
    try:
        element = ET.fromstring(tag + "</" + tag[1:].split()[0] + ">")
    except ET.ParseError as error:
        raise _invalid(tag) from error
    names = (
        (element.attrib["name"],)
        if "name" in element.attrib
        else tuple(element.get("any-of", "").split())
    )
    scope = element.get("scope", "direct")
    if (
        element.tag not in {"if-tool", "if-skill"}
        or set(element.attrib) - {"name", "any-of", "scope"}
        or ("name" in element.attrib) == ("any-of" in element.attrib)
        or not names
        or not all(names)
        or scope not in {"direct", "reachable", "selected"}
    ):
        raise _invalid(tag)
    return element.tag[3:], names, scope


def _invalid(tag: str) -> ValueError:
    return ValueError(CAPABILITY_PROMPTS.invalid_condition.format(tag=tag))
