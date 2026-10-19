"""Locate code and comments shared by conditional and reference rendering."""

import re

from exmekh_lib.config.capability_prompts import CAPABILITY_PROMPTS

_LITERAL_START = re.compile(CAPABILITY_PROMPTS.literal_start)


def literal_spans(text: str) -> tuple[tuple[int, int], ...]:
    spans = []
    position = 0
    while match := _LITERAL_START.search(text, position):
        if fence := match.group("fence"):
            closing = re.compile(
                rf"(?m)^[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*(?:\r?\n|$)"
            ).search(text, match.end())
            end = closing.end() if closing else len(text)
        elif match.group("comment"):
            closing_position = text.find("-->", match.end())
            end = closing_position + 3 if closing_position >= 0 else len(text)
        else:
            delimiter = match.group("inline")
            closing = re.compile(rf"(?<!`){delimiter}(?!`)").search(text, match.end())
            if closing is None:
                position = match.end()
                continue
            end = closing.end()
        spans.append((match.start(), end))
        position = end
    return tuple(spans)
