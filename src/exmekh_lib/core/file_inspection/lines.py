"""Physical CR/LF lines, independent of a parser's Unicode newline rules."""

import re
from bisect import bisect_right


class SourceLines:
    """Translate character positions into 1-based physical line numbers."""

    def __init__(self, text: str) -> None:
        self._starts = [0, *(match.end() for match in re.finditer(r"\r\n|\r|\n", text))]

    def line_number(self, character_offset: int) -> int:
        return bisect_right(self._starts, character_offset)


def physical_lines(text: str) -> list[str]:
    """A trailing CR/LF ends a line without creating an extra physical line."""
    if not text:
        return []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines[-1]:
        lines.pop()
    return lines
