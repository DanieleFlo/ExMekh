import re

from qs.config.validation import VALIDATION_PATTERNS

_REFERENCE_PATTERN = re.compile(VALIDATION_PATTERNS.capability_reference)


def parse_references(body: str) -> tuple[str, tuple[str, ...]]:
    references: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        references.append(name)
        return name

    return _REFERENCE_PATTERN.sub(replace, body), tuple(dict.fromkeys(references))
