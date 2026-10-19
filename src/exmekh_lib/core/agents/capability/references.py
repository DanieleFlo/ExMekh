import re

from exmekh_lib.config.validation import VALIDATION_PATTERNS
from exmekh_lib.core.agents.capability.markdown_literals import literal_spans

_REFERENCE_PATTERN = re.compile(VALIDATION_PATTERNS.capability_reference)


def parse_references(body: str) -> tuple[str, tuple[str, ...]]:
    references: list[str] = []
    protected = literal_spans(body)

    def replace(match: re.Match[str]) -> str:
        if any(start <= match.start() < end for start, end in protected):
            return match.group()
        name = match.group(1)
        references.append(name)
        return name

    return _REFERENCE_PATTERN.sub(replace, body), tuple(dict.fromkeys(references))
