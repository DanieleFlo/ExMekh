import re

_REFERENCE_PATTERN = re.compile(r"{{\s*!([A-Za-z0-9][A-Za-z0-9_-]*)\s*}}")


def parse_references(body: str) -> tuple[str, tuple[str, ...]]:
    references: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        references.append(name)
        return name

    return _REFERENCE_PATTERN.sub(replace, body), tuple(dict.fromkeys(references))
