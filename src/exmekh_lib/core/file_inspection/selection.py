"""Validate and rank bounded navigation selections."""

import re

from exmekh_lib.config.file_inspection import FileInspectionSettings

from .errors import FileInspectionError


def validate_selection(
    detailed: bool,
    max_items: int,
    offset: int,
    query: str | None,
    settings: FileInspectionSettings,
) -> None:
    if not isinstance(detailed, bool):
        raise FileInspectionError("detailed must be a boolean.")
    if (
        isinstance(max_items, bool)
        or not isinstance(max_items, int)
        or not 1 <= max_items <= settings.max_items
    ):
        raise FileInspectionError(
            f"max_items must be an integer between 1 and {settings.max_items}."
        )
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise FileInspectionError("offset must be a nonnegative integer.")
    if query is not None and (
        not isinstance(query, str)
        or len(query) > settings.max_query_chars
        or not query_terms(query)
    ):
        raise FileInspectionError(
            f"query must contain searchable words and at most {settings.max_query_chars} characters."
        )


def query_terms(value: str) -> set[str]:
    # Split camelCase and snake_case identifiers into searchable words.
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", value).replace("_", " ")
    return set(re.findall(r"[^\W_]+", expanded.casefold()))
