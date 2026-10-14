"""Validate and rank bounded navigation selections."""

import re

from exmekh_lib.config.file_inspection import (
    FILE_INSPECTION_TEXT,
    FileInspectionSettings,
)

from .errors import FileInspectionError


def validate_selection(
    detailed: bool,
    max_items: int,
    offset: int,
    query: str | None,
    settings: FileInspectionSettings,
) -> None:
    if not isinstance(detailed, bool):
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.detailed_boolean, code="invalid_input"
        )
    if (
        isinstance(max_items, bool)
        or not isinstance(max_items, int)
        or not 1 <= max_items <= settings.max_items
    ):
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.max_items_range.format(max_items=settings.max_items),
            code="invalid_input",
        )
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.offset_nonnegative, code="invalid_input"
        )
    if query is not None and (
        not isinstance(query, str)
        or len(query) > settings.max_query_chars
        or not query_terms(query)
    ):
        raise FileInspectionError(
            FILE_INSPECTION_TEXT.query_required.format(
                max_query_chars=settings.max_query_chars
            ),
            code="invalid_query",
        )


def query_terms(value: str) -> set[str]:
    # Split camelCase and snake_case identifiers into searchable words.
    expanded = re.sub(r"([a-z])([A-Z])", r"\1 \2", value).replace("_", " ")
    return set(re.findall(r"[^\W_]+", expanded.casefold()))
