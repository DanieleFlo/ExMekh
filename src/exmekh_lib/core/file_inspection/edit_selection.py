"""Literal edit selection and reserved marker validation, independent of transport."""

import re

from exmekh_lib.config.file_editing import FILE_EDITING

from .errors import FileInspectionError


def line_starts(text: str) -> list[int]:
    if not text:
        return []
    return [
        0,
        *(
            match.end()
            for match in re.finditer(r"\r\n|\r|\n", text)
            if match.end() < len(text)
        ),
    ]


def select_replacement(
    text: str, old_text: str, new_text: str, *, exists: bool
) -> str | None:
    """Return exact replacement content, or None for an explicit file removal."""
    if not isinstance(old_text, str) or not isinstance(new_text, str):
        raise FileInspectionError(
            "old_text and new_text must be strings; no coercion is performed."
        )
    if "\x00" in old_text or "\x00" in new_text:
        raise FileInspectionError(
            "NUL is not supported by the interactive text contract."
        )
    marker = re.fullmatch(FILE_EDITING.line_pattern, old_text)
    all_text = old_text == FILE_EDITING.all_marker
    delete = new_text == FILE_EDITING.delete_marker
    if (
        (
            re.search(FILE_EDITING.reserved_marker_pattern, old_text)
            and not (marker or all_text)
        )
        or (re.search(FILE_EDITING.reserved_marker_pattern, new_text) and not delete)
        or (delete and not (all_text and exists))
    ):
        raise FileInspectionError(FILE_EDITING.invalid_marker)
    if not exists and not (all_text or old_text == FILE_EDITING.first_line_marker):
        raise FileInspectionError(FILE_EDITING.creation_required)
    if delete:
        return None
    if all_text:
        return new_text
    starts = line_starts(text)
    if marker:
        # Avoid constructing enormous integer values from untrusted marker text.
        digits = marker.group(1)
        if len(digits) > len(str(len(starts) + 1)):
            raise FileInspectionError(
                f"Insertion line must be in 1..{len(starts) + 1}."
            )
        line = int(digits)
        if line > len(starts) + 1:
            raise FileInspectionError(
                f"Insertion line must be in 1..{len(starts) + 1}."
            )
        offset = starts[line - 1] if line <= len(starts) else len(text)
        return text[:offset] + new_text + text[offset:]
    if not old_text:
        raise FileInspectionError(FILE_EDITING.selection_error)
    offset = text.find(old_text)
    if offset < 0 or text.find(old_text, offset + 1) >= 0 or offset not in starts:
        raise FileInspectionError(FILE_EDITING.selection_error)
    return text[:offset] + new_text + text[offset + len(old_text) :]
