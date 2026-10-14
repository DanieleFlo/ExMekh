"""Compose unique literal selections against one immutable file snapshot."""

from dataclasses import dataclass

from exmekh_lib.config.file_editing import FILE_EDITING
from exmekh_lib.core.models.edit_document import EditElement, literal_selection_offset

from .errors import FileInspectionError


@dataclass(frozen=True, slots=True)
class _Replacement:
    start: int
    end: int
    text: str


def _require_literal_selection(text: str, old: str) -> int:
    offset = literal_selection_offset(text, old)
    if offset is not None:
        return offset
    if not old:
        raise FileInspectionError(FILE_EDITING.selection_empty, code="old_text_empty")
    first = text.find(old)
    if first < 0:
        raise FileInspectionError(
            FILE_EDITING.selection_missing, code="old_text_not_found"
        )
    if text.find(old, first + 1) >= 0:
        raise FileInspectionError(
            FILE_EDITING.selection_ambiguous, code="old_text_ambiguous"
        )
    raise FileInspectionError(
        FILE_EDITING.selection_midline, code="old_text_not_at_line_start"
    )


def select_replacements(text: str, elements: tuple[EditElement, ...]) -> str:
    if not elements:
        raise FileInspectionError(
            FILE_EDITING.selection_required, code="invalid_edit_document"
        )
    if not text:
        if len(elements) != 1 or elements[0].old_text:
            raise FileInspectionError(
                FILE_EDITING.selection_error, code="invalid_edit_document"
            )
        return elements[0].replacement
    replacements: list[_Replacement] = []
    for element in elements:
        old = element.old_text
        offset = _require_literal_selection(text, old)
        replacement = _Replacement(offset, offset + len(old), element.replacement)
        if any(
            max(replacement.start, other.start) < min(replacement.end, other.end)
            for other in replacements
        ):
            raise FileInspectionError(
                FILE_EDITING.conflict_error, code="overlapping_edits"
            )
        replacements.append(replacement)
    result = text
    for replacement in sorted(replacements, key=lambda item: item.start, reverse=True):
        result = (
            result[: replacement.start] + replacement.text + result[replacement.end :]
        )
    return result
